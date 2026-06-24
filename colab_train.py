# ============================================================
# colab_train.py  Single Colab Entry Point (Resilient)
# ============================================================
# Run this ONE script in Google Colab to do EVERYTHING:
#   1. Mount Google Drive
#   2. Install all dependencies
#   3. Setup data directories
#   4. Detect & display existing checkpoints
#   5. Train with auto-resume from last checkpoint
#   6. Evaluate on test set
#   7. Save everything to Google Drive
#
# USAGE IN COLAB (just 2 cells):
# --------------------------------------------
# Cell 1:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/Nit_trichy
#
# Cell 2:
#   !python colab_train.py --mode lora --epochs 5
#
# IF COLAB DISCONNECTS  just re-run Cell 1 & Cell 2!
# Training auto-resumes from the last saved checkpoint.
# ============================================================

import os
import sys
import time
import shutil
import subprocess
import argparse
from datetime import datetime

# ------------------------------------------------------------
#  PHASE 0: Parse Arguments (before any heavy imports)
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="Colab NER Training (Resilient)")
parser.add_argument("--mode", type=str, choices=["lora", "full", "both"], default="lora",
                    help="Training mode: lora (fast), full (all params), both (compare)")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
parser.add_argument("--max-hours", type=float, default=1.9,
                    help="Max training hours before graceful stop (default: 1.9)")
parser.add_argument("--save-steps", type=int, default=200,
                    help="Save checkpoint every N steps (default: 200)")
parser.add_argument("--no-resume", action="store_true",
                    help="Force fresh start (ignore existing checkpoints)")
parser.add_argument("--skip-install", action="store_true",
                    help="Skip dependency installation (if already installed)")
parser.add_argument("--smoke-test", action="store_true",
                    help="Quick smoke test (1 epoch, save every 20 steps)")
args = parser.parse_args()

# ------------------------------------------------------------
#  Pretty Printing (before rich is installed)
# ------------------------------------------------------------

def print_box(title, lines, char="-"):
    """Print a boxed section (works before rich is installed)."""
    width = 62
    print(f"\n{'-' + char * width + '-'}")
    print(f"-  {title:<{width - 2}}-")
    print(f"{'' + char * width + ''}")
    for line in lines:
        print(f"-  {line:<{width - 2}}-")
    print(f"{'-' + char * width + '-'}\n")


def print_step(step_num, total, description, status=""):
    """Print a numbered step with status."""
    bar_width = 20
    filled = int(bar_width * step_num / total)
    bar = "-" * filled + "" * (bar_width - filled)
    print(f"  [{bar}] Step {step_num}/{total}: {description} {status}")


# ------------------------------------------------------------
#  PHASE 1: Environment Setup & Validation
# ------------------------------------------------------------

SESSION_START = time.time()

print_box("NER Fine-Tuning: Colab-Resilient Training", [
    f"Mode:       {args.mode.upper()}",
    f"Epochs:     {args.epochs}",
    f"Max Hours:  {args.max_hours}",
    f"Save Steps: {args.save_steps}",
    f"Resume:     {'Disabled (--no-resume)' if args.no_resume else 'Auto-resume enabled'}",
    f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
])

# -- Check we're on Colab --
IS_COLAB = os.path.exists("/content")
if IS_COLAB:
    print("  Running on Google Colab")
    
    # Check Drive is mounted
    if not os.path.exists("/content/drive"):
        print("  Google Drive not mounted! Run this first:")
        print("     from google.colab import drive")
        print("     drive.mount('/content/drive')")
        sys.exit(1)
    else:
        print("  Google Drive is mounted")
else:
    print("    Running locally (not on Colab)")

# -- Check GPU --
print("\n   Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("    No GPU detected! Training will be very slow.")
        print("     Go to Runtime  Change runtime type  T4 GPU")
except ImportError:
    print("    PyTorch not installed yet (will be installed in next step)")


# ------------------------------------------------------------
#  PHASE 2: Install Dependencies
# ------------------------------------------------------------

if not args.skip_install:
    print("\n" + "=" * 62)
    print("   PHASE 2: Installing Dependencies (~2-3 min)")
    print("=" * 62 + "\n")
    
    install_start = time.time()
    
    packages = [
        ("torch transformers", "Core ML (PyTorch + HuggingFace)"),
        ("peft", "LoRA/QLoRA Fine-Tuning"),
        ("bitsandbytes", "4-bit Quantization"),
        ("accelerate", "Multi-GPU / Mixed Precision"),
        ("datasets seqeval scikit-learn", "Evaluation Metrics"),
        ("rich tqdm matplotlib pandas", "UI & Visualization"),
    ]
    
    for i, (pkgs, desc) in enumerate(packages, 1):
        print(f"  [{i}/{len(packages)}] Installing {desc}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q"] + pkgs.split(),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"      Warning: {result.stderr[:200]}")
        else:
            print(f"    Done")
    
    install_time = time.time() - install_start
    print(f"\n  All dependencies installed in {install_time:.0f}s")
else:
    print("\n    Skipping installation (--skip-install)")

# -- Re-import torch after potential install --
import torch
print(f"\n   PyTorch {torch.__version__}")
print(f"   CUDA: {'' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")


# ------------------------------------------------------------
#  PHASE 3: Setup Data Directories
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("  PHASE 3: Setting Up Data & Output Directories")
print("=" * 62 + "\n")

# -- Create data directory structure --
os.makedirs('data/raw', exist_ok=True)

# -- Copy data files to expected location --
data_files_ok = True
for fname in ['train.txt', 'valid.txt', 'test.txt']:
    src = fname
    dst = f'data/raw/{fname}'
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  Copied {src}  {dst} ({size_mb:.1f} MB)")
    elif os.path.exists(dst):
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"  {dst} exists ({size_mb:.1f} MB)")
    elif os.path.exists(src):
        size_mb = os.path.getsize(src) / (1024 * 1024)
        print(f"  {src} exists ({size_mb:.1f} MB)")
    else:
        print(f"  {fname} NOT FOUND! Please upload it to the project directory.")
        data_files_ok = False

if not data_files_ok:
    print("\n  Missing data files. Please upload train.txt, valid.txt, test.txt")
    sys.exit(1)

# -- Create output directories --
DRIVE_RESULTS = "/content/drive/MyDrive/Nit_trichy/results"
if IS_COLAB:
    for subdir in ["lora", "full_ft"]:
        path = os.path.join(DRIVE_RESULTS, subdir)
        os.makedirs(path, exist_ok=True)
    print(f"\n    Output dir (Drive): {DRIVE_RESULTS}")
else:
    os.makedirs("results/lora", exist_ok=True)
    os.makedirs("results/full_ft", exist_ok=True)
    print(f"\n   Output dir (local): ./results/")


# ------------------------------------------------------------
#  PHASE 4: Scan Existing Checkpoints
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("   PHASE 4: Scanning for Existing Checkpoints")
print("=" * 62 + "\n")

from src.trainer import list_all_checkpoints

def show_checkpoints(mode_name, dir_path):
    """Display checkpoint info for a training mode."""
    checkpoints = list_all_checkpoints(dir_path)
    if checkpoints:
        print(f"  {mode_name}: {len(checkpoints)} checkpoint(s) found")
        for ckpt in checkpoints:
            name = os.path.basename(ckpt["path"])
            epoch = ckpt.get("epoch", "?")
            loss = ckpt.get("last_loss", "?")
            print(f"     - {name}  (epoch {epoch}, loss {loss}, {ckpt['size_mb']:.0f} MB)")
        
        latest = checkpoints[-1]
        print(f"    Will resume from: {os.path.basename(latest['path'])}")
        return True
    else:
        print(f"  {mode_name}: No checkpoints  will start fresh")
        return False

if IS_COLAB:
    base = DRIVE_RESULTS
else:
    base = "results"

has_lora_ckpt = False
has_full_ckpt = False

if args.mode in ["lora", "both"]:
    has_lora_ckpt = show_checkpoints("LoRA", os.path.join(base, "lora"))

if args.mode in ["full", "both"]:
    has_full_ckpt = show_checkpoints("Full FT", os.path.join(base, "full_ft"))

if args.no_resume:
    print("\n    --no-resume flag set: will start training from scratch")


# ------------------------------------------------------------
#  PHASE 5: Run Training Pipeline
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("  PHASE 5: Starting Training Pipeline")
print("=" * 62)

# Now import the heavy modules (after dependencies are installed)
from rich.console import Console
console = Console()

from src.config import Config

# -- Build config --
config = Config()
config.training.num_epochs = args.epochs
config.colab.max_training_hours = args.max_hours
config.colab.save_steps = args.save_steps
config.colab.auto_resume = not args.no_resume

if args.lr:
    config.training.learning_rate = args.lr
if args.batch_size:
    config.training.per_device_train_batch_size = args.batch_size
if args.smoke_test:
    config.training.num_epochs = 1
    config.colab.save_steps = 20
    console.print("  [yellow]SMOKE TEST  1 epoch, save every 20 steps[/yellow]\n")

# -- Build the CLI command to forward to run_training.py --
cmd = [sys.executable, "run_training.py", "--mode", args.mode, "--epochs", str(config.training.num_epochs)]

if args.no_resume:
    cmd.append("--no-resume")
if args.max_hours is not None:
    cmd.extend(["--max-hours", str(args.max_hours)])
if args.smoke_test:
    cmd.extend(["--save-steps", "20"])
elif args.save_steps is not None:
    cmd.extend(["--save-steps", str(args.save_steps)])
if args.lr:
    cmd.extend(["--lr", str(args.lr)])
if args.batch_size:
    cmd.extend(["--batch-size", str(args.batch_size)])
if args.smoke_test:
    cmd.append("--smoke-test")

console.print(f"\n   [dim]Command: {' '.join(cmd)}[/dim]\n")

# -- Run the training pipeline --
training_start = time.time()
result = subprocess.run(cmd)
training_time = time.time() - training_start

if result.returncode != 0:
    console.print(f"\n  [bold red]Training exited with code {result.returncode}[/bold red]")
    console.print("     Check the error messages above for details.\n")
else:
    console.print(f"\n  [bold green]Training pipeline completed successfully![/bold green]")
    console.print(f"     Total time: {training_time/60:.1f} min")


# ------------------------------------------------------------
#  PHASE 6: Post-Training Summary
# ------------------------------------------------------------

total_session_time = time.time() - SESSION_START

print("\n" + "-" * 62)
print("   SESSION SUMMARY")
print("-" * 62)
print(f"  Total Session Time:  {total_session_time/60:.1f} min ({total_session_time/3600:.2f} hours)")
print(f"  Mode:                {args.mode.upper()}")
print(f"  Epochs Configured:   {config.training.num_epochs}")
print(f"  Max Hours:           {args.max_hours}")

# -- Show final checkpoint state --
print(f"\n  Final Checkpoint State:")
if args.mode in ["lora", "both"]:
    show_checkpoints("LoRA", os.path.join(base, "lora"))
if args.mode in ["full", "both"]:
    show_checkpoints("Full FT", os.path.join(base, "full_ft"))

# -- Show saved results --
print(f"\n   Results Location:")
if IS_COLAB:
    print(f"     {DRIVE_RESULTS}")
    print(f"     (Saved to Google Drive  persists across sessions)")
else:
    print(f"     ./results/")
    print(f"     (Saved locally)")

# -- Next steps --
print(f"\n  Next Steps:")
print(f"      If training was interrupted, just re-run this script!")
print(f"      Training will auto-resume from the last checkpoint")
print(f"      To force a fresh start, use --no-resume")
print(f"      To download results: Cell 8 in COLAB_SETUP_GUIDE.md")
print("-" * 62 + "\n")

# ============================================================
# run_training.py — Main Entry Point (Colab-Resilient)
# ============================================================
# This is the single script you run to execute the full pipeline:
#   1. Load & preprocess data
#   2. Load model (LoRA or Full FT)
#   3. Train with checkpoint-based resume
#   4. Evaluate on test set (P, R, F1)
#   5. Save everything (model, metrics, predictions)
#
# Usage:
#   python run_training.py --mode lora                    (fresh start)
#   python run_training.py --mode lora --no-resume        (force fresh start)
#   python run_training.py --mode lora --max-hours 3.0    (custom time limit)
#   python run_training.py --mode lora --save-steps 100   (save more frequently)
#   python run_training.py --mode full                    (full fine-tuning)
#   python run_training.py --mode both                    (run both & compare)
#
# On Colab disconnect: just re-run the same command with --resume!
# ============================================================

import os
import sys
import json
import time
import argparse
import warnings
from datetime import datetime

# Set utf-8 encoding for rich on Windows (Bug 22)
os.environ["PYTHONIOENCODING"] = "utf-8"

warnings.filterwarnings("ignore")

import torch
from rich.console import Console
from rich.panel import Panel

# -- Import our custom modules --
from src.config import Config
from src.data_utils import load_and_prepare_data, verify_dataset_sample
from src.model_utils import load_model_for_ner, load_tokenizer, count_parameters
from src.trainer import train_model, list_all_checkpoints
from src.evaluator import run_full_evaluation, compare_results

console = Console()


def print_header():
    """Print the main project header with system info."""
    header = \"\"\"
------------------------------------------------------------------
   NER Fine-Tuning: Abbreviation/Acronym Detection           
   Task:   Sequence Labeling (BIO tags)                           
   Labels: O, B-short, I-short, B-long, I-long                   
   Method: LLM Fine-Tuning with LoRA / Full FT                   
   Colab-Resilient with Checkpoint Resume                      
------------------------------------------------------------------
\"\"\"
    console.print(header, style="bold cyan")
    
    # -- Print system info --
    console.print("  [bold]System Information:[/bold]")
    console.print(f"    Python:    {sys.version.split()[0]}")
    console.print(f"    PyTorch:   {torch.__version__}")
    console.print(f"    CUDA:      {'Available' if torch.cuda.is_available() else 'Not Available'}")
    
    if torch.cuda.is_available():
        console.print(f"    GPU:       {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"    GPU Memory:{gpu_mem:.1f} GB")
    
    # -- Detect environment --
    is_colab = os.path.exists("/content/drive")
    if is_colab:
        console.print(f"    Env:       [bold green]Google Colab (Drive mounted)[/bold green]")
    else:
        console.print(f"    Env:       Local machine")
    
    console.print("")


def print_checkpoint_status(mode: str, config):
    """Show existing checkpoint status before training starts."""
    is_colab = os.path.exists("/content/drive")
    if is_colab:
        base = config.colab.drive_output_dir
    else:
        base = "results"
    
    output_dir = os.path.join(base, "lora" if mode == "lora" else "full_ft")
    checkpoints = list_all_checkpoints(output_dir)
    
    if checkpoints:
        console.print(f"\n  [bold]Existing checkpoints for {mode.upper()}:[/bold]")
        for ckpt in checkpoints:
            name = os.path.basename(ckpt["path"])
            console.print(
                f"    └- {name}  "
                f"(step {ckpt['step']:,}, epoch {ckpt.get('epoch', '?')}, "
                f"loss {ckpt.get('last_loss', '?')}, {ckpt['size_mb']:.0f} MB)"
            )
        console.print("")
    else:
        console.print(f"\n  [dim]No existing checkpoints for {mode.upper()}[/dim]\n")


def run_pipeline(mode: str, config: Config):
    """
    Run the full training + evaluation pipeline for a given mode.
    
    Args:
        mode: 'lora' or 'full'
        config: Configuration object
        
    Returns:
        (training_stats, eval_metrics) tuple
    """
    total_start = time.time()
    
    # -- Select model based on mode --
    model_name = config.training.model_name_lora if mode == "lora" else config.training.model_name_full
    
    # -- Output dir (Drive-aware) --
    is_colab = os.path.exists("/content/drive")
    if is_colab:
        output_dir = os.path.join(config.colab.drive_output_dir, "lora" if mode == "lora" else "full_ft")
    else:
        output_dir = config.training.output_dir_lora if mode == "lora" else config.training.output_dir_full
    
    # ----------------------------------------------------------
    #  STEP 1: Load & Preprocess Data
    # ----------------------------------------------------------
    
    # Load tokenizer first (needed for data preprocessing)
    tokenizer = load_tokenizer(model_name)
    
    # Load and preprocess all data splits
    (train_dataset, valid_dataset, test_dataset,
     label2id, id2label, test_sentences, data_stats) = load_and_prepare_data(config, tokenizer)
    
    # Quick verification
    verify_dataset_sample(train_dataset, tokenizer, id2label, n=2)
    
    # ----------------------------------------------------------
    #  STEP 2: Load Model
    # ----------------------------------------------------------
    
    lora_params = {
        'r': config.lora.r,
        'lora_alpha': config.lora.lora_alpha,
        'lora_dropout': config.lora.lora_dropout,
        'bias': config.lora.bias,
        'target_modules': config.lora.target_modules,
    }
    
    model, tokenizer = load_model_for_ner(
        model_name=model_name,
        label2id=label2id,
        id2label=id2label,
        mode=mode,
        lora_config_params=lora_params if mode == "lora" else None,
        tokenizer=tokenizer,
    )
    
    # ----------------------------------------------------------
    #  STEP 3: Train (with auto-resume)
    # ----------------------------------------------------------
    
    training_stats, trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        id2label=id2label,
        config=config,
        mode=mode,
    )
    
    # ----------------------------------------------------------
    #  STEP 4: Evaluate on Test Set
    # ----------------------------------------------------------
    
    # Only run full evaluation if training is complete (>99% done)
    pct_complete = training_stats.get("pct_complete", 100)
    
    if pct_complete >= 99.5:
        console.print("\n  [bold]Training complete! Running full test evaluation...[/bold]\n")
        
        eval_metrics = run_full_evaluation(
            trainer=trainer,
            test_dataset=test_dataset,
            test_sentences=test_sentences,
            id2label=id2label,
            mode=mode,
            output_dir=output_dir,
            training_stats=training_stats,
        )
    else:
        console.print(
            f"\n   [bold yellow]Training {pct_complete:.0f}% complete.[/bold yellow] "
            f"     Output saved to: [green]{output_dir}[/green]\n"
            f"     Auto-resume is ON. Run with [cyan]--no-resume[/cyan] to start fresh.\n"
        )
        eval_metrics = {"status": "incomplete", "pct_complete": pct_complete}
    
    # -- Total pipeline time --
    total_time = time.time() - total_start
    console.print(f"\n   Total pipeline time ({mode.upper()}): "
                  f"[bold cyan]{total_time/60:.1f} min[/bold cyan]\n")
    
    return training_stats, eval_metrics


def main():
    """Main entry point — parse args and run the pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for Abbreviation/Acronym NER (Colab-Resilient)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full", "both"],
        default="lora",
        help="Training mode: 'lora' (QLoRA, fast), 'full' (all params), 'both' (run both & compare)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-device batch size"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start fresh"
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=None,
        help="Max training hours before graceful stop (default: 3.5)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: 200)"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test (1 epoch, save every 20 steps)"
    )
    
    args = parser.parse_args()
    
    # -- Initialize config --
    config = Config()
    
    # -- Apply CLI overrides --
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.no_resume:
        config.colab.auto_resume = False
    if args.max_hours is not None:
        config.colab.max_training_hours = args.max_hours
    if args.save_steps is not None:
        config.colab.save_steps = args.save_steps
    if args.smoke_test:
        config.training.num_epochs = 1
        config.colab.save_steps = 20
        console.print("  [yellow]SMOKE TEST MODE — 1 epoch, save every 20 steps[/yellow]\n")
    
    # -- Print header --
    print_header()
    
    # -- Create output directories --
    os.makedirs("results/lora", exist_ok=True)
    os.makedirs("results/full_ft", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    # Also create Drive dirs if on Colab
    if os.path.exists("/content/drive"):
        os.makedirs(os.path.join(config.colab.drive_output_dir, "lora"), exist_ok=True)
        os.makedirs(os.path.join(config.colab.drive_output_dir, "full_ft"), exist_ok=True)
    
    # ----------------------------------------------------------
    #  Show Checkpoint Status
    # ----------------------------------------------------------
    
    if args.mode in ["lora", "both"]:
        print_checkpoint_status("lora", config)
    if args.mode in ["full", "both"]:
        print_checkpoint_status("full", config)
    
    # ----------------------------------------------------------
    #  Run Pipeline(s)
    # ----------------------------------------------------------
    
    lora_metrics = None
    full_metrics = None
    
    if args.mode in ["lora", "both"]:
        console.print("\n" + "-" * 62)
        console.print("  [bold]RUNNING LoRA FINE-TUNING (QLoRA 4-bit)[/bold]")
        console.print("-" * 62)
        
        config.print_summary("lora")
        lora_stats, lora_metrics = run_pipeline("lora", config)
    
    if args.mode in ["full", "both"]:
        console.print("\n" + "-" * 62)
        console.print("  [bold]RUNNING FULL FINE-TUNING[/bold]")
        console.print("-" * 62)
        
        config.print_summary("full")
        full_stats, full_metrics = run_pipeline("full", config)
    
    # ----------------------------------------------------------
    #  Compare Results (if both modes were run)
    # ----------------------------------------------------------
    
    if lora_metrics and full_metrics:
        # Only compare if both completed
        lora_complete = lora_metrics.get("status") != "incomplete"
        full_complete = full_metrics.get("status") != "incomplete"
        
        if lora_complete and full_complete:
            lora_metrics["training_time_minutes"] = lora_stats.get("training_time_minutes", 0)
            full_metrics["training_time_minutes"] = full_stats.get("training_time_minutes", 0)
            compare_results(lora_metrics, full_metrics)
            
            # Save comparison
            comparison = {
                "lora": lora_metrics,
                "full_ft": full_metrics,
            }
            with open("results/comparison_report.json", 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            console.print("  Comparison saved to: [green]results/comparison_report.json[/green]\n")
        else:
            console.print("\n   [yellow]Comparison skipped — not all training runs completed.[/yellow]")
            console.print("     Resume incomplete runs and then compare.\n")
    
    # -- Final message --
    console.print("----------------------------------------------------------------")
    console.print("-         [bold green]SESSION DONE! Training & Evaluation Complete[/bold green]    -")
    console.print("----------------------------------------------------------------")
    console.print("")
    console.print("  [bold]Tips:[/bold]")
    console.print("     • If training was interrupted, run without [cyan]--no-resume[/cyan] to continue automatically")
    console.print("     • Logs and checkpoints are safe in your Drive. (Colab) or ./results/ (local)")
    console.print("     • Use [cyan]--max-hours 2.0[/cyan] for shorter Colab sessions")
    console.print("")


if __name__ == "__main__":
    main()

# ============================================================
# trainer.py — Training Loop with Colab-Resilient Checkpointing
# ============================================================
# Handles:
#   1. HuggingFace Trainer setup for both LoRA and Full FT
#   2. Custom metrics computation (P, R, F1) during training
#   3. Rich progress display with ETA & loss tracking
#   4. TimeLimitCallback for graceful Colab shutdown
#   5. Auto-resume from latest checkpoint
#   6. Step-based checkpoint saving to Google Drive
#   7. GPU memory monitoring throughout training
# ============================================================

import os
import re
import json
import time
import math
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ------------------------------------------------------------
#  Custom Metrics Computation
# ------------------------------------------------------------

def build_compute_metrics_fn(id2label: Dict[int, str]):
    """
    Build a compute_metrics function for the HF Trainer.
    
    This function:
      1. Takes raw predictions (logits) and labels
      2. Converts predicted token IDs to BIO tag strings
      3. Computes entity-level Precision, Recall, F1 using seqeval
    
    Args:
        id2label: Mapping from integer IDs to tag strings
        
    Returns:
        compute_metrics function compatible with HF Trainer
    """
    
    def compute_metrics(eval_preds):
        """
        Compute entity-level P, R, F1 from model predictions.
        
        Args:
            eval_preds: EvalPrediction object with .predictions and .label_ids
            
        Returns:
            Dict with precision, recall, f1, and accuracy
        """
        predictions, labels = eval_preds
        
        # predictions shape: (batch_size, seq_len, num_labels)
        # Take argmax to get predicted label IDs
        predictions = np.argmax(predictions, axis=2)
        
        # -- Convert numeric IDs back to BIO tag strings --
        # Only include tokens where label != -100 (ignore subwords & padding)
        true_labels = []
        pred_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            true_sent = []
            pred_sent = []
            
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != -100:
                    # Valid token — convert to tag string
                    true_sent.append(id2label.get(label_id, "O"))
                    pred_sent.append(id2label.get(pred_id, "O"))
            
            true_labels.append(true_sent)
            pred_labels.append(pred_sent)
        
        # -- Compute entity-level metrics using seqeval --
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        # Token-level accuracy (for reference)
        correct = 0
        total = 0
        for true_sent, pred_sent in zip(true_labels, pred_labels):
            for t, p in zip(true_sent, pred_sent):
                total += 1
                if t == p:
                    correct += 1
        accuracy = correct / max(total, 1)
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }
    
    return compute_metrics


# ------------------------------------------------------------
#  Checkpoint Discovery (Auto-Resume)
# ------------------------------------------------------------

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint-XXXX directory in the output directory.
    
    Checkpoints are named 'checkpoint-<global_step>' by HuggingFace.
    We find the one with the highest step number.
    
    Args:
        output_dir: Directory where checkpoints are saved
        
    Returns:
        Full path to the latest checkpoint, or None if no checkpoints exist
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoint_dirs = []
    for entry in os.listdir(output_dir):
        full_path = os.path.join(output_dir, entry)
        if os.path.isdir(full_path) and re.match(r'^checkpoint-\d+$', entry):
            step_num = int(entry.split('-')[1])
            checkpoint_dirs.append((step_num, full_path))
    
    if not checkpoint_dirs:
        return None
    
    # Return the one with the highest step number
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_dirs[0][1]


def get_checkpoint_info(checkpoint_path: str) -> Dict:
    """
    Extract metadata from a checkpoint directory.
    
    Returns dict with step number, epoch info, and trainer state if available.
    """
    info = {
        "path": checkpoint_path,
        "step": int(os.path.basename(checkpoint_path).split('-')[1]),
        "size_mb": 0,
    }
    
    # Calculate checkpoint size
    for root, dirs, files in os.walk(checkpoint_path):
        for f in files:
            info["size_mb"] += os.path.getsize(os.path.join(root, f))
    info["size_mb"] = round(info["size_mb"] / (1024 ** 2), 1)
    
    # Try to read trainer_state.json for epoch info
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            info["epoch"] = round(state.get("epoch", 0), 2)
            info["best_metric"] = state.get("best_metric")
            info["total_flos"] = state.get("total_flos", 0)
            
            # Get last logged loss
            log_history = state.get("log_history", [])
            for entry in reversed(log_history):
                if "loss" in entry:
                    info["last_loss"] = round(entry["loss"], 4)
                    break
        except Exception:
            pass
    
    return info


def list_all_checkpoints(output_dir: str) -> list:
    """List all checkpoints with their metadata, sorted by step."""
    if not os.path.exists(output_dir):
        return []
    
    checkpoints = []
    for entry in os.listdir(output_dir):
        full_path = os.path.join(output_dir, entry)
        if os.path.isdir(full_path) and re.match(r'^checkpoint-\d+$', entry):
            checkpoints.append(get_checkpoint_info(full_path))
    
    checkpoints.sort(key=lambda x: x["step"])
    return checkpoints


# ------------------------------------------------------------
#  TimeLimitCallback — Graceful Colab Shutdown
# ------------------------------------------------------------

class TimeLimitCallback(TrainerCallback):
    """
    Gracefully stop training when approaching Colab's time limit.
    
    Monitors elapsed time and triggers a clean shutdown before
    Colab force-kills the session, ensuring the latest checkpoint
    is saved to Google Drive.
    
    Also provides periodic terminal updates with:
      - Elapsed time / remaining time
      - Current GPU memory usage
      - Estimated completion time
      - Steps completed / remaining
    """
    
    def __init__(self, max_hours: float = 3.5, total_steps: int = 0, total_epochs: int = 0):
        self.max_hours = max_hours
        self.max_seconds = max_hours * 3600
        self.start_time = None
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.warned_75 = False
        self.warned_90 = False
        self.last_status_step = 0
        self.status_interval = 50  # Print status every N steps
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Record training start time."""
        self.start_time = time.time()
        self.total_steps = state.max_steps
        
        console.print(f"\n   [bold]Time Limit:[/bold] [cyan]{self.max_hours:.1f} hours[/cyan] "
                      f"({self.max_seconds/60:.0f} min)")
        console.print(f"  🎯 [bold]Total Steps:[/bold] [cyan]{self.total_steps:,}[/cyan]")
        
        est_time_per_step = None
        if self.total_steps > 0 and self.max_hours > 0:
            est_time_per_step = self.max_seconds / self.total_steps
            console.print(f"  📊 [bold]Max time/step:[/bold] [cyan]{est_time_per_step:.2f}s[/cyan]")
        
        console.print(f"  🕐 [bold]Started at:[/bold]  [cyan]{datetime.now().strftime('%H:%M:%S')}[/cyan]\n")
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check time limit and print periodic status updates."""
        if self.start_time is None:
            self.start_time = time.time()
        
        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600
        elapsed_min = elapsed / 60
        remaining_sec = self.max_seconds - elapsed
        remaining_min = remaining_sec / 60
        pct_time_used = (elapsed / self.max_seconds) * 100
        
        current_step = state.global_step
        steps_done = current_step
        steps_remaining = max(self.total_steps - current_step, 0)
        
        # -- Estimate time to completion based on actual pace --
        if steps_done > 0:
            time_per_step = elapsed / steps_done
            eta_seconds = steps_remaining * time_per_step
            eta_min = eta_seconds / 60
            est_finish_time = datetime.now() + timedelta(seconds=eta_seconds)
        else:
            time_per_step = 0
            eta_min = 0
            est_finish_time = datetime.now()
        
        # -- Current epoch --
        current_epoch = state.epoch if state.epoch else 0
        
        # -- Periodic status update (every N steps) --
        if current_step - self.last_status_step >= self.status_interval and current_step > 0:
            self.last_status_step = current_step
            
            # GPU memory info
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_alloc = torch.cuda.memory_allocated() / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_pct = (gpu_alloc / gpu_total) * 100
                gpu_info = f"  GPU: {gpu_alloc:.1f}/{gpu_total:.0f}GB ({gpu_pct:.0f}%)"
            
            # Get current loss from state
            loss_info = ""
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        loss_info = f"  Loss: {entry['loss']:.4f}"
                        break
            
            # Progress bar
            pct_steps = (steps_done / max(self.total_steps, 1)) * 100
            bar_width = 25
            filled = int(bar_width * pct_steps / 100)
            bar = "-" * filled + "░" * (bar_width - filled)
            
            console.print(
                f"  [{bar}] {pct_steps:5.1f}%"
                f"  Step {steps_done:>5}/{self.total_steps}"
                f"  Epoch {current_epoch:.1f}/{self.total_epochs}"
                f"  ⏱ {elapsed_min:.0f}m/{self.max_hours*60:.0f}m"
                f"  ETA: {eta_min:.0f}m ({est_finish_time.strftime('%H:%M')})"
                f"{loss_info}{gpu_info}",
                style="dim"
            )
        
        # -- 75% time warning --
        if pct_time_used >= 75 and not self.warned_75:
            self.warned_75 = True
            console.print(
                f"\n  ⚠️  [bold yellow]TIME WARNING:[/bold yellow] "
                f"[yellow]75% of time limit used ({elapsed_min:.0f}/{self.max_hours*60:.0f} min). "
                f"~{remaining_min:.0f} min remaining.[/yellow]\n"
            )
        
        # -- 90% time warning --
        if pct_time_used >= 90 and not self.warned_90:
            self.warned_90 = True
            console.print(
                f"\n  🚨 [bold red]TIME CRITICAL:[/bold red] "
                f"[red]90% of time limit used ({elapsed_min:.0f}/{self.max_hours*60:.0f} min). "
                f"Only ~{remaining_min:.0f} min remaining![/red]\n"
            )
        
        # -- Time limit reached — graceful shutdown --
        if elapsed >= self.max_seconds:
            console.print(
                f"\n  🛑 [bold red]TIME LIMIT REACHED![/bold red] "
                f"({elapsed_hours:.1f}h / {self.max_hours:.1f}h)\n"
                f"     [red]Saving final checkpoint and stopping gracefully...[/red]\n"
                f"     [dim]Training completed {steps_done}/{self.total_steps} steps "
                f"({pct_steps:.1f}%). Resume later to continue.[/dim]\n"
            )
            control.should_training_stop = True
            control.should_save = True
        
        return control
    
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Log when a checkpoint is saved."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        console.print(
            f"  [green]Checkpoint saved[/green] at step [cyan]{state.global_step}[/cyan] "
            f"(epoch {state.epoch:.2f}, elapsed {elapsed/60:.0f}m)"
        )
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        """Log evaluation results with rich formatting."""
        if metrics:
            f1 = metrics.get("eval_f1", 0)
            precision = metrics.get("eval_precision", 0)
            recall = metrics.get("eval_recall", 0)
            eval_loss = metrics.get("eval_loss", 0)
            
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            # Determine F1 quality indicator
            if f1 >= 0.8:
                f1_style = "bold green"
                indicator = "🟢"
            elif f1 >= 0.5:
                f1_style = "bold yellow"
                indicator = "🟡"
            else:
                f1_style = "bold red"
                indicator = "🔴"
            
            console.print(f"\n  {'-' * 55}")
            console.print(
                f"  📊 [bold]Eval @ step {state.global_step}[/bold] "
                f"(epoch {state.epoch:.1f}, {elapsed/60:.0f}m elapsed)"
            )
            console.print(
                f"     {indicator} F1: [{f1_style}]{f1:.4f}[/{f1_style}]  "
                f"P: {precision:.4f}  R: {recall:.4f}  "
                f"Loss: {eval_loss:.4f}"
            )
            console.print(f"  {'-' * 55}\n")
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Print training completion summary."""
        if self.start_time is None:
            return
        
        total_time = time.time() - self.start_time
        total_min = total_time / 60
        total_hours = total_time / 3600
        
        steps_done = state.global_step
        steps_total = self.total_steps
        pct_complete = (steps_done / max(steps_total, 1)) * 100
        
        if pct_complete >= 99.5:
            status = "COMPLETE"
            style = "bold green"
        else:
            status = " PAUSED (resume later)"
            style = "bold yellow"
        
        console.print(f"\n  {'-' * 55}")
        console.print(f"  [{style}]{status}[/{style}]")
        console.print(f"  {'-' * 55}")
        console.print(f"    Steps:    {steps_done:,} / {steps_total:,} ({pct_complete:.1f}%)")
        console.print(f"    Epochs:   {state.epoch:.2f} / {self.total_epochs}")
        console.print(f"    Duration: {total_min:.1f} min ({total_hours:.2f} hours)")
        console.print(f"    Finished: {datetime.now().strftime('%H:%M:%S')}")
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            console.print(f"    Peak GPU: {peak_mem:.1f} GB")
        
        console.print(f"  {'-' * 55}\n")


# ------------------------------------------------------------
#  Training Banner Display
# ------------------------------------------------------------

def print_training_banner(
    model_name: str,
    mode: str,
    total_params: int,
    trainable_params: int,
    num_epochs: int,
    train_size: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    save_steps: int,
    max_hours: float,
    resume_checkpoint: Optional[str] = None,
):
    """Print a visually rich training configuration banner."""
    
    effective_batch = batch_size * grad_accum
    steps_per_epoch = train_size // effective_batch
    total_steps = steps_per_epoch * num_epochs
    trainable_pct = (trainable_params / total_params) * 100
    saves_per_epoch = max(steps_per_epoch // save_steps, 1)
    
    # Estimate time based on mode
    if mode == "lora":
        est_time = f"~{num_epochs * 7}-{num_epochs * 9} min"
    else:
        est_time = f"~{num_epochs * 25}-{num_epochs * 35} min"
    
    gpu_name = "CPU"
    gpu_mem = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem = f"{gpu_total:.0f} GB"
    
    console.print(f"\n-{'-' * 60}-")
    console.print(f"-{'NER Fine-Tuning: Abbreviation Detection':^60}-")
    console.print(f"╠{'-' * 60}╣")
    console.print(f"-  Model:        {model_name:<43}-")
    console.print(f"-  Mode:         {mode.upper():<43}-")
    console.print(f"-  GPU:          {gpu_name:<43}-")
    console.print(f"-  GPU Memory:   {gpu_mem:<43}-")
    console.print(f"╠{'-' * 60}╣")
    console.print(f"-  Total Params:     {total_params:>12,}{' ' * 27}-")
    console.print(f"-  Trainable Params: {trainable_params:>12,}  ({trainable_pct:.2f}%){' ' * 15}-")
    console.print(f"╠{'-' * 60}╣")
    console.print(f"-  Epochs:           {num_epochs:<6}{' ' * 33}-")
    console.print(f"-  Batch Size:       {batch_size} × {grad_accum} = {effective_batch:<4} (effective){' ' * 20}-")
    console.print(f"-  Learning Rate:    {lr:<10}{' ' * 29}-")
    console.print(f"-  Steps/Epoch:      {steps_per_epoch:<6}{' ' * 33}-")
    console.print(f"-  Total Steps:      {total_steps:<6}{' ' * 33}-")
    console.print(f"-  Est. Time:        {est_time:<20}{' ' * 19}-")
    console.print(f"╠{'-' * 60}╣")
    console.print(f"-  Save Every:       {save_steps} steps (~{saves_per_epoch} saves/epoch){' ' * 18}-")
    console.print(f"-  Time Limit:       {max_hours:.1f} hours{' ' * 33}-")
    
    if resume_checkpoint:
        ckpt_name = os.path.basename(resume_checkpoint)
        console.print(f"-  RESUMING FROM: {ckpt_name:<39}-")
    else:
        console.print(f"-  Resume:           {'Fresh start (no checkpoint)':<39}-")
    
    console.print(f"-{'-' * 60}-\n")


# ------------------------------------------------------------
#  Resume Status Display
# ------------------------------------------------------------

def print_resume_status(output_dir: str, auto_resume: bool) -> Optional[str]:
    """
    Scan for existing checkpoints and display resume status.
    
    Returns the path to resume from, or None for fresh start.
    """
    checkpoints = list_all_checkpoints(output_dir)
    
    if not checkpoints:
        console.print("  [dim]No existing checkpoints found — starting fresh[/dim]\n")
        return None
    
    # -- Display found checkpoints --
    console.print(f"\n  [bold cyan]Found {len(checkpoints)} existing checkpoint(s):[/bold cyan]\n")
    
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Checkpoint", style="cyan", width=18)
    table.add_column("Step", justify="right", width=8)
    table.add_column("Epoch", justify="right", width=8)
    table.add_column("Loss", justify="right", width=10)
    table.add_column("Size", justify="right", width=10)
    
    for i, ckpt in enumerate(checkpoints, 1):
        is_latest = (i == len(checkpoints))
        name = os.path.basename(ckpt["path"])
        if is_latest:
            name = f"→ {name}"
        
        table.add_row(
            str(i),
            name,
            f"{ckpt['step']:,}",
            f"{ckpt.get('epoch', '?')}",
            f"{ckpt.get('last_loss', '?')}",
            f"{ckpt['size_mb']:.0f} MB",
        )
    
    console.print(table)
    
    latest = checkpoints[-1]
    latest_path = latest["path"]
    
    if auto_resume:
        console.print(
            f"\n  [bold green]AUTO-RESUME ENABLED[/bold green] — "
            f"will continue from [cyan]{os.path.basename(latest_path)}[/cyan] "
            f"(step {latest['step']:,}, epoch {latest.get('epoch', '?')})\n"
        )
        return latest_path
    else:
        console.print(
            f"\n  ℹ️  [dim]Auto-resume disabled. Remove --no-resume to resume from "
            f"{os.path.basename(latest_path)}[/dim]\n"
        )
        return None


# ------------------------------------------------------------
#  Main Training Function
# ------------------------------------------------------------

def train_model(
    model,
    tokenizer,
    train_dataset,
    valid_dataset,
    id2label: Dict[int, str],
    config,
    mode: str = "lora",
) -> Tuple[Dict, object]:
    """
    Train the model using HuggingFace Trainer with Colab-resilient checkpointing.
    
    This function:
      1. Configures TrainingArguments with step-based saving
      2. Detects and resumes from the latest checkpoint (if auto_resume=True)
      3. Adds TimeLimitCallback for graceful Colab shutdown
      4. Sets up data collator for dynamic padding
      5. Runs training with periodic validation
      6. Returns training statistics (time, best F1, loss curve)
    
    Args:
        model: The model to train (with or without LoRA)
        tokenizer: Tokenizer for the model
        train_dataset: NERDataset for training
        valid_dataset: NERDataset for validation
        id2label: ID-to-label mapping
        config: Config object with all hyperparameters
        mode: 'lora' or 'full'
        
    Returns:
        (training_stats dict, trainer object)
    """
    
    # -- Determine output directory --
    # Use Colab Drive path if available, otherwise local
    is_colab = os.path.exists("/content/drive")
    if is_colab:
        base_output = config.colab.drive_output_dir
        console.print(f"  ☁️  [bold green]Google Drive detected[/bold green] — "
                      f"checkpoints will be saved to Drive")
    else:
        base_output = "results"
        console.print(f"  💻 [dim]Local mode — checkpoints saved to ./results/[/dim]")
    
    output_dir = os.path.join(base_output, "lora" if mode == "lora" else "full_ft")
    os.makedirs(output_dir, exist_ok=True)
    
    # -- Select mode-specific hyperparameters --
    if mode == "lora":
        grad_accum = config.training.gradient_accumulation_steps_lora
        optimizer = config.training.optim_lora
        model_name = config.training.model_name_lora
    else:
        grad_accum = config.training.gradient_accumulation_steps_full
        optimizer = config.training.optim_full
        model_name = config.training.model_name_full
    
    # -- Check for existing checkpoints --
    console.print(f"\n  🔍 [bold]Scanning for existing checkpoints...[/bold]")
    resume_checkpoint = print_resume_status(output_dir, config.colab.auto_resume)
    
    # -- Count parameters for banner --
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # -- Calculate training steps for the TimeLimitCallback --
    effective_batch = config.training.per_device_train_batch_size * grad_accum
    steps_per_epoch = max(len(train_dataset) // effective_batch, 1)
    total_training_steps = steps_per_epoch * config.training.num_epochs
    
    # -- Print training banner --
    print_training_banner(
        model_name=model_name,
        mode=mode,
        total_params=total_params,
        trainable_params=trainable_params,
        num_epochs=config.training.num_epochs,
        train_size=len(train_dataset),
        batch_size=config.training.per_device_train_batch_size,
        grad_accum=grad_accum,
        lr=config.training.learning_rate,
        save_steps=config.colab.save_steps,
        max_hours=config.colab.max_training_hours,
        resume_checkpoint=resume_checkpoint,
    )
    
    # ----------------------------------------------------------
    #  Configure Training Arguments (Step-Based Saving)
    # ----------------------------------------------------------
    
    training_args = TrainingArguments(
        # -- Output --
        output_dir=output_dir,

        
        # -- Training Duration --
        num_train_epochs=config.training.num_epochs,
        
        # -- Batch Size & Gradient Accumulation --
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        
        # -- Optimizer & Scheduler --
        optim=optimizer,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
        
        # -- Mixed Precision --
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        
        # -- STEP-BASED Evaluation & Saving (key for Colab resilience) --
        eval_strategy="steps",
        eval_steps=config.colab.save_steps,          # Eval at same frequency as saves
        save_strategy="steps",
        save_steps=config.colab.save_steps,           # Save every N steps
        save_total_limit=config.colab.save_total_limit,  # Keep N latest checkpoints
        load_best_model_at_end=True,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        
        # -- Logging --
        logging_steps=config.training.logging_steps,
        logging_first_step=True,
        report_to=config.training.report_to,
        
        # -- Reproducibility --
        seed=config.training.seed,
        data_seed=config.training.seed,
        
        # -- Performance --
        dataloader_num_workers=2 if os.name != 'nt' else 0,
        remove_unused_columns=False,
    )
    
    # ----------------------------------------------------------
    #  Data Collator (handles dynamic padding per batch)
    # ----------------------------------------------------------
    
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=config.data.max_length,
    )
    
    # ----------------------------------------------------------
    #  Build Metrics Function
    # ----------------------------------------------------------
    
    compute_metrics = build_compute_metrics_fn(id2label)
    
    # ----------------------------------------------------------
    #  Create Callbacks
    # ----------------------------------------------------------
    
    time_limit_callback = TimeLimitCallback(
        max_hours=config.colab.max_training_hours,
        total_steps=total_training_steps,
        total_epochs=config.training.num_epochs,
    )
    
    # ----------------------------------------------------------
    #  Create Trainer
    # ----------------------------------------------------------
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[time_limit_callback, EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # ----------------------------------------------------------
    #  START TRAINING (with auto-resume)
    # ----------------------------------------------------------
    
    if resume_checkpoint:
        console.print(f"  [bold green]RESUMING[/bold green] from: [cyan]{os.path.basename(resume_checkpoint)}[/cyan]")
        
        ckpt_info = get_checkpoint_info(resume_checkpoint)
        console.print(f"     Step: {ckpt_info['step']:,}  |  "
                      f"Epoch: {ckpt_info.get('epoch', '?')}  |  "
                      f"Last Loss: {ckpt_info.get('last_loss', '?')}")
        
        remaining_steps = total_training_steps - ckpt_info['step']
        if remaining_steps > 0:
            est_remaining_min = remaining_steps * 1.0  # rough ~1s/step for LoRA
            if mode != "lora":
                est_remaining_min = remaining_steps * 3.0  # ~3s/step for Full FT
            est_remaining_min /= 60
            console.print(f"     Steps remaining: ~{remaining_steps:,} "
                          f"(est. ~{est_remaining_min:.0f} min)")
        else:
            console.print(f"     [green]Training already complete! Running final eval only.[/green]")
    else:
        console.print("  [bold green]Starting training from scratch...[/bold green]")
    
    console.print(f"\n  {'-' * 55}\n")
    
    train_start_time = time.time()
    
    # Run training — with or without resume
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    train_end_time = time.time()
    training_time_sec = train_end_time - train_start_time
    training_time_min = training_time_sec / 60
    
    # ----------------------------------------------------------
    #  Training Complete — Report Results
    # ----------------------------------------------------------
    
    console.print(f"\n  {'-' * 55}")
    console.print(f"  [bold green]TRAINING SESSION COMPLETE[/bold green]")
    console.print(f"  {'-' * 55}")
    console.print(f"    This Session Time: [cyan]{training_time_min:.1f} min ({training_time_sec:.0f}s)[/cyan]")
    console.print(f"    Final Train Loss:  [yellow]{train_result.training_loss:.4f}[/yellow]")
    console.print(f"    Steps Completed:   [cyan]{trainer.state.global_step:,} / {total_training_steps:,}[/cyan]")
    
    pct_complete = (trainer.state.global_step / max(total_training_steps, 1)) * 100
    if pct_complete >= 99.5:
        console.print(f"    Status:            [bold green]All epochs completed![/bold green]")
    else:
        console.print(f"    Status:            [bold yellow] {pct_complete:.0f}% complete — "
                      f"resume later to finish[/bold yellow]")
    
    # -- Run final validation --
    console.print(f"\n  📊 Running final validation...")
    eval_results = trainer.evaluate()
    
    console.print(f"    Val Precision:  [green]{eval_results.get('eval_precision', 0):.4f}[/green]")
    console.print(f"    Val Recall:     [green]{eval_results.get('eval_recall', 0):.4f}[/green]")
    console.print(f"    Val F1:         [bold green]{eval_results.get('eval_f1', 0):.4f}[/bold green]")
    console.print(f"    Val Accuracy:   [green]{eval_results.get('eval_accuracy', 0):.4f}[/green]")
    
    # -- GPU Memory Summary --
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        current_mem = torch.cuda.memory_allocated() / (1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        console.print(f"\n    GPU Memory: {current_mem:.1f} GB allocated, "
                      f"{peak_mem:.1f} GB peak, {total_mem:.0f} GB total")
    
    # -- Compile training statistics --
    training_stats = {
        "mode": mode,
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round((trainable_params / total_params) * 100, 2),
        "num_epochs": config.training.num_epochs,
        "learning_rate": config.training.learning_rate,
        "effective_batch_size": config.training.per_device_train_batch_size * grad_accum,
        "training_time_seconds": round(training_time_sec, 1),
        "training_time_minutes": round(training_time_min, 1),
        "final_train_loss": round(train_result.training_loss, 4),
        "best_val_f1": round(eval_results.get('eval_f1', 0), 4),
        "best_val_precision": round(eval_results.get('eval_precision', 0), 4),
        "best_val_recall": round(eval_results.get('eval_recall', 0), 4),
        "best_val_accuracy": round(eval_results.get('eval_accuracy', 0), 4),
        "train_log": train_result.metrics,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "steps_completed": trainer.state.global_step,
        "total_steps": total_training_steps,
        "pct_complete": round(pct_complete, 1),
        "resumed_from": os.path.basename(resume_checkpoint) if resume_checkpoint else None,
        "save_steps": config.colab.save_steps,
        "max_training_hours": config.colab.max_training_hours,
        "timestamp": datetime.now().isoformat(),
    }
    
    # -- Save training stats --
    stats_path = os.path.join(output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    console.print(f"\n  Training stats saved to: [green]{stats_path}[/green]")
    
    # -- Save the trained model --
    from src.model_utils import save_model
    save_model(model, tokenizer, output_dir, mode)
    
    console.print(f"  {'-' * 55}\n")
    
    return training_stats, trainer

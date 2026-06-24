# ============================================================
# evaluator.py — Test Set Evaluation & Comparison
# ============================================================
# Handles:
#   1. Running inference on test set
#   2. Computing entity-level P, R, F1 (per-class & overall)
#   3. Generating detailed classification reports
#   4. Comparing LoRA vs Full FT side-by-side
#   5. Saving predictions and metrics to files
# ============================================================

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
from transformers import Trainer, DataCollatorForTokenClassification
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ------------------------------------------------------------
#  STEP 1: Run Predictions on Test Set
# ------------------------------------------------------------

def predict_on_test(
    trainer: Trainer,
    test_dataset,
    id2label: Dict[int, str],
    test_sentences: Optional[List[Dict]] = None,
) -> Tuple[List[List[str]], List[List[str]], Optional[List[Dict]]]:
    """
    Run the trained model on the test dataset and extract predictions.
    
    Args:
        trainer: HF Trainer with the trained model
        test_dataset: NERDataset for the test split
        id2label: ID-to-label mapping
        test_sentences: Original test sentences list to filter and align
        
    Returns:
        (true_labels, pred_labels, filtered_sentences)
    """
    console.print("\n  🔮 Running inference on test set...")
    start_time = time.time()
    
    # Get raw predictions
    raw_predictions = trainer.predict(test_dataset)
    predictions = np.argmax(raw_predictions.predictions, axis=2)
    labels = raw_predictions.label_ids
    
    # Convert numeric IDs to BIO tag strings
    true_labels = []
    pred_labels = []
    filtered_sentences = [] if test_sentences is not None else None
    
    for sent_idx, (pred_seq, label_seq) in enumerate(zip(predictions, labels)):
        true_sent = []
        pred_sent = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                true_sent.append(id2label.get(label_id, "O"))
                pred_sent.append(id2label.get(pred_id, "O"))
        
        if true_sent:  # Skip empty sentences
            true_labels.append(true_sent)
            pred_labels.append(pred_sent)
            if filtered_sentences is not None and sent_idx < len(test_sentences):
                filtered_sentences.append(test_sentences[sent_idx])
    
    elapsed = time.time() - start_time
    console.print(f"    Inference completed in [green]{elapsed:.1f}s[/green]")
    console.print(f"    Processed [cyan]{len(true_labels):,}[/cyan] sentences\n")
    
    return true_labels, pred_labels, filtered_sentences


# ------------------------------------------------------------
#  STEP 2: Compute & Display Evaluation Metrics
# ------------------------------------------------------------

def evaluate_predictions(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    mode: str = "lora",
    output_dir: str = "results",
) -> Dict:
    """
    Compute entity-level metrics and display detailed results.
    
    Metrics computed:
      - Overall Precision, Recall, F1 (micro-averaged)
      - Per-entity-type P, R, F1 (short and long separately)
      - Detailed seqeval classification report
    
    Args:
        true_labels: Ground truth BIO tags (list of lists)
        pred_labels: Predicted BIO tags (list of lists)
        mode: 'lora' or 'full' (for display and saving)
        output_dir: Where to save the metrics file
        
    Returns:
        Dict with all computed metrics
    """
    
    console.print("----------------------------------------------------------------")
    console.print(f"-     📊 [bold cyan]EVALUATION RESULTS — {mode.upper()} Fine-Tuning[/bold cyan]              -")
    console.print("----------------------------------------------------------------\n")
    
    # -- Overall Metrics --
    overall_precision = precision_score(true_labels, pred_labels, zero_division=0)
    overall_recall = recall_score(true_labels, pred_labels, zero_division=0)
    overall_f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    # -- Token-level Accuracy --
    correct = 0
    total = 0
    for true_sent, pred_sent in zip(true_labels, pred_labels):
        for t, p in zip(true_sent, pred_sent):
            total += 1
            if t == p:
                correct += 1
    token_accuracy = correct / max(total, 1)
    
    # -- Display Overall Metrics --
    console.print("  [bold]Overall Metrics (Entity-Level, Micro-Averaged):[/bold]\n")
    
    metrics_table = Table(show_header=True, header_style="bold green")
    metrics_table.add_column("Metric", style="cyan", width=15)
    metrics_table.add_column("Value", justify="center", width=12)
    
    metrics_table.add_row("Precision", f"{overall_precision:.4f}")
    metrics_table.add_row("Recall", f"{overall_recall:.4f}")
    metrics_table.add_row("[bold]F1 Score[/bold]", f"[bold]{overall_f1:.4f}[/bold]")
    metrics_table.add_row("Token Accuracy", f"{token_accuracy:.4f}")
    
    console.print(metrics_table)
    
    # -- Detailed Classification Report --
    console.print("\n  [bold]Per-Entity-Type Breakdown:[/bold]\n")
    
    report_str = classification_report(true_labels, pred_labels, zero_division=0)
    console.print(f"  {report_str}")
    
    # -- Compile metrics dict --
    metrics = {
        "mode": mode,
        "overall": {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "token_accuracy": round(token_accuracy, 4),
        },
        "num_test_sentences": len(true_labels),
        "num_test_tokens": total,
        "classification_report": report_str,
    }
    
    # -- Save metrics to JSON --
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "evaluation_results.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    console.print(f"  Metrics saved to: [green]{metrics_path}[/green]\n")
    
    return metrics


# ------------------------------------------------------------
#  STEP 3: Save Predictions to File
# ------------------------------------------------------------

def save_predictions(
    test_sentences: List[Dict],
    pred_labels: List[List[str]],
    output_dir: str,
):
    """
    Save predictions in CoNLL format alongside ground truth for comparison.
    
    Output format (per line):
      word    true_tag    predicted_tag
    
    Args:
        test_sentences: Original test sentence dicts
        pred_labels: Predicted BIO tags
        output_dir: Where to save predictions file
    """
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, "test_predictions.txt")
    
    with open(pred_path, 'w', encoding='utf-8') as f:
        f.write("# Format: word\\ttrue_tag\\tpredicted_tag\\n")
        f.write("# Blank lines separate sentences\\n\\n")
        
        for sent_idx, (sentence, preds) in enumerate(zip(test_sentences, pred_labels)):
            words = sentence['words']
            true_tags = sentence['tags']
            
            # preds might be shorter than words due to truncation
            for i, word in enumerate(words):
                true_tag = true_tags[i] if i < len(true_tags) else "O"
                pred_tag = preds[i] if i < len(preds) else "O"
                
                # Mark incorrect predictions
                marker = "  ✗" if true_tag != pred_tag else ""
                f.write(f"{word}\t{true_tag}\t{pred_tag}{marker}\n")
            
            f.write("\n")  # Blank line between sentences
    
    console.print(f"  Predictions saved to: [green]{pred_path}[/green]")
    console.print(f"     ({len(pred_labels):,} sentences)\n")


# ------------------------------------------------------------
#  STEP 4: Compare LoRA vs Full Fine-Tuning Results
# ------------------------------------------------------------

def compare_results(lora_metrics: Dict, full_metrics: Optional[Dict] = None):
    """
    Print a side-by-side comparison of LoRA and Full FT results.
    
    Args:
        lora_metrics: Metrics dict from LoRA evaluation
        full_metrics: Metrics dict from Full FT evaluation (optional)
    """
    console.print("\n----------------------------------------------------------------")
    console.print("-           📊 [bold cyan]COMPARISON: LoRA vs Full Fine-Tuning[/bold cyan]           -")
    console.print("----------------------------------------------------------------\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("LoRA (QLoRA 4-bit)", justify="center", width=20)
    
    if full_metrics:
        table.add_column("Full Fine-Tuning", justify="center", width=20)
        table.add_column("Winner", justify="center", width=10)
    
    lora_overall = lora_metrics.get("overall", {})
    full_overall = full_metrics.get("overall", {}) if full_metrics else {}
    
    metrics_to_compare = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("Token Accuracy", "token_accuracy"),
    ]
    
    for display_name, key in metrics_to_compare:
        lora_val = lora_overall.get(key, 0)
        
        if full_metrics:
            full_val = full_overall.get(key, 0)
            winner = "LoRA" if lora_val >= full_val else "Full FT"
            winner_style = "[green]" if winner == "LoRA" else "[yellow]"
            table.add_row(
                display_name,
                f"{lora_val:.4f}",
                f"{full_val:.4f}",
                f"{winner_style}{winner}[/]",
            )
        else:
            table.add_row(display_name, f"{lora_val:.4f}")
    
    console.print(table)
    
    # -- Training time comparison --
    if full_metrics:
        console.print("\n  [bold]Training Time Comparison:[/bold]")
        lora_time = lora_metrics.get("training_time_minutes", "N/A")
        full_time = full_metrics.get("training_time_minutes", "N/A")
        console.print(f"    LoRA:    {lora_time} min")
        console.print(f"    Full FT: {full_time} min")
        
        if isinstance(lora_time, (int, float)) and isinstance(full_time, (int, float)):
            speedup = full_time / max(lora_time, 0.1)
            console.print(f"    Speedup: [green]{speedup:.1f}x faster with LoRA[/green]")
    
    console.print("")


# ------------------------------------------------------------
#  STEP 5: Full Evaluation Pipeline
# ------------------------------------------------------------

def run_full_evaluation(
    trainer: Trainer,
    test_dataset,
    test_sentences: List[Dict],
    id2label: Dict[int, str],
    mode: str,
    output_dir: str,
    training_stats: Optional[Dict] = None,
) -> Dict:
    """
    Complete evaluation pipeline:
      1. Predict on test set
      2. Compute P, R, F1
      3. Save predictions
      4. Return metrics with training stats merged
      
    Args:
        trainer: Trained HF Trainer
        test_dataset: NERDataset for test split
        test_sentences: Original test sentences (for saving predictions)
        id2label: ID-to-label mapping
        mode: 'lora' or 'full'
        output_dir: Output directory for this mode
        training_stats: Optional training stats to merge
        
    Returns:
        Complete metrics dict
    """
    console.print(f"\n{'-' * 62}")
    console.print(f"  [bold]STEP: Evaluating {mode.upper()} model on TEST set[/bold]")
    console.print(f"{'-' * 62}\n")
    
    # Step 1: Get predictions
    true_labels, pred_labels, filtered_sentences = predict_on_test(trainer, test_dataset, id2label, test_sentences)
    
    # Step 2: Compute metrics
    metrics = evaluate_predictions(true_labels, pred_labels, mode, output_dir)
    
    # Step 3: Save predictions
    save_predictions(filtered_sentences or test_sentences, pred_labels, output_dir)
    
    # Merge training stats if available
    if training_stats:
        metrics["training_time_seconds"] = training_stats.get("training_time_seconds")
        metrics["training_time_minutes"] = training_stats.get("training_time_minutes")
        metrics["total_params"] = training_stats.get("total_params")
        metrics["trainable_params"] = training_stats.get("trainable_params")
        metrics["trainable_pct"] = training_stats.get("trainable_pct")
        metrics["model_name"] = training_stats.get("model_name")
    
    return metrics

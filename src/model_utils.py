# ============================================================
# model_utils.py — Model Loading & Configuration
# ============================================================
# Handles:
#   1. Loading pre-trained models (Flan-T5)
#   2. Applying LoRA adapters via PEFT
#   3. 4-bit quantization via BitsAndBytes
#   4. Parameter counting (total vs trainable)
#   5. GPU memory reporting
# ============================================================

import os
import time
import torch
from typing import Dict, Tuple, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from rich.console import Console
from rich.table import Table

console = Console()


# ------------------------------------------------------------
#  STEP 1: Load Tokenizer
# ------------------------------------------------------------

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the tokenizer for the specified model.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'google/flan-t5-base')
        
    Returns:
        Configured tokenizer
    """
    console.print(f"  📝 Loading tokenizer: [cyan]{model_name}[/cyan]")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    console.print(f"    Vocab size: {tokenizer.vocab_size:,}")
    console.print(f"    Pad token:  '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    
    return tokenizer


# ------------------------------------------------------------
#  STEP 2: Load Model for Token Classification
# ------------------------------------------------------------

def load_model_for_ner(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    mode: str = "lora",
    lora_config_params: Optional[dict] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple:
    """
    Load a pre-trained model configured for token classification (NER).
    
    Supports two modes:
      - 'lora': Load with 4-bit quantization + LoRA adapters (fast, memory-efficient)
      - 'full': Load full model in FP16 (all parameters trainable)
    
    Args:
        model_name: HuggingFace model identifier
        label2id: Label-to-ID mapping
        id2label: ID-to-label mapping
        mode: 'lora' or 'full'
        lora_config_params: Dict of LoRA hyperparameters (r, alpha, etc.)
        
    Returns:
        (model, tokenizer) tuple
    """
    console.print(f"\n----------------------------------------------------------------")
    console.print(f"-        🤖 [bold cyan]STEP 2: Loading Model ({mode.upper()})[/bold cyan]                      -")
    console.print(f"----------------------------------------------------------------\n")
    
    num_labels = len(label2id)
    start_time = time.time()
    
    # -- Load Tokenizer --
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    
    if mode == "lora":
        # --------------------------------------------------------
        #  QLoRA: 4-bit quantized base + trainable LoRA adapters
        # --------------------------------------------------------
        
        console.print(f"\n  [bold yellow]Loading model with 4-bit quantization (QLoRA)...[/bold yellow]")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                           # Enable 4-bit loading
            bnb_4bit_quant_type="nf4",                   # NormalFloat4 (best quality)
            bnb_4bit_compute_dtype=torch.float32,         # Compute in FP32 to avoid T5 overflow
            bnb_4bit_use_double_quant=True,              # Double quantization for extra savings
            llm_int8_skip_modules=["classifier"],        # DO NOT quantize the classifier head
        )
        
        # Load the base model with quantization
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            quantization_config=bnb_config,
            device_map="auto",                           # Auto-place on GPU
            dtype=torch.float32,
        )
        
        # Prepare model for k-bit training (enables gradient checkpointing etc.)
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA adapters
        if lora_config_params is None:
            lora_config_params = {}
        
        peft_config = LoraConfig(
            r=lora_config_params.get('r', 16),
            lora_alpha=lora_config_params.get('lora_alpha', 32),
            lora_dropout=lora_config_params.get('lora_dropout', 0.05),
            bias=lora_config_params.get('bias', 'none'),
            task_type=TaskType.TOKEN_CLS,
            target_modules=lora_config_params.get('target_modules', ["q", "v"]),
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, peft_config)
        
        console.print(f"    LoRA rank (r):   {peft_config.r}")
        console.print(f"    LoRA alpha:      {peft_config.lora_alpha}")
        console.print(f"    Target modules:  {peft_config.target_modules}")
        
    else:
        # --------------------------------------------------------
        #  Full Fine-Tuning: All parameters trainable in FP16
        # --------------------------------------------------------
        
        console.print(f"\n  🔧 [bold yellow]Loading model for full fine-tuning (FP16)...[/bold yellow]")
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            dtype=torch.float32,
        )
        
        # Move to GPU
        if torch.cuda.is_available():
            model = model.cuda()
    
    # -- Report parameter counts --
    total_params, trainable_params = count_parameters(model)
    elapsed = time.time() - start_time
    
    # -- Print Summary --
    print_model_summary(model_name, mode, total_params, trainable_params, elapsed)
    
    return model, tokenizer


# ------------------------------------------------------------
#  STEP 3: Count Model Parameters
# ------------------------------------------------------------

def count_parameters(model) -> Tuple[int, int]:
    """
    Count total and trainable parameters in the model.
    
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(
    model_name: str,
    mode: str,
    total_params: int,
    trainable_params: int,
    load_time: float
):
    """Print a formatted model summary with parameter statistics."""
    
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100 if total_params > 0 else 0
    
    console.print(f"\n  ---------------------------------------------------")
    console.print(f"  📊 [bold]MODEL SUMMARY[/bold]")
    console.print(f"  ---------------------------------------------------")
    console.print(f"    Model:            {model_name}")
    console.print(f"    Mode:             {mode.upper()}")
    console.print(f"    Total params:     {total_params:>12,}")
    console.print(f"    Trainable params: {trainable_params:>12,}  ({trainable_pct:.2f}%)")
    console.print(f"    Frozen params:    {frozen_params:>12,}  ({100 - trainable_pct:.2f}%)")
    console.print(f"    Load time:        {load_time:.1f}s")
    
    # -- GPU Memory --
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        
        console.print(f"    GPU:              {gpu_name}")
        console.print(f"    GPU Memory:       {gpu_mem_allocated:.1f} GB / {gpu_total:.1f} GB "
                      f"(allocated/total)")
    
    console.print(f"  ---------------------------------------------------\n")
    
    return {
        "model_name": model_name,
        "mode": mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_pct, 2),
        "load_time_sec": round(load_time, 1),
    }


# ------------------------------------------------------------
#  STEP 4: Save & Load Fine-Tuned Models
# ------------------------------------------------------------

def save_model(model, tokenizer, output_dir: str, mode: str = "lora"):
    """
    Save the fine-tuned model and tokenizer to disk.
    
    For LoRA: saves only the adapter weights (small files)
    For Full: saves the complete model
    """
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    if mode == "lora":
        # Save only the LoRA adapter weights (very small: ~10 MB)
        model.save_pretrained(model_dir)
        console.print(f"  LoRA adapters saved to: [green]{model_dir}[/green]")
    else:
        # Save the full model
        model.save_pretrained(model_dir)
        console.print(f"  Full model saved to: [green]{model_dir}[/green]")
    
    # Always save the tokenizer alongside
    tokenizer.save_pretrained(model_dir)
    console.print(f"  Tokenizer saved to: [green]{model_dir}[/green]")
    
    # Report saved file sizes
    total_size = 0
    for f in os.listdir(model_dir):
        fpath = os.path.join(model_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
    
    console.print(f"  📦 Total saved size: [cyan]{total_size / (1024**2):.1f} MB[/cyan]\n")

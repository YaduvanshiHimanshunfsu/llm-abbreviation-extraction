# ============================================================
# config.py — All Hyperparameters & Configuration
# ============================================================
# This file centralizes every tunable parameter so you can
# adjust training without touching any other source file.
# ============================================================

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Configuration for dataset paths and preprocessing."""
    
    # -- Paths to the raw CoNLL-format data files --
    train_file: str = "data/raw/train.txt"
    valid_file: str = "data/raw/valid.txt"
    test_file: str = "data/raw/test.txt"
    
    # -- Label schema --
    # BIO tags used in the dataset
    label_list: List[str] = field(default_factory=lambda: [
        "O",         # Outside any entity
        "B-short",   # Beginning of abbreviation/acronym
        "I-short",   # Inside abbreviation (multi-token)
        "B-long",    # Beginning of long-form expansion
        "I-long",    # Inside long-form expansion
    ])
    
    # -- Tokenization settings --
    max_length: int = 256          # Max sequence length (tokens)
    truncation: bool = True        # Truncate sequences longer than max_length
    
    # -- Case sensitivity --
    # The test file uses UPPERCASE tags (B-LONG, I-SHORT, etc.)
    # We normalize everything to lowercase for consistency
    normalize_tags: bool = True


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""
    
    # -- LoRA Hyperparameters --
    r: int = 16                        # Rank of the low-rank matrices (higher = more params)
    lora_alpha: int = 32               # Scaling factor (alpha/r = scaling)
    lora_dropout: float = 0.05         # Dropout on LoRA layers
    bias: str = "none"                 # Don't train bias terms
    
    # -- Which modules to apply LoRA to --
    # For T5: query (q) and value (v) projections in attention
    target_modules: List[str] = field(default_factory=lambda: ["q", "v"])
    
    # -- Quantization (QLoRA) --
    use_4bit: bool = True              # Enable 4-bit quantization of base model
    bnb_4bit_compute_dtype: str = "float16"   # Compute dtype for 4-bit
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4 quantization


@dataclass  
class TrainingConfig:
    """Configuration for the training loop."""
    
    # -- Model Selection --
    # flan-t5-base (250M params) for LoRA — good balance of speed & quality
    # flan-t5-small (77M params) for Full FT — fits in T4 memory
    model_name_lora: str = "google/flan-t5-base"
    model_name_full: str = "google/flan-t5-small"
    
    # -- Training Hyperparameters --
    num_epochs: int = 5                # Number of training epochs
    learning_rate: float = 1e-4        # Peak learning rate
    weight_decay: float = 0.01         # L2 regularization
    warmup_ratio: float = 0.06         # Warmup steps as fraction of total
    
    # -- Batch Sizes --
    # Effective batch size = per_device_batch * gradient_accumulation
    per_device_train_batch_size: int = 16    # Per-GPU batch size (training)
    per_device_eval_batch_size: int = 32     # Per-GPU batch size (eval, can be larger)
    gradient_accumulation_steps_lora: int = 2     # For LoRA (effective: 32)
    gradient_accumulation_steps_full: int = 4     # For Full FT (effective: 32)
    
    # -- Mixed Precision --
    fp16: bool = False                 # FP16 disabled because T5 overflows in FP16
    bf16: bool = False                 # T4 doesn't support BF16 natively
    
    # -- Learning Rate Schedule --
    lr_scheduler_type: str = "cosine"  # Cosine decay after warmup
    
    # -- Optimizer --
    optim_lora: str = "paged_adamw_8bit"  # Memory-efficient optimizer for LoRA
    optim_full: str = "adamw_torch"        # Standard AdamW for Full FT
    
    # -- Gradient Clipping --
    max_grad_norm: float = 1.0         # Clip gradients to prevent explosion
    
    # -- Evaluation Strategy --
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"  # Use F1 to pick best checkpoint
    greater_is_better: bool = True
    
    # -- Logging --
    logging_steps: int = 50            # Log every N steps
    report_to: str = "none"            # Disable wandb by default
    
    # -- Output Directories --
    output_dir_lora: str = "results/lora"
    output_dir_full: str = "results/full_ft"
    
    # -- Reproducibility --
    seed: int = 42


@dataclass
class ColabConfig:
    """Configuration for Colab-resilient training with checkpoint resume."""
    
    # -- Google Drive persistence --
    # Checkpoints saved here survive Colab disconnects
    drive_output_dir: str = "/content/drive/MyDrive/Nit_trichy/results"
    
    # -- Checkpoint frequency --
    # With ~500 steps/epoch (16K samples, batch=16, grad_accum=2):
    #   save_steps=200 → ~6 saves/epoch → lose at most ~3 min of work
    save_steps: int = 200
    
    # -- Time-limit safety --
    # Colab free-tier: ~4 hours max
    # We stop 30 min early to guarantee final checkpoint + eval complete
    max_training_hours: float = 3.5
    
    # -- Auto-resume --
    # Automatically detect and resume from the latest checkpoint
    auto_resume: bool = True
    
    # -- Checkpoint retention --
    # Keep N most recent checkpoints (rolling window)
    save_total_limit: int = 3


@dataclass
class Config:
    """Master configuration that bundles everything together."""
    
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    
    def print_summary(self, mode: str = "lora"):
        """Print a formatted summary of the current configuration."""
        model_name = self.training.model_name_lora if mode == "lora" else self.training.model_name_full
        grad_accum = self.training.gradient_accumulation_steps_lora if mode == "lora" else self.training.gradient_accumulation_steps_full
        effective_batch = self.training.per_device_train_batch_size * grad_accum
        
        print("\n" + "=" * 62)
        print("  CONFIGURATION SUMMARY")
        print("=" * 62)
        print(f"  Mode:              {mode.upper()}")
        print(f"  Model:             {model_name}")
        print(f"  Epochs:            {self.training.num_epochs}")
        print(f"  Learning Rate:     {self.training.learning_rate}")
        print(f"  Batch Size:        {self.training.per_device_train_batch_size} × {grad_accum} = {effective_batch} (effective)")
        print(f"  Max Seq Length:     {self.data.max_length}")
        print(f"  FP16:              {self.training.fp16}")
        
        if mode == "lora":
            print(f"  LoRA Rank (r):     {self.lora.r}")
            print(f"  LoRA Alpha:        {self.lora.lora_alpha}")
            print(f"  LoRA Dropout:      {self.lora.lora_dropout}")
            print(f"  4-bit Quant:       {self.lora.use_4bit}")
            print(f"  Optimizer:         {self.training.optim_lora}")
        else:
            print(f"  Optimizer:         {self.training.optim_full}")
        
        print(f"  LR Scheduler:      {self.training.lr_scheduler_type}")
        print(f"  Seed:              {self.training.seed}")
        print("=" * 62 + "\n")

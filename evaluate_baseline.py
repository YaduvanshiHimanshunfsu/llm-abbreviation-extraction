import os
import warnings

# Suppress warnings for cleaner output
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

from transformers import Trainer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from src.config import Config
from src.data_utils import load_and_prepare_data
from src.model_utils import load_tokenizer
from src.evaluator import run_full_evaluation

def evaluate_untrained_baseline():
    """
    Evaluates the raw, un-finetuned FLAN-T5 model on the test dataset.
    This establishes a zero-shot baseline to prove the necessity and 
    effectiveness of the fine-tuning process.
    """
    config = Config()
    
    print("\n---------------------------------------------------------")
    print("   EVALUATING UNTRAINED BASELINE MODEL ")
    print("---------------------------------------------------------\n")
    
    print("1. Loading test dataset...")
    tokenizer = load_tokenizer(config.training.model_name_full)
    _, _, test_dataset, label2id, id2label, test_sentences, _ = load_and_prepare_data(config, tokenizer)
    
    print("\n2. Loading raw base model (NO TRAINING WEIGHTS)...")
    # We load the base model directly from HuggingFace with random token classification weights
    model = AutoModelForTokenClassification.from_pretrained(
        config.training.model_name_full,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    print("\n3. Running evaluation on the untrained model...")
    # Fix: We must provide the DataCollator so that the batch sequences are padded correctly
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(model=model, data_collator=data_collator)
    
    os.makedirs("results/baseline", exist_ok=True)
    
    # Run the standard evaluation loop using our existing evaluator
    eval_metrics = run_full_evaluation(
        trainer=trainer,
        test_dataset=test_dataset,
        test_sentences=test_sentences,
        id2label=id2label,
        mode="baseline",
        output_dir="results/baseline",
        training_stats={}
    )
    
    print("\n======================================")
    print("   BASELINE (UNTRAINED) RESULTS ")
    print("======================================")
    print(f"Precision : {eval_metrics['overall']['precision']*100:.2f}%")
    print(f"Recall    : {eval_metrics['overall']['recall']*100:.2f}%")
    print(f"F1-Score  : {eval_metrics['overall']['f1']*100:.2f}%")
    print("======================================\n")

if __name__ == "__main__":
    evaluate_untrained_baseline()

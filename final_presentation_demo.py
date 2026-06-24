import os
import json
import time
import torch
import warnings

# Keep terminal clean from annoying HuggingFace warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.config import Config

def print_project_brief():
    print("=====================================================================")
    print("     FINAL PROJECT: ABBREVIATION & ACRONYM EXTRACTION USING AI")
    print("     Developed by Himanshu Yadav")
    print("=====================================================================\n")
    print("PROJECT IDEA:")
    print("Identifying acronyms and their long-form definitions is a major challenge in")
    print("processing medical, scientific, and technical texts. This project uses AI to")
    print("automatically extract these acronyms and definitions from raw text.\n")
    print("WORKING CONCEPT:")
    print("We framed this as a 'Named Entity Recognition' (token classification) problem.")
    print("We fine-tuned a Large Language Model (FLAN-T5) using the HuggingFace framework.")
    print("The model tags each word as either an abbreviation (B-short), a definition ")
    print("(B-long), or outside/normal text (O).\n")
    print("=====================================================================\n")

def load_and_print_metrics():
    print("COMPARISON: UNTRAINED BASELINE vs. FINE-TUNED MODEL\n")
    
    # We use hardcoded metrics for a smooth presentation, though they can be loaded from json
    baseline_metrics = {"precision": 0.0178, "recall": 0.2585, "f1": 0.0333}
    trained_metrics = {"precision": 0.2865, "recall": 0.7912, "f1": 0.4207}

    print("                 |  Untrained Model  |  Fine-Tuned Model |")
    print("----------------------------------------------------------")
    print(f" Precision       |       {baseline_metrics['precision']*100:>5.2f}%    |       {trained_metrics['precision']*100:>5.2f}%    |")
    print(f" Recall          |       {baseline_metrics['recall']*100:>5.2f}%    |       {trained_metrics['recall']*100:>5.2f}%    |")
    print(f" F1-Score        |       {baseline_metrics['f1']*100:>5.2f}%    |       {trained_metrics['f1']*100:>5.2f}%    |")
    print("----------------------------------------------------------")
    print("\nCONCLUSION: As seen above, the untrained model completely fails at this task.")
    print("After our fine-tuning process, the model successfully learned to extract acronyms!\n")
    print("=====================================================================\n")

def extract_entities(sentence, model, tokenizer, id2label):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    results = []
    for token, pred_id in zip(tokens, predictions):
        if token in ["<s>", "</s>", "<pad>"]:
            continue
            
        label = id2label[pred_id]
        
        # Clean up T5 subword formatting to look nice in the terminal
        clean_token = token.replace("\u2581", "")
        
        if label != "O":
            results.append(f"[{label:<7}] -> {clean_token}")

    if not results:
        return "  (No abbreviations found)"
    return "\n  ".join(results)

def run_demonstration():
    print_project_brief()
    time.sleep(1)
    load_and_print_metrics()
    time.sleep(1)
    
    config = Config()
    
    print("Loading AI Models into memory... (This will take a moment)\n")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name_full)
    
    id2label = {0: "O", 1: "B-short", 2: "I-short", 3: "B-long", 4: "I-long"}
    label2id = {"O": 0, "B-short": 1, "I-short": 2, "B-long": 3, "I-long": 4}
    
    # 2. Load Untrained Model
    untrained_model = AutoModelForTokenClassification.from_pretrained(
        config.training.model_name_full,
        num_labels=5,
        id2label=id2label,
        label2id=label2id
    )
    
    # 3. Load Trained Model
    trained_model_path = "results/full_ft/model"
    # Fallback to local path if running locally
    if not os.path.exists(trained_model_path):
        trained_model_path = "results/full/model"
        if not os.path.exists(trained_model_path):
            trained_model_path = "results/full/saved_model"
            
    try:
        trained_model = AutoModelForTokenClassification.from_pretrained(
            trained_model_path,
            num_labels=5,
            id2label=id2label,
            label2id=label2id
        )
    except:
        print(f"Error: Trained model not found at {trained_model_path}.")
        return
        
    print("Models loaded successfully!\n")
    print("=====================================================================")
    print("                 AUTOMATED TEST CASES (DEMO)")
    print("=====================================================================\n")
    
    test_cases = [
        "The National Aeronautics and Space Administration ( NASA ) launched a new rocket.",
        "A Convolutional Neural Network ( CNN ) is very good at identifying images.",
        "The World Health Organization ( WHO ) announced new guidelines.",
        "He went to the ATM to get some cash.",
        "The patient has Acute Respiratory Distress Syndrome ( ARDS ) and needs oxygen."
    ]
    
    for i, sentence in enumerate(test_cases, 1):
        print(f"TEST {i}: {sentence}")
        untrained_result = extract_entities(sentence, untrained_model, tokenizer, id2label)
        trained_result = extract_entities(sentence, trained_model, tokenizer, id2label)
        
        print("  [UNTRAINED AI GUESS]:")
        print(f"  {untrained_result}")
        print("  [TRAINED AI EXTRACTION]:")
        print(f"  {trained_result}\n")
        time.sleep(0.5)
        
    print("=====================================================================")
    print("                 INTERACTIVE LIVE TESTING")
    print("=====================================================================\n")
    print("Now you can test the models side-by-side with your own sentences!")
    
    while True:
        try:
            user_input = input("\nEnter a custom sentence to test (or type 'quit' to exit):\n> ")
            if user_input.strip().lower() in ['quit', 'exit', 'q']:
                print("\nThank you for exploring the abbreviation extraction project. Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            print("\nPROCESSING...")
            untrained_result = extract_entities(user_input, untrained_model, tokenizer, id2label)
            trained_result = extract_entities(user_input, trained_model, tokenizer, id2label)
            
            print("\n  [UNTRAINED AI GUESS]:")
            print(f"  {untrained_result}")
            print("\n  [TRAINED AI EXTRACTION]:")
            print(f"  {trained_result}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting interactive loop.")
            break

if __name__ == "__main__":
    run_demonstration()

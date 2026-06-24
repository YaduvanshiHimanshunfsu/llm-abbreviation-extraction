import time
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ==============================================================================
# PROJECT OVERVIEW & SETUP DETAILS
# ==============================================================================
# 
# What we built:
# An AI system that acts like a reader, automatically identifying short 
# abbreviations (like "WHO") and their long definitions (like "World Health 
# Organization") within sentences.
#
# How we trained it:
# We tested two approaches on a dataset of over 37,000 entities:
#   1. LoRA Fine-tuning on a 250M parameter model (took about an hour).
#   2. Full Fine-tuning on a 35M parameter model (took about 10 minutes).
#
# Our big optimization:
# We used a technique called "Dynamic Padding". Instead of forcing all sentences 
# to take up the maximum 256 words of memory, we only padded each batch to 
# the length of its longest sentence (around 32 words). This made training 
# incredibly fast and dropped GPU usage below 3%.
#
# The findings:
# The tiny fully-trained model (35M) performed exactly as well as the massive 
# model! So, this script loads up that lightweight, highly efficient model to 
# show you its capabilities in real-time.
# ==============================================================================

import json

def print_project_summary():
    """Prints a beautiful dashboard explaining the project and showing final P, R, F1 scores."""
    print("=" * 65)
    print("   ABBREVIATION EXTRACTION AI - FINAL PROJECT DEMO   ")
    print("=" * 65)
    time.sleep(0.5)
    print("Project Stats:")
    print(" - Task: Named Entity Recognition (NER)")
    print(" - Model Loaded: google/flan-t5-small (Full Fine-Tuned)")
    print(" - Parameters Trained: 35,335,365")
    
    # Let's dynamically load our actual P, R, F1 scores!
    try:
        with open("results/full_ft/metrics/evaluation_results.json", "r") as f:
            metrics = json.load(f)
            p = metrics["overall"]["precision"] * 100
            r = metrics["overall"]["recall"] * 100
            f1 = metrics["overall"]["f1"] * 100
            print("-" * 30)
            print("  FINAL TEST SET METRICS:")
            print(f"    * Precision : {p:.2f}%")
            print(f"    * Recall    : {r:.2f}%")
            print(f"    * F1-Score  : {f1:.2f}%")
            print("-" * 30)
    except Exception as e:
        print(f" - (Error loading metrics: {e})")

    print("=" * 65)
    print("\nInitializing model engine... please wait...\n")

def run_prediction(text: str, model_path: str, model, tokenizer):
    """
    Takes a single sentence, runs it through our custom-trained AI, 
    and cleanly maps the BIO tags back to the words.
    """
    print(f"INPUT SENTENCE:\n> {text}\n")
    time.sleep(0.5)
    
    # 1. Tokenize the input sentence just like we did during training
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    # We grab the actual token strings so we can print them later
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # 2. Feed the sentence into the brain of the AI
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
        # The model returns raw logits (confidence scores for each tag). 
        # We grab the tag with the highest score using argmax.
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # Sometimes, a tiny sentence results in a single integer instead of a list. 
    # Let's wrap it in a list to prevent crashes.
    if not isinstance(predictions, list):
        predictions = [predictions]

    # 3. Map the raw numbers back to our human-readable BIO tags
    id2label = {0: 'O', 1: 'B-short', 2: 'I-short', 3: 'B-long', 4: 'I-long'}

    print("AI EXTRACTION:")
    print("-" * 40)
    
    found_something = False
    
    # 4. Loop through the tokens and print out the ones the AI flagged
    for token, pred_id in zip(tokens, predictions):
        # We don't care about the padding or end-of-sentence markers
        if token in ['<pad>', '</s>', '[PAD]', '[CLS]', '[SEP]']:
            continue
            
        tag = id2label.get(pred_id, 'O')
        
        # If the tag is 'O', the AI thinks it's just a normal word. 
        # We only want to print the special abbreviation words!
        if tag != 'O':
            found_something = True
            
            # T5 uses a weird underscore character (U+2581) to represent spaces. 
            # Let's clean that up so it looks nice and doesn't crash Windows terminals.
            clean_token = token.replace('\u2581', '')
            
            # Print the word nicely formatted
            print(f"  [ {tag:<7} ] -> {clean_token}")

    if not found_something:
        print("  (The AI found no abbreviations in this sentence)")
        
    print("-" * 40 + "\n")
    time.sleep(1)


if __name__ == "__main__":
    # Point this to the full fine-tuned model you just downloaded
    MODEL_DIR = "results/full_ft/model"
    
    # Show the cool intro dashboard
    print_project_summary()

    # Let's load the model and tokenizer into memory once so it's fast
    try:
        my_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        my_model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure your model files exist in: {MODEL_DIR}")
        exit()

    # Let's test the absolute limits of our AI with some tricky sentences
    test_cases = [
        # Test 1: A very standard medical sentence
        "The patient was diagnosed with severe Acute Respiratory Distress Syndrome ( ARDS ).",
        
        # Test 2: Multiple abbreviations in a single sentence
        "The World Health Organization ( WHO ) and the Centers for Disease Control ( CDC ) are working together.",
        
        # Test 3: Tricky case where there are brackets but NO abbreviation
        "He went to the store (which was closed) and went home.",
        
        # Test 4: An abbreviation that appears with NO definition attached
        "We used NLP to parse the dataset quickly."
    ]

    for sentence in test_cases:
        run_prediction(sentence, MODEL_DIR, my_model, my_tokenizer)

    print("\n" + "=" * 65)
    print("   INTERACTIVE MODE ENABLED   ")
    print("=" * 65)
    
    # ---------------------------------------------------------
    # INTERACTIVE TERMINAL LOOP
    # ---------------------------------------------------------
    while True:
        user_input = input("\nEnter a custom sentence to test (or type 'quit' to exit):\n> ")
        
        # Check if the user wants to quit
        if user_input.strip().lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode. Congratulations[ developed by Himanshu Yadav]")
            break
            
        # Ignore empty inputs
        if user_input.strip() == "":
            continue
            
        # Run the AI prediction on their custom sentence!
        run_prediction(user_input, MODEL_DIR, my_model, my_tokenizer)

# Abbreviation and Acronym Extraction using LLMs

This repository contains the code and documentation for my internship final project. The goal of this project was to fine-tune a Large Language Model (LLM) to perform Named Entity Recognition (NER) specifically for extracting abbreviations, acronyms, and their corresponding long-form definitions from text.

## Project Overview

Identifying acronyms and their definitions is a common challenge in processing medical, scientific, and technical texts. In this project, we treated acronym extraction as a token classification problem. 

The model was trained to identify four specific tags using the standard BIO format:
- `B-short` / `I-short`: The abbreviation/acronym itself (e.g., NASA)
- `B-long` / `I-long`: The expanded definition of the acronym (e.g., National Aeronautics and Space Administration)
- `O`: Outside (normal words)

We experimented with two different fine-tuning approaches using HuggingFace's `transformers` library:
1. **LoRA (Low-Rank Adaptation):** Fine-tuning the larger `flan-t5-base` model by injecting lightweight adapter layers, keeping the base model frozen to save memory.
2. **Full Fine-Tuning:** Completely retraining all weights of the smaller `flan-t5-small` model.

Ultimately, the full fine-tuning of the smaller model yielded better domain adaptation for this specific sequence labeling task.

## Repository Structure

- `/src/` - Core source code (data processing, model configuration, training loops)
- `/data/` - Directory for CoNLL-formatted train, valid, and test text datasets
- `colab_train.py` - Google Colab specific script for training with Google Drive checkpointing
- `run_training.py` - Main script for running training pipelines locally
- `test_inference.py` - An interactive terminal script used to test the trained model on custom sentences
- `evaluate_baseline.py` - A script to evaluate the completely untrained model to establish a zero-shot baseline.
- `final_presentation_demo.py` - A comprehensive presentation script that compares the untrained and trained models side-by-side.

## How to Run

### 1. Training the Model
To train the model from scratch, ensure your datasets are in `data/raw/` and run:
```bash
python run_training.py --mode full
```

### 2. Testing the Model interactively
If you want to test the trained model live in your terminal, run the inference script. It will first run through hardcoded test cases and then open an interactive loop where you can type your own sentences.
```bash
python test_inference.py
```

### 3. Running the Final Presentation Demo
To see a complete side-by-side comparison of the untrained model versus the fine-tuned model, run the final presentation script. This will print the project summary, show the metrics comparison, run automated tests, and allow you to test custom sentences interactively.
```bash
python final_presentation_demo.py
```

### 4. Evaluating the Zero-Shot Baseline
To prove the effectiveness of the fine-tuning, you can evaluate the raw, untrained model using the baseline script. This will output a near 0% F1 score, demonstrating that the model could not perform the task prior to our training pipeline.
```bash
python evaluate_baseline.py
```

## Results & Evaluation

The final model achieved the following metrics on our `test.txt` dataset:
- **Recall:** 79.12%
- **Precision:** 28.65%
- **F1-Score:** 42.07%

While the recall is quite strong (successfully identifying ~80% of true abbreviations), the precision is noticeably lower. This discrepancy was primarily caused by a massive class imbalance in the provided test dataset (1,824 short acronyms vs. only 25 long definitions), causing the model to over-predict abbreviations in the absence of balanced long-form examples. Future work would involve dataset balancing and potentially transitioning from a seq2seq model to an encoder-only architecture like DeBERTa for stricter token classification.
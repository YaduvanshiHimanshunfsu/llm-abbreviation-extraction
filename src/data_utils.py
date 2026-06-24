# ============================================================
# data_utils.py — Data Loading & Preprocessing Pipeline
# ============================================================
# Handles:
#   1. Parsing CoNLL format (word\ttag per line, blank=sentence)
#   2. Normalizing tag casing (B-LONG → B-long)
#   3. Converting to HuggingFace Dataset objects
#   4. Tokenization with subword alignment
#   5. Dataset statistics reporting
# ============================================================

import os
import re
import json
import time
import warnings
from typing import List, Tuple, Dict, Optional
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


# ------------------------------------------------------------
#  STEP 1: Parse CoNLL Format Files
# ------------------------------------------------------------

def parse_conll_file(filepath: str, normalize_tags: bool = True) -> List[Dict]:
    """
    Parse a CoNLL-format file into a list of sentences.
    
    Each sentence is a dict with:
      - 'words': list of tokens
      - 'tags':  list of BIO tags (same length as words)
    
    Format expected:
      word1\\tTAG1
      word2\\tTAG2
      (blank line = sentence boundary)
    
    Args:
        filepath: Path to the CoNLL file
        normalize_tags: If True, lowercase all tags for consistency
        
    Returns:
        List of sentence dicts
    """
    sentences = []
    current_words = []
    current_tags = []
    
    console.print(f"  📄 Reading: [cyan]{filepath}[/cyan]")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace and carriage returns
            line = line.strip()
            
            if line == "":
                # -- Blank line = end of sentence --
                if current_words:
                    sentences.append({
                        'words': current_words,
                        'tags': current_tags
                    })
                    current_words = []
                    current_tags = []
            else:
                # -- Parse word and tag --
                parts = line.split('\t')
                if len(parts) == 2:
                    word, tag = parts
                    
                    # Normalize tag casing (B-LONG → B-long, I-SHORT → I-short)
                    if normalize_tags:
                        tag = normalize_bio_tag(tag)
                    
                    # BIO Tag Corrector (Issue 8): Convert orphaned I- to B-
                    if tag.startswith("I-"):
                        entity_type = tag[2:]
                        if not current_tags:
                            tag = f"B-{entity_type}"
                        else:
                            prev_tag = current_tags[-1]
                            if prev_tag == "O" or (prev_tag.startswith("B-") and prev_tag[2:] != entity_type) or (prev_tag.startswith("I-") and prev_tag[2:] != entity_type):
                                tag = f"B-{entity_type}"
                    
                    current_words.append(word)
                    current_tags.append(tag)
                else:
                    # Handle malformed lines (Bug 5)
                    warnings.warn(f"Malformed CoNLL line skipped in {os.path.basename(file_path)} at line {line_num}: {repr(line)}")
    
    # Don't forget the last sentence if file doesn't end with blank line
    if current_words:
        sentences.append({
            'words': current_words,
            'tags': current_tags
        })
    
    return sentences


def normalize_bio_tag(tag: str) -> str:
    """
    Normalize BIO tag to consistent lowercase format.
    
    Examples:
        'B-LONG'  → 'B-long'
        'I-SHORT' → 'I-short'
        'B-short' → 'B-short' (already normalized)
        'O'       → 'O'
    """
    if tag == 'O' or tag == 'o':
        return 'O'
    
    # Split on first hyphen: 'B-LONG' → ['B', 'LONG']
    if '-' in tag:
        prefix, entity_type = tag.split('-', 1)
        prefix = prefix.upper()       # B or I
        entity_type = entity_type.lower()  # long, short
        return f"{prefix}-{entity_type}"
    
    return tag


# ------------------------------------------------------------
#  STEP 2: Compute & Display Dataset Statistics
# ------------------------------------------------------------

def compute_dataset_stats(sentences: List[Dict], split_name: str) -> Dict:
    """
    Compute and display statistics for a dataset split.
    
    Returns a dict with counts for entities, tokens, sentences.
    """
    total_tokens = 0
    tag_counts = Counter()
    entity_counts = {"short": 0, "long": 0}
    
    for sent in sentences:
        total_tokens += len(sent['words'])
        for tag in sent['tags']:
            tag_counts[tag] += 1
            # Count entities by their B- tags
            if tag.startswith('B-short') or tag.startswith('B-SHORT'):
                entity_counts["short"] += 1
            elif tag.startswith('B-long') or tag.startswith('B-LONG'):
                entity_counts["long"] += 1
    
    stats = {
        "split": split_name,
        "num_sentences": len(sentences),
        "num_tokens": total_tokens,
        "avg_sentence_length": round(total_tokens / max(len(sentences), 1), 1),
        "tag_distribution": dict(tag_counts),
        "entity_counts": entity_counts,
        "total_entities": entity_counts["short"] + entity_counts["long"]
    }
    
    return stats


def print_dataset_summary(all_stats: List[Dict]):
    """Print a rich formatted table summarizing all dataset splits."""
    
    console.print("\n" + "-" * 62)
    console.print("  📊 [bold cyan]DATASET STATISTICS[/bold cyan]")
    console.print("-" * 62)
    
    # -- Summary Table --
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", width=10)
    table.add_column("Sentences", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Avg Len", justify="right")
    table.add_column("Short Entities", justify="right", style="green")
    table.add_column("Long Entities", justify="right", style="yellow")
    table.add_column("Total Entities", justify="right", style="bold")
    
    for stats in all_stats:
        table.add_row(
            stats["split"],
            f"{stats['num_sentences']:,}",
            f"{stats['num_tokens']:,}",
            f"{stats['avg_sentence_length']}",
            f"{stats['entity_counts']['short']:,}",
            f"{stats['entity_counts']['long']:,}",
            f"{stats['total_entities']:,}",
        )
    
    console.print(table)
    
    # -- Tag Distribution for first split --
    console.print("\n  📋 [bold]Tag Distribution (Train):[/bold]")
    train_stats = all_stats[0]
    for tag, count in sorted(train_stats["tag_distribution"].items()):
        pct = count / train_stats["num_tokens"] * 100
        bar = "-" * int(pct / 2)
        console.print(f"    {tag:<10} {count:>8,}  ({pct:5.1f}%)  {bar}")
    
    console.print("")


# ------------------------------------------------------------
#  STEP 3: NER Dataset for Token Classification
# ------------------------------------------------------------

class NERDataset(Dataset):
    """
    PyTorch Dataset for token-classification NER.
    
    Tokenizes sentences using a HuggingFace tokenizer and aligns
    BIO labels with subword tokens. Subword tokens that are NOT the
    first piece of a word get label_id = -100 (ignored in loss).
    
    Args:
        sentences: List of sentence dicts with 'words' and 'tags'
        tokenizer: HuggingFace tokenizer
        label2id: Mapping from tag string to integer id
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        sentences: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Tokenize a single sentence and align labels.
        
        Returns dict with:
          - input_ids: token ids
          - attention_mask: 1 for real tokens, 0 for padding
          - labels: aligned label ids (-100 for subwords & padding)
        """
        sentence = self.sentences[idx]
        words = sentence['words']
        tags = sentence['tags']
        
        # -- Tokenize with word-level alignment --
        # is_split_into_words=True tells the tokenizer that input is pre-tokenized
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
        )
        
        # -- Align labels with subword tokens --
        word_ids = encoding.word_ids()  # Maps each token to its word index
        aligned_labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # Special tokens ([CLS], [SEP], [PAD]) → ignore
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word → use the word's label
                if word_id < len(tags):
                    aligned_labels.append(self.label2id.get(tags[word_id], 0))
                else:
                    aligned_labels.append(-100)
            else:
                # Continuation subword → ignore (set to -100)
                aligned_labels.append(-100)
            
            previous_word_id = word_id
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': aligned_labels,
        }


# ------------------------------------------------------------
#  STEP 4: Build Label Mappings
# ------------------------------------------------------------

def build_label_mappings(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build bidirectional label ↔ id mappings.
    
    Args:
        label_list: List of unique tags (e.g., ["O", "B-short", ...])
        
    Returns:
        (label2id, id2label) dicts
    """
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    return label2id, id2label


# ------------------------------------------------------------
#  STEP 5: Full Data Pipeline
# ------------------------------------------------------------

def load_and_prepare_data(config, tokenizer):
    """
    Complete data pipeline:
      1. Parse all three splits from CoNLL files
      2. Print dataset statistics  
      3. Create NERDataset objects with tokenization
      
    Args:
        config: Config object with data paths and settings
        tokenizer: HuggingFace tokenizer
        
    Returns:
        (train_dataset, valid_dataset, test_dataset, 
         label2id, id2label, test_sentences)
    """
    console.print("\n----------------------------------------------------------------")
    console.print("-        📦 [bold cyan]STEP 1: Loading & Preprocessing Data[/bold cyan]              -")
    console.print("----------------------------------------------------------------\n")
    
    start_time = time.time()
    
    # -- Parse CoNLL files --
    console.print("  [bold]Parsing CoNLL files...[/bold]")
    train_sentences = parse_conll_file(config.data.train_file, config.data.normalize_tags)
    valid_sentences = parse_conll_file(config.data.valid_file, config.data.normalize_tags)
    test_sentences = parse_conll_file(config.data.test_file, config.data.normalize_tags)
    
    # -- Compute and display statistics --
    train_stats = compute_dataset_stats(train_sentences, "Train")
    valid_stats = compute_dataset_stats(valid_sentences, "Valid")
    test_stats = compute_dataset_stats(test_sentences, "Test")
    print_dataset_summary([train_stats, valid_stats, test_stats])
    
    # -- Build label mappings --
    label2id, id2label = build_label_mappings(config.data.label_list)
    console.print(f"  🏷️  Label mapping: {label2id}\n")
    
    # -- Create NERDataset objects --
    console.print("  [bold]Tokenizing datasets...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task1 = progress.add_task("  Tokenizing train set...", total=None)
        train_dataset = NERDataset(train_sentences, tokenizer, label2id, config.data.max_length)
        progress.update(task1, completed=True, description=f"  Train: {len(train_dataset):,} samples")
        
        task2 = progress.add_task("  Tokenizing valid set...", total=None)
        valid_dataset = NERDataset(valid_sentences, tokenizer, label2id, config.data.max_length)
        progress.update(task2, completed=True, description=f"  Valid: {len(valid_dataset):,} samples")
        
        task3 = progress.add_task("  Tokenizing test set...", total=None)
        test_dataset = NERDataset(test_sentences, tokenizer, label2id, config.data.max_length)
        progress.update(task3, completed=True, description=f"  Test:  {len(test_dataset):,} samples")
    
    elapsed = time.time() - start_time
    console.print(f"\n   Data preprocessing completed in [green]{elapsed:.1f}s[/green]\n")
    
    # -- Save stats for later reference --
    all_stats = {"train": train_stats, "valid": valid_stats, "test": test_stats}
    
    return train_dataset, valid_dataset, test_dataset, label2id, id2label, test_sentences, all_stats


# ------------------------------------------------------------
#  Utility: Quick data verification
# ------------------------------------------------------------

def verify_dataset_sample(dataset: NERDataset, tokenizer, id2label: Dict, n: int = 3):
    """Print a few samples from the dataset for visual verification."""
    
    console.print("  🔍 [bold]Sample verification:[/bold]\n")
    
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        labels = sample['labels']
        
        console.print(f"  [cyan]Sample {i+1}:[/cyan]")
        parts = []
        for tok, lab in zip(tokens, labels):
            if tok in ['[PAD]', '<pad>', '</s>']:
                continue
            if lab == -100:
                parts.append(f"{tok}")
            else:
                tag = id2label.get(lab, "?")
                if tag != "O":
                    parts.append(f"[bold green]{tok}[/bold green]/{tag}")
                else:
                    parts.append(f"{tok}")
        
        console.print("    " + " ".join(parts[:40]) + " ...")
        console.print("")

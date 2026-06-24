# 🧠 NER Fine-Tuning: Abbreviation/Acronym Detection

## Google Colab Setup Guide (Resilient with Auto-Resume)

> ⚡ **Key Feature**: Training automatically resumes from the last checkpoint.
> If Colab disconnects, just re-run the same cells — no progress is lost!

---

## 📋 Prerequisites

- A **Google account** (for Google Colab access)
- Your 3 data files: `train.txt`, `valid.txt`, `test.txt`
- This project folder uploaded to Google Drive

---

## 🚀 STEP 0: Upload Project to Google Drive

1. **Zip this entire folder** (`Nit_trichy/`) on your laptop
2. Upload `Nit_trichy.zip` to your **Google Drive** root folder
3. Unzip it so you have: `My Drive/Nit_trichy/`

Your Drive should look like:
```
My Drive/
  └── Nit_trichy/
      ├── train.txt
      ├── valid.txt
      ├── test.txt
      ├── colab_train.py      ← The single script you run
      ├── run_training.py
      ├── requirements.txt
      ├── src/
      │   ├── __init__.py
      │   ├── config.py
      │   ├── data_utils.py
      │   ├── model_utils.py
      │   ├── trainer.py
      │   └── evaluator.py
      └── ...
```

---

## 🚀 STEP 1: Open Google Colab & Set GPU

1. Go to **[Google Colab](https://colab.research.google.com/)**
2. Click **"New Notebook"**
3. **IMPORTANT: Enable GPU**:
   - Go to **Runtime → Change runtime type**
   - Set **Hardware accelerator** → **T4 GPU**
   - Click **Save**

---

## 🚀 STEP 2: Run Training (Just 3 Cells!)

### Cell 1: Mount Google Drive
```python
# ============================================================
# CELL 1: Mount Google Drive & Navigate to Project
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Nit_trichy')
print(f"✅ Working directory: {os.getcwd()}")
print(f"📂 Files: {os.listdir('.')}")
```

### Cell 2: Run LoRA Training (with Auto-Resume)
```python
# ============================================================
# CELL 2: LoRA Fine-Tuning — Auto-Resumes on Disconnect!
# ============================================================
# First run:  Trains from scratch
# Re-run:     Automatically resumes from last checkpoint
#
# Options:
#   --epochs 5        Number of epochs (default: 5)
#   --max-hours 3.5   Time limit before graceful stop (default: 3.5)
#   --save-steps 200  Save checkpoint frequency (default: 200)
#   --no-resume       Force fresh start (ignore checkpoints)
#   --smoke-test      Quick test (1 epoch, save every 20 steps)
# ============================================================

!python colab_train.py --mode lora --epochs 5
```

### Cell 3: (Optional) Full Fine-Tuning
```python
# ============================================================
# CELL 3: Full Fine-Tuning — Run after LoRA if you want to compare
# ============================================================
# Takes ~2-3 hours on T4 GPU
# Also auto-resumes from checkpoints!

!python colab_train.py --mode full --epochs 5
```

---

## 🔄 What Happens When Colab Disconnects?

```
┌──────────────────────────────────────────────────────────────┐
│                    HOW AUTO-RESUME WORKS                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Session 1:                                                  │
│  ─────────                                                   │
│  [Epoch 1] ✅ → [Epoch 2] ✅ → [Epoch 3] 💾 → ❌ DISCONNECT │
│                                    ↑                         │
│                         checkpoint saved to Drive            │
│                                                              │
│  Session 2 (re-run same cells):                              │
│  ──────────────────────────────                              │
│  🔍 Found checkpoint at epoch 3.0!                           │
│  🔄 Resuming from checkpoint-1500...                         │
│  [Epoch 3] ✅ → [Epoch 4] ✅ → [Epoch 5] ✅ → 🎉 DONE!     │
│                                                              │
│  ✅ All checkpoints saved to Google Drive                    │
│  ✅ Nothing is lost between sessions                         │
└──────────────────────────────────────────────────────────────┘
```

**Steps to resume:**
1. Open the **same Colab notebook** (or create a new one)
2. Run **Cell 1** (mount Drive)
3. Run **Cell 2** (training auto-resumes from last checkpoint)

That's it! The script detects existing checkpoints and continues seamlessly.

---

## 📊 What You See in the Terminal

During training, you'll see rich progress updates:

```
╔════════════════════════════════════════════════════════════════╗
║         🧠 NER Fine-Tuning: Abbreviation Detection           ║
╠════════════════════════════════════════════════════════════════╣
║  Model:        google/flan-t5-base                            ║
║  Mode:         LORA                                           ║
║  GPU:          Tesla T4                                       ║
╠════════════════════════════════════════════════════════════════╣
║  Save Every:   200 steps (~6 saves/epoch)                     ║
║  Time Limit:   3.5 hours                                      ║
║  🔄 RESUMING FROM: checkpoint-1500                            ║
╚════════════════════════════════════════════════════════════════╝

  [████████████░░░░░░░░░░░░░] 48.2%  Step  1200/ 2490  Epoch 2.4/5
  ⏱ 22m/210m  ETA: 24m (14:35)  Loss: 0.0312  GPU: 4.2/15GB

  ───────────────────────────────────────────────────
  📊 Eval @ step 1200 (epoch 2.4, 22m elapsed)
     🟢 F1: 0.8234  P: 0.8156  R: 0.8314  Loss: 0.0412
  ───────────────────────────────────────────────────

  💾 Checkpoint saved at step 1200 (epoch 2.41, elapsed 22m)
```

---

## ⏱️ Expected Timeline

| Step | What Happens | Time |
|------|-------------|------|
| Cell 1 | Mount Drive | ~10 sec |
| Cell 2 | **LoRA training (5 epochs)** | **~30-45 min** |
| Cell 3 | Full FT training (5 epochs) | ~2-3 hours |

**With checkpoints saved every 200 steps, you lose at most ~3 minutes of training on disconnect.**

---

## 🎛️ Advanced Options

### Customize Training
```python
# Shorter time limit (if Colab seems unstable)
!python colab_train.py --mode lora --epochs 5 --max-hours 2.0

# Save more frequently (less data loss risk)
!python colab_train.py --mode lora --epochs 5 --save-steps 100

# Custom learning rate and batch size
!python colab_train.py --mode lora --epochs 5 --lr 5e-5 --batch-size 8

# Quick smoke test (verify everything works)
!python colab_train.py --mode lora --smoke-test

# Force fresh start (delete and retrain from scratch)
!python colab_train.py --mode lora --epochs 5 --no-resume
```

### View Existing Checkpoints
```python
# Quick look at what checkpoints exist
import os
results_dir = '/content/drive/MyDrive/Nit_trichy/results/lora'
if os.path.exists(results_dir):
    for item in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, item)
        if os.path.isdir(path) and item.startswith('checkpoint'):
            size = sum(os.path.getsize(os.path.join(path, f))
                      for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            print(f"  📂 {item} ({size/1024/1024:.0f} MB)")
```

### Download Results
```python
# Zip and download all results
import shutil
from google.colab import files

shutil.make_archive('/content/ner_results', 'zip',
                    '/content/drive/MyDrive/Nit_trichy', 'results')
files.download('/content/ner_results.zip')
print("📥 Download started!")
```

---

## 🔧 Troubleshooting

### "CUDA out of memory"
→ Reduce batch size:
```python
!python colab_train.py --mode lora --batch-size 8
```

### "Module not found"
→ Make sure you're in the right directory:
```python
os.chdir('/content/drive/MyDrive/Nit_trichy')
```

### "Drive disconnected"
→ Colab disconnects after ~90 min of inactivity. Keep the tab active.
→ Even if it disconnects, your checkpoints are safe on Drive!
→ Just re-run Cell 1 + Cell 2 to resume.

### Training too slow
→ Make sure you selected **T4 GPU** in Runtime settings
→ Check GPU: `!nvidia-smi`

### Want to start completely fresh
```python
# Delete all checkpoints and start over
import shutil
shutil.rmtree('/content/drive/MyDrive/Nit_trichy/results/lora', ignore_errors=True)
!python colab_train.py --mode lora --epochs 5 --no-resume
```

---

## 📁 What Gets Saved (to Google Drive)

After training, your `results/` folder on Drive contains:

```
My Drive/Nit_trichy/results/
├── lora/
│   ├── checkpoint-200/          # Checkpoint at step 200
│   ├── checkpoint-400/          # Checkpoint at step 400
│   ├── checkpoint-600/          # Latest checkpoint (rolling)
│   ├── model/                   # Final LoRA adapter weights (~10 MB)
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── tokenizer files...
│   ├── predictions/
│   │   └── test_predictions.txt  # word  true_tag  pred_tag
│   ├── metrics/
│   │   └── evaluation_results.json  # P, R, F1 scores
│   └── training_stats.json      # Time, params, loss curves
│
├── full_ft/                     # Same structure for Full FT
│   ├── checkpoint-XXX/
│   ├── model/
│   ├── predictions/
│   ├── metrics/
│   └── training_stats.json
│
└── comparison_report.json       # Side-by-side comparison
```

**Note:** Only the 3 most recent checkpoints are kept (rolling window).
This prevents Drive from filling up while ensuring you can always resume.

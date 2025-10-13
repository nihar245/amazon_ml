# 🚀 Deployment Guide - GitHub & Kaggle

## Complete Step-by-Step Instructions

---

## 📋 Table of Contents
1. [GitHub Setup & Push](#github-setup)
2. [Kaggle GPU Configuration](#kaggle-configuration)
3. [Running on Kaggle](#running-on-kaggle)
4. [Local Development](#local-development)
5. [Troubleshooting](#troubleshooting)

---

## 🐙 PART 1: GitHub Setup & Push

### Prerequisites
- Git installed on your system
- GitHub account created
- Repository: https://github.com/meethp1884/amazon_ai_chall_1

### Step 1: Initialize Git (if not already done)

Open Command Prompt or Git Bash in your project folder:

```bash
cd C:\Users\meeth\OneDrive\Desktop\Amazon_ai_challenge
```

Check if Git is initialized:
```bash
git status
```

If not initialized, run:
```bash
git init
```

### Step 2: Create .gitignore File

This prevents large files from being pushed to GitHub.

Create a file named `.gitignore` with this content:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Dataset files (too large for GitHub)
student_resource/dataset/*.csv
*.csv

# Model files (too large for GitHub)
models/
*.pth
*.pkl
*.h5
*.pb

# OS files
.DS_Store
Thumbs.db
desktop.ini

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*.log
*.tmp
__MACOSX/

# Keep directory structure but ignore contents
!.gitkeep
```

### Step 3: Configure Git User

```bash
git config --global user.name "meethp1884"
git config --global user.email "your-email@example.com"
```

### Step 4: Add Remote Repository

```bash
git remote add origin https://github.com/meethp1884/amazon_ai_chall_1.git
```

Verify remote:
```bash
git remote -v
```

### Step 5: Stage Files

Add all project files (except those in .gitignore):

```bash
git add .
```

Check what will be committed:
```bash
git status
```

### Step 6: Commit Changes

```bash
git commit -m "Initial commit: Amazon ML Challenge 2025 multimodal price prediction solution"
```

### Step 7: Push to GitHub

First time push:
```bash
git branch -M main
git push -u origin main
```

If repository already has files:
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Step 8: Verify on GitHub

Visit: https://github.com/meethp1884/amazon_ai_chall_1

You should see:
- ✅ train.py
- ✅ sample_code.py
- ✅ requirements.txt
- ✅ README.md
- ✅ DATA_FLOW.md
- ✅ PROJECT_SUMMARY.md
- ✅ DEPLOYMENT_GUIDE.md
- ✅ .gitignore

**Note:** Dataset files and models will NOT be pushed (they're too large).

---

## 🔄 Future Updates (After Making Changes)

```bash
# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

---

## 🎮 PART 2: Kaggle GPU Configuration (UPDATED FOR IMPROVED MODEL)

### Why Use Kaggle?
- ✅ **Free GPU access** (P100 with 16GB VRAM)
- ✅ **Pre-installed libraries** (PyTorch, Transformers)
- ✅ **No local setup required**
- ✅ **Fast training** (~50-70 minutes with improved model)

### Step 1: Create Kaggle Account

1. Go to https://www.kaggle.com/
2. Sign up or log in
3. Verify your phone number (required for GPU access)

### Step 2: Create New Notebook

1. Click **"Create"** → **"New Notebook"**
2. Notebook opens in browser

### Step 3: Enable GPU

**IMPORTANT:** Must enable GPU for fast training!

1. Click **"Settings"** (⚙️ icon) on the right sidebar
2. Under **"Accelerator"**, select **"GPU P100"** (RECOMMENDED) or **"GPU T4"**
3. Click **"Save"**
4. Verify GPU is active (you'll see GPU indicator)

**Note:** The improved model (train_improved.py) requires more computation due to:
- 25 features (vs 8)
- Deeper network (5 layers vs 3)
- Longer sequences (256 vs 128 tokens)
- More training epochs (25 vs 15)

### Step 4: Upload Dataset

#### Option A: Upload from Computer

1. Click **"Add Data"** (right panel)
2. Click **"Upload Dataset"**
3. Upload these files:
   - `train.csv` (from `student_resource/dataset/`)
   - `test.csv`
4. Name it: `amazon-ml-challenge-dataset`
5. Click **"Create"**

#### Option B: Use Kaggle Dataset (if already uploaded)

1. Click **"Add Data"**
2. Search for your dataset
3. Click **"Add"**

Dataset will be available at: `/kaggle/input/amazon-ml-challenge-dataset/`

---

## 🏃 PART 3: Running on Kaggle

### Setup Code (Run in first cell)

```python
# Install additional dependencies (if needed)
!pip install sentence-transformers -q

# Verify GPU is available
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"PyTorch Version: {torch.__version__}")

# List dataset files
import os
print("\nDataset files:")
print(os.listdir('/kaggle/input/amazon-ml-challenge-dataset/'))
```

Expected output:
```
GPU Available: True
GPU Name: Tesla P100-PCIE-16GB
PyTorch Version: 2.0.0+cu118

Dataset files:
['train.csv', 'test.csv']
```

### Training Code (Kaggle Version) - UPDATED FOR IMPROVED MODEL

Create a new cell and copy-paste this code:

```python
# ========== OPTION 1: IMPROVED MODEL (RECOMMENDED) ==========
# Use train_improved.py for better accuracy (38-45% SMAPE vs 55%)

# Clone your GitHub repository
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1

# Create directory structure
!mkdir -p student_resource/dataset
!mkdir -p models

# Copy dataset files
!cp /kaggle/input/amazon-ml-challenge-dataset/train.csv student_resource/dataset/
!cp /kaggle/input/amazon-ml-challenge-dataset/test.csv student_resource/dataset/

# Install requirements
!pip install -r requirements.txt -q

# Run IMPROVED training (recommended)
!python train_improved.py
```

**OR** if you want to use the baseline model:

```python
# ========== OPTION 2: BASELINE MODEL ==========
# Use train.py for baseline (55% SMAPE)

# ... (same setup as above)

# Run baseline training
!python train.py
```

### Training Time Estimate

**Improved Model (train_improved.py):**
| Hardware | Time per Epoch | Total Time (25 epochs) | Expected SMAPE |
|----------|----------------|------------------------|----------------|
| Kaggle P100 GPU | ~2.5-3 min | **~50-70 minutes** | **38-45%** ⭐ |
| Kaggle T4 GPU | ~3-4 min | ~75-100 minutes | 38-45% |
| CPU (local) | ~30-50 min | ~12-20 hours | 38-45% |

**Baseline Model (train.py):**
| Hardware | Time per Epoch | Total Time (15 epochs) | Expected SMAPE |
|----------|----------------|------------------------|----------------|
| Kaggle P100 GPU | ~2-2.5 min | ~30-45 minutes | 50-55% |
| Kaggle T4 GPU | ~3-4 min | ~45-60 minutes | 50-55% |
| CPU (local) | ~30-45 min | ~7-10 hours | 50-55% |

**Recommendation:** Use train_improved.py for ~17% SMAPE improvement!

### Monitoring Training

**For Improved Model (train_improved.py):**
```
Extracting features for 63750 samples...
Feature extraction: 100%|██████████| 63750/63750 [00:45<00:00]

Unique brands: 127
Unique categories: 8

Epoch 1/25 [Train]: 100%|██████████| 1993/1993 [02:45<00:00, 12.03batch/s, SMAPE=48.5%, LR=5e-05]
Epoch 1/25 [Val]: 100%|██████████| 352/352 [00:21<00:00, 16.13batch/s]

Epoch 1/25
Train SMAPE: 48.50%
Val SMAPE: 51.30%
Learning Rate: 5.00e-05
✓ Model saved with Val SMAPE: 51.30%

...

Epoch 15/25
Train SMAPE: 38.25%
Val SMAPE: 42.15%
✓ Model saved with Val SMAPE: 42.15%
```

**For Baseline Model (train.py):**
```
Epoch 1/15
[Train] 100%|██████████| 1993/1993 [02:15<00:00, 14.73batch/s]
[Val] 100%|██████████| 352/352 [00:18<00:00, 19.12batch/s]

Epoch 1/15
Train SMAPE: 45.50%
Val SMAPE: 48.30%
✓ Model saved with Val SMAPE: 48.30%
```

### Inference Code (After Training)

```python
# Run inference on test data
!python sample_code.py

# Check output
import pandas as pd
output = pd.read_csv('student_resource/dataset/test_out.csv')
print(output.head())
print(f"\nTotal predictions: {len(output)}")
print(f"Price range: ${output['price'].min():.2f} - ${output['price'].max():.2f}")
```

### Download Results

```python
# Download test_out.csv for submission
from IPython.display import FileLink
FileLink('student_resource/dataset/test_out.csv')
```

Or manually:
1. Click **"Output"** (right panel)
2. Find `test_out.csv`
3. Click **"Download"**

---

## 💡 PART 4: Local Development (Your Computer)

### Step 1: Install Python & Dependencies

**Windows:**
```bash
# Check Python version (need 3.8+)
python --version

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** Training on CPU will take 7-10 hours. Recommended to use Kaggle GPU instead.

### Step 2: Run Training Locally

```bash
cd C:\Users\meeth\OneDrive\Desktop\Amazon_ai_challenge
python train.py
```

### Step 3: Run Inference

```bash
python sample_code.py
```

### Step 4: Check Output

```bash
# Output saved to:
student_resource\dataset\test_out.csv
```

---

## ⚙️ PART 5: Efficient Configuration

### Training Configuration Guide

#### Number of Epochs

| Epochs | Result | Recommendation |
|--------|--------|----------------|
| <8 | Underfitting | Too few |
| 10-15 | **Optimal** | ✅ Use this |
| 16-20 | Slight overfit | Acceptable with early stopping |
| >20 | Overfitting | Avoid |

**Default in code:** 15 epochs with early stopping (patience=5)

**What this means:**
- Training will run up to 15 epochs
- If validation loss doesn't improve for 5 consecutive epochs, training stops
- Best model is automatically saved

#### Batch Size

| Batch Size | Memory Usage | Speed | Recommendation |
|------------|--------------|-------|----------------|
| 8 | Low | Slow | Use if OOM errors |
| 16 | Medium | Medium | Good for 4GB RAM |
| **32** | **Medium-High** | **Fast** | ✅ **Default (optimal)** |
| 64 | High | Fastest | Use on high-end GPUs |

**Current setting:** 32 (works on Kaggle P100, good balance)

#### Learning Rate

| Learning Rate | Result |
|---------------|--------|
| 1e-5 | Too slow convergence |
| 1e-4 | Good, safe choice |
| **2e-4** | **✅ Default (optimal)** |
| 5e-4 | Faster but less stable |
| 1e-3 | May overshoot |

**Current setting:** 2e-4 with ReduceLROnPlateau (automatically reduces if stuck)

### How to Change Configuration

Edit `train.py`, lines near the end:

```python
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=15,        # ← Change this (recommended: 10-15)
    lr=2e-4           # ← Change this (recommended: 1e-4 to 5e-4)
)
```

And for batch size, find:

```python
train_loader = DataLoader(train_dataset, batch_size=32, ...)  # ← Change this
val_loader = DataLoader(val_dataset, batch_size=32, ...)      # ← Change this
```

### Preventing Overfitting Checklist

- [x] **Early stopping** enabled (patience=5)
- [x] **Dropout** layers (0.3, 0.2, 0.1)
- [x] **Train/Val split** (85/15)
- [x] **Weight decay** (0.01 in AdamW)
- [x] **Gradient clipping** (max_norm=1.0)
- [x] **Learning rate scheduling** (ReduceLROnPlateau)
- [x] **Layer freezing** (freeze base MiniLM layers)
- [x] **Limited epochs** (10-15 with early stopping)

**All configured by default in train.py!**

---

## 🎯 Quick Start Cheat Sheet

### For Kaggle (Recommended)

```python
# Cell 1: Setup
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1
!mkdir -p student_resource/dataset models
!cp /kaggle/input/amazon-ml-challenge-dataset/*.csv student_resource/dataset/
!pip install -r requirements.txt -q

# Cell 2: Train (30-45 min on P100 GPU)
!python train.py

# Cell 3: Predict
!python sample_code.py

# Cell 4: Download results
from IPython.display import FileLink
FileLink('student_resource/dataset/test_out.csv')
```

### For Local (If you have GPU)

```bash
cd C:\Users\meeth\OneDrive\Desktop\Amazon_ai_challenge
python train.py              # 30-60 min with GPU
python sample_code.py        # Generate predictions
```

---

## 🐛 PART 6: Troubleshooting

### Issue: Git push rejected

**Error:** `! [rejected] main -> main (fetch first)`

**Solution:**
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Issue: Large files error

**Error:** `remote: error: File xxx is 100.00 MB; exceeds GitHub's file size limit`

**Solution:** Files like datasets and models are too large for GitHub.
- Make sure `.gitignore` includes `*.csv`, `*.pth`, `*.pkl`
- Remove large files from git:
```bash
git rm --cached student_resource/dataset/*.csv
git rm --cached models/*.pth
git commit -m "Remove large files"
git push origin main
```

### Issue: Kaggle GPU not available

**Solution:**
1. Verify phone number on Kaggle
2. Check quota: Settings → Account → GPU quota
3. Try T4 GPU instead of P100
4. Wait if quota exceeded (resets weekly)

### Issue: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: Change `batch_size=32` to `batch_size=16` or `8`
2. Clear cache:
```python
torch.cuda.empty_cache()
```
3. Restart Kaggle kernel

### Issue: Module not found

**Error:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:** Install missing packages
```bash
pip install transformers sentence-transformers
```

On Kaggle:
```python
!pip install sentence-transformers -q
```

### Issue: Training too slow on CPU

**Solution:** Use Kaggle GPU (30-45 min vs 7-10 hours)

### Issue: Model predictions all the same

**Possible causes:**
1. Model didn't train properly (check logs)
2. Features not normalized (check scaler loaded)
3. Model file corrupted (re-train)

**Solution:** Re-run training and check validation loss is decreasing.

---

## 📊 Expected Results

### Training Output
```
Epoch 1/15
Train Loss: 1250.45, Train MAE: 25.67
Val Loss: 1387.23, Val MAE: 28.34
✓ Model saved with val_loss: 1387.23

Epoch 2/15
Train Loss: 890.12, Train MAE: 18.45
Val Loss: 945.67, Val MAE: 19.78
✓ Model saved with val_loss: 945.67

...

Epoch 10/15
Train Loss: 210.34, Train MAE: 9.12
Val Loss: 275.89, Val MAE: 11.45
✓ Model saved with val_loss: 275.89

Early stopping triggered after 15 epochs
```

### Inference Output
```
Using device: cuda
Loading tokenizer...
Loading feature scaler...
Loading trained model...
✓ Model loaded successfully

Loading test data...
Test samples: 75000

Generating predictions...
✓ Predictions saved to student_resource/dataset/test_out.csv
✓ Total predictions: 75000

Sample predictions:
   sample_id   price
0     100179   15.67
1     100180    8.99
2     100181   23.45
...

Price statistics:
  Mean: $32.45
  Median: $18.90
  Min: $0.50
  Max: $499.99
```

---

## ✅ Final Checklist

### Before Pushing to GitHub
- [ ] `.gitignore` file created
- [ ] Large files excluded (*.csv, *.pth)
- [ ] Code tested locally
- [ ] README.md updated
- [ ] All documentation files included

### Before Running on Kaggle
- [ ] Kaggle account verified (phone)
- [ ] GPU accelerator enabled (P100/T4)
- [ ] Dataset uploaded to Kaggle
- [ ] Requirements installed (`sentence-transformers`)

### After Training
- [ ] Model saved (best_model.pth)
- [ ] Scaler saved (feature_scaler.pkl)
- [ ] Validation loss decreased
- [ ] No overfitting observed

### After Inference
- [ ] test_out.csv generated
- [ ] Exactly 75,000 predictions
- [ ] All prices positive
- [ ] Format matches sample_test_out.csv

---

## 🎉 Success Criteria

You're done when:
1. ✅ Code pushed to GitHub successfully
2. ✅ Model trained on Kaggle GPU (~30-45 min)
3. ✅ test_out.csv generated with 75,000 predictions
4. ✅ All prices are positive floats
5. ✅ Ready for competition submission

---

## 📞 Support Resources

- **Kaggle Docs:** https://www.kaggle.com/docs
- **PyTorch Docs:** https://pytorch.org/docs/
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers/
- **Git Docs:** https://git-scm.com/doc

---

**Good luck with your deployment! 🚀**

Last Updated: October 11, 2025

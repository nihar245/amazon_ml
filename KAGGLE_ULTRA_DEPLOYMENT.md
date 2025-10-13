# 🚀 ULTRA Model - Kaggle Deployment Guide

## Complete Step-by-Step Instructions for DeBERTa-v3-base Model

---

## 📋 Table of Contents
1. [Quick Start](#-quick-start)
2. [Kaggle GPU Configuration](#-kaggle-configuration)
3. [Running on Kaggle](#-running-on-kaggle)
4. [Expected Output](#-expected-output)
5. [Troubleshooting](#-troubleshooting)
6. [FAQ](#-frequently-asked-questions)

---

## ⚡ Quick Start

### **One-Click Kaggle Notebook**
[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/meethp1884/amazon_ai_chall_1/main/ULTRA_kaggle_notebook.ipynb)

### **OR Copy-Paste This Code into a Kaggle Notebook**

```python
# ========== KAGGLE SETUP FOR ULTRA MODEL ==========
# IMPORTANT: This takes 80-100 minutes on P100 GPU
# DO NOT run inference before training!

# 1. Install dependencies (ignore version warnings)
!pip install -q transformers==4.30.0

# 2. Clone repository
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1

# 3. Create necessary directories
!mkdir -p student_resource/dataset models

# 4. Copy competition data
!cp /kaggle/input/amazon-ml-challenge-dataset/train.csv student_resource/dataset/
!cp /kaggle/input/amazon-ml-challenge-dataset/test.csv student_resource/dataset/

# 5. Verify setup
import os
print("\n" + "="*70)
print("✅ ULTRA MODEL - SETUP COMPLETE")
print("="*70)
print(f"Train data exists: {os.path.exists('student_resource/dataset/train.csv')}")
print(f"Test data exists: {os.path.exists('student_resource/dataset/test.csv')}")
print("\nModel: DeBERTa-v3-base (184M params)")
print("Expected: Val SMAPE 37-42% (from 47.5%)")
print("Training time: 80-100 minutes on P100")
print("="*70 + "\n")

# 6. Train the model (80-100 minutes)
# This will create:
#   - models/best_model_ultra.pth
#   - models/feature_scaler_improved.pkl
#   - models/brand_encoder.pkl
#   - models/category_encoder.pkl
!python train_improved.py

# 7. Generate predictions (ONLY after training completes!)
!python sample_code_improved.py

# 8. Verify predictions
!head -n 10 student_resource/dataset/test_out.csv
print("\n✅ Done! Predictions saved to: student_resource/dataset/test_out.csv")
```

---

## ⚙️ Kaggle Configuration

### **Hardware Requirements**
- **GPU:** P100 or better (T4/V100/A100 also work)
- **RAM:** 16GB+ recommended
- **Disk Space:** 5GB free space

### **Environment Setup**
1. Go to [Kaggle Kernels](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "Settings" on the right
4. Set:
   - **Accelerator:** GPU P100 (or better)
   - **Internet:** On (for downloading DeBERTa)
   - **GPU Verifier:** Check with `!nvidia-smi`

### **Required Data Files**
1. Training Data: `train.csv` (from competition dataset)
2. Test Data: `test.csv` (from competition dataset)

---

## 🚀 Running on Kaggle

### **Step 1: Create New Notebook**
1. Go to [Kaggle Kernels](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Set to GPU (P100 or better)

### **Step 2: Run Setup (5 minutes)**
```python
# Install dependencies
!pip install -q transformers==4.30.0 torch==2.0.0

# Clone repository
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1

# Setup directories
!mkdir -p student_resource/dataset models

# Copy competition data
!cp /kaggle/input/amazon-ml-challenge-dataset/train.csv student_resource/dataset/
!cp /kaggle/input/amazon-ml-challenge-dataset/test.csv student_resource/dataset/
```

### **Step 3: Train the Model (80-100 minutes)**
```python
# Train the ULTRA model
!python train_improved.py
```

### **Step 4: Generate Predictions (5 minutes)**
```python
# Generate predictions
!python sample_code_improved.py

# Verify output
!head -n 5 student_resource/dataset/test_out.csv
```

---

## 📊 Expected Output

### **Training Progress**
```
Epoch 1/35
Train Loss: 0.5123 - Val SMAPE: 49.82%
...
Epoch 20/35  
Train Loss: 0.3210 - Val SMAPE: 39.75%  🎉
...
Epoch 35/35
Train Loss: 0.2856 - Val SMAPE: 37.92%  🚀
```

### **Final Output**
```
✅ Training complete! Best model saved to: models/best_model_ultra.pth
✅ Best Validation SMAPE: 37.92%
✅ Predictions saved to: student_resource/dataset/test_out.csv
```

---

## 🐛 Troubleshooting

### **❌ ERROR: "No such file or directory: 'models/feature_scaler_improved.pkl'"**
**Cause:** You're running `sample_code_improved.py` (inference) BEFORE training!

**Solution:**
```python
# ✅ CORRECT ORDER:
# 1. Train first (creates model files)
!python train_improved.py

# 2. Then generate predictions
!python sample_code_improved.py
```

**Why:** The training script creates these files:
- `models/best_model_ultra.pth` (model weights)
- `models/feature_scaler_improved.pkl` (feature normalization)
- `models/brand_encoder.pkl` (brand encoding)
- `models/category_encoder.pkl` (category encoding)

**Without training, these files don't exist!**

---

### **❌ ERROR: "SyntaxError: '(' was never closed"**
**Cause:** Old version of `train_improved.py` with syntax error

**Solution:**
```python
# Pull latest changes from GitHub
!git pull origin main

# Or re-clone
!rm -rf amazon_ai_chall_1
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
```

---

### **⚠️ WARNING: "pip's dependency resolver conflicts"**
**Message:** `transformers 4.30.0 conflicts with sentence-transformers 4.1.0`

**Solution:** **IGNORE IT!** These warnings are safe.
- Kaggle has pre-installed packages
- Our code works despite warnings
- No action needed

---

### **❌ ERROR: "CUDA out of memory"**
**Cause:** GPU memory exhausted (batch size too large)

**Solution:** Reduce batch size in `train_improved.py`:
```python
# Find line ~479 and change:
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)

# If still OOM, try batch_size=4
```

Then restart kernel and try again.

---

### **⚠️ Slow Training**
**Issue:** Training is very slow

**Solution:**
1. Verify GPU is being used:
   ```python
   import torch
   print("GPU available:", torch.cuda.is_available())
   print("GPU name:", torch.cuda.get_device_name(0))
   ```
2. Expected: `Tesla P100-PCIE-16GB`
3. Make sure you selected GPU accelerator in notebook settings

---

### **❌ ERROR: "FileNotFoundError: train.csv"**
**Cause:** Competition data not copied correctly

**Solution:**
```python
# Verify data paths
!ls /kaggle/input/amazon-ml-challenge-dataset/

# Copy again
!mkdir -p student_resource/dataset
!cp /kaggle/input/amazon-ml-challenge-dataset/*.csv student_resource/dataset/

# Verify
!ls -lh student_resource/dataset/
```

**Expected file structure:**
```
amazon_ai_chall_1/
├── train_improved.py
├── sample_code_improved.py
├── student_resource/
│   └── dataset/
│       ├── train.csv (must exist before training)
│       └── test.csv (must exist before inference)
└── models/ (created during training)
    ├── best_model_ultra.pth
    ├── feature_scaler_improved.pkl
    ├── brand_encoder.pkl
    └── category_encoder.pkl
```

---

## ❓ Frequently Asked Questions

### **Q: How long does training take?**
**A:** 
- P100 GPU: 80-100 minutes
- T4 GPU: 100-120 minutes
- CPU: Not recommended (12+ hours)

### **Q: What SMAPE should I expect?**
**A:** 
- **Before (MiniLM):** 47.5% SMAPE
- **After (DeBERTa):** 37-42% SMAPE
- **Improvement:** ~10% reduction

### **Q: How to save model outputs?**
```python
# Save model to Kaggle output
!cp -r models/ /kaggle/working/

# Save predictions
!cp student_resource/dataset/test_out.csv /kaggle/working/
```

### **Q: How to submit to competition?**
1. Download `test_out.csv`
2. Go to competition page
3. Click "Submit Predictions"
4. Upload the file

---

## 🎯 Performance Optimization

### **Faster Training**
1. Use A100 GPU if available
2. Increase batch size (if memory allows)
3. Use mixed precision (already enabled)

### **Better Accuracy**
1. Train for more epochs (try 50)
2. Use DeBERTa-v3-large (change in `train_improved.py`)
3. Ensemble multiple models

---

## 📞 Support

For issues:
1. Check [Troubleshooting](#-troubleshooting)
2. Open an issue on [GitHub](https://github.com/meethp1884/amazon_ai_chall_1/issues)
3. Contact: [Your Email]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated:** October 12, 2025  
**Model Version:** ULTRA v1.0 (DeBERTa-v3-base)  
**Expected SMAPE:** 37-42%  
**Training Time:** 80-100 minutes (P100 GPU)

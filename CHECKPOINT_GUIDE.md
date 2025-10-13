# 💾 CHECKPOINT SYSTEM - Complete Guide

## 🎯 **What is Checkpoint Saving?**

The checkpoint system automatically saves your training progress at **every epoch**, so you can:
1. ✅ **Resume training** if it crashes or gets interrupted
2. ✅ **Download best model** at any point
3. ✅ **Recover from internet failures** on Kaggle
4. ✅ **Continue training** with more epochs later

---

## 📁 **Files Saved During Training:**

### **For Each Model (ULTRA, FAST, BLAZING):**

| File | Purpose | When Saved | Keep? |
|------|---------|------------|-------|
| `checkpoint_{model}_latest.pth` | Most recent epoch | Every epoch (overwrites) | ✅ Yes |
| `checkpoint_{model}_epoch_N.pth` | Specific epoch N | Every epoch (accumulates) | ⚠️ Optional |
| `best_model_{model}.pth` | Best performing model | When SMAPE improves | ✅ **YES!** |

**Example for ULTRA model:**
```
models/
├── checkpoint_ultra_latest.pth     ← Latest checkpoint
├── checkpoint_ultra_epoch_1.pth    ← Epoch 1
├── checkpoint_ultra_epoch_2.pth    ← Epoch 2
├── checkpoint_ultra_epoch_3.pth    ← Epoch 3
├── ...
├── checkpoint_ultra_epoch_20.pth   ← Epoch 20
└── best_model_ultra.pth            ← Best model! ⭐
```

---

## 🔄 **How It Works:**

### **Automatic Saving (Every Epoch):**

```
Epoch 1/20 - Train Loss: 0.4821 - Val SMAPE: 48.12%
💾 Checkpoint saved: epoch 1
⭐ Best model saved with Val SMAPE: 48.12%

Epoch 2/20 - Train Loss: 0.4156 - Val SMAPE: 45.31%
💾 Checkpoint saved: epoch 2
⭐ Best model saved with Val SMAPE: 45.31%

Epoch 3/20 - Train Loss: 0.3821 - Val SMAPE: 43.15%
💾 Checkpoint saved: epoch 3
⭐ Best model saved with Val SMAPE: 43.15%
```

**What's saved in each checkpoint:**
- ✅ Model weights
- ✅ Optimizer state
- ✅ Scheduler state
- ✅ Scaler state (for mixed precision)
- ✅ Current epoch number
- ✅ Training loss
- ✅ Validation SMAPE
- ✅ Best SMAPE so far

---

## 🚀 **How to Resume Training:**

### **Scenario 1: Training Crashed at Epoch 12**

```python
# Original training (crashed at epoch 12)
!python train_improved.py

# Resume from latest checkpoint
# Add this to train_improved.py main() before training:

model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=1e-4,
    gradient_accumulation_steps=4,
    resume_from='models/checkpoint_ultra_latest.pth'  # Add this!
)
```

**Output:**
```
📂 Resuming from checkpoint: models/checkpoint_ultra_latest.pth
✓ Resumed from epoch 13, best SMAPE: 42.15%

Epoch 13/20 - Train Loss: 0.3421 - Val SMAPE: 41.82%
💾 Checkpoint saved: epoch 13
⭐ Best model saved with Val SMAPE: 41.82%
```

---

### **Scenario 2: Resume from Specific Epoch**

```python
# Resume from epoch 10 (instead of latest)
resume_from='models/checkpoint_ultra_epoch_10.pth'
```

---

### **Scenario 3: Add More Epochs Later**

```python
# Original: Trained 20 epochs
# Now: Want to train 5 more (total 25 epochs)

model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,  # Increase from 20 to 25
    lr=1e-4,
    gradient_accumulation_steps=4,
    resume_from='models/best_model_ultra.pth'  # Resume from best
)
```

---

## 📥 **How to Download Checkpoints from Kaggle:**

### **Method 1: Download Best Model Only (Recommended)**

```python
# At the end of your Kaggle notebook, add:

from IPython.display import FileLink
import os

# Show file size
best_model_path = 'models/best_model_ultra.pth'
size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
print(f"Best model size: {size_mb:.2f} MB")

# Create download link
FileLink(best_model_path)
```

**Click the link to download!**

---

### **Method 2: Download All Checkpoints (If You Want Backups)**

```python
# Create a zip of all checkpoints
!zip -r checkpoints.zip models/

# Download the zip
from IPython.display import FileLink
FileLink('checkpoints.zip')
```

---

### **Method 3: Download During Training (Every 5 Epochs)**

Add this inside the training loop:

```python
# In train_improved.py, after checkpoint saving:
if (epoch + 1) % 5 == 0:
    print(f"\n📥 Epoch {epoch+1} checkpoint ready for download!")
    print(f"   File: models/checkpoint_ultra_epoch_{epoch+1}.pth")
```

Then in Kaggle, download it:
```python
from IPython.display import FileLink
FileLink(f'amazon_ai_chall_1/models/checkpoint_ultra_epoch_5.pth')
```

---

## 🛠️ **Common Use Cases:**

### **Use Case 1: Kaggle Session Expired**

**Problem:** Kaggle auto-saves your work, but you lost the session.

**Solution:**
1. Reopen notebook
2. Run setup cells
3. Add `resume_from='models/checkpoint_ultra_latest.pth'`
4. Continue training!

---

### **Use Case 2: Internet Died During Training**

**Problem:** Internet connection lost, can't push to GitHub.

**Solution:**
1. Checkpoints are saved locally
2. When internet returns, download `best_model_ultra.pth`
3. Or continue training from latest checkpoint

---

### **Use Case 3: Want to Try Different Learning Rates**

**Problem:** Trained 10 epochs with lr=1e-4, want to try lr=5e-5 for 10 more.

**Solution:**
```python
# Load checkpoint from epoch 10
model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,  # Train to epoch 20
    lr=5e-5,  # NEW learning rate!
    gradient_accumulation_steps=4,
    resume_from='models/checkpoint_ultra_epoch_10.pth'
)
```

---

### **Use Case 4: Compare Different Epoch Models**

**Problem:** Want to test if epoch 15 is better than epoch 20 for inference.

**Solution:**
```python
# Test epoch 15
checkpoint = torch.load('models/checkpoint_ultra_epoch_15.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# Run inference

# Test epoch 20
checkpoint = torch.load('models/checkpoint_ultra_epoch_20.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# Run inference
```

---

## 💾 **Checkpoint Contents:**

Each checkpoint file contains:

```python
{
    'epoch': 12,                        # Current epoch number
    'model_state_dict': {...},          # Model weights
    'optimizer_state_dict': {...},      # Optimizer state
    'scheduler_state_dict': {...},      # Learning rate scheduler
    'scaler_state_dict': {...},         # Mixed precision scaler
    'val_smape': 42.15,                 # Validation SMAPE
    'train_loss': 0.3421,               # Training loss
    'best_val_smape': 41.82             # Best SMAPE so far
}
```

---

## 🧹 **Managing Disk Space:**

### **Problem:** Many epoch checkpoints take up space

**Option 1: Keep Only Recent (Recommended for Kaggle)**

Add to your training script:

```python
# After saving checkpoint
if epoch > 5:  # Keep only last 5 epoch checkpoints
    old_checkpoint = f'models/checkpoint_ultra_epoch_{epoch-5}.pth'
    if os.path.exists(old_checkpoint):
        os.remove(old_checkpoint)
        print(f"🗑️ Removed old checkpoint: epoch {epoch-5}")
```

**Option 2: Keep Every 5th Epoch**

```python
# Only save epoch-specific checkpoint every 5 epochs
if (epoch + 1) % 5 == 0:
    torch.save(checkpoint, f'models/checkpoint_ultra_epoch_{epoch+1}.pth')
```

**Option 3: Keep Only Latest + Best**

```python
# Don't save epoch-specific checkpoints at all
# Only keep:
# - checkpoint_ultra_latest.pth (overwrites)
# - best_model_ultra.pth (when improves)

# Comment out this line:
# torch.save(checkpoint, f'models/checkpoint_ultra_epoch_{epoch+1}.pth')
```

---

## 📊 **Monitoring Training Progress:**

### **Check What's Been Saved:**

```python
# List all checkpoints
!ls -lh models/checkpoint_ultra_*.pth

# Check latest checkpoint info
import torch
checkpoint = torch.load('models/checkpoint_ultra_latest.pth')
print(f"Latest epoch: {checkpoint['epoch'] + 1}")
print(f"Val SMAPE: {checkpoint['val_smape']:.2f}%")
print(f"Best SMAPE: {checkpoint['best_val_smape']:.2f}%")
```

---

## 🔄 **Example: Complete Resume Workflow**

### **1. Training Started:**
```python
!python train_blazing.py
```

**Output:**
```
Epoch 1/10 - Train Loss: 0.4321 - Val SMAPE: 45.12%
💾 Checkpoint saved: epoch 1
⭐ Best model saved with Val SMAPE: 45.12%

Epoch 2/10 - Train Loss: 0.3812 - Val SMAPE: 43.54%
💾 Checkpoint saved: epoch 2
⭐ Best model saved with Val SMAPE: 43.54%

Epoch 3/10 - Train Loss: 0.3621 - Val SMAPE: 42.38%
💾 Checkpoint saved: epoch 3
⭐ Best model saved with Val SMAPE: 42.38%

[CRASH!] Kaggle session expired...
```

---

### **2. Resume Training:**

Edit `train_blazing.py` at the bottom:

```python
# Find this line:
model = train_blazing(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=3e-4,
    grad_accum=8
)

# Change to:
model = train_blazing(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=3e-4,
    grad_accum=8,
    resume_from='models/checkpoint_blazing_latest.pth'  # Add this!
)
```

**Run again:**
```python
!python train_blazing.py
```

**Output:**
```
📂 Resuming from checkpoint: models/checkpoint_blazing_latest.pth
✓ Resumed from epoch 4, best SMAPE: 42.38%

Epoch 4/10 - Train Loss: 0.3452 - Val SMAPE: 41.82%
💾 Checkpoint saved: epoch 4
⭐ Best model saved with Val SMAPE: 41.82%

[Continues normally...]
```

---

## ✅ **Best Practices:**

### **1. Always Keep Best Model**
```
✅ DO: Download best_model_{model}.pth
❌ DON'T: Delete it accidentally
```

### **2. Download Checkpoints Periodically**
```
✅ DO: Download after every 5-10 epochs
❌ DON'T: Wait until training finishes
```

### **3. Test Resume Before Long Training**
```
✅ DO: Test resume with 2-3 epochs first
❌ DON'T: Assume it works without testing
```

### **4. Monitor Disk Space**
```
✅ DO: Clean old epoch checkpoints
❌ DON'T: Let them accumulate (100+ files)
```

### **5. Save Checkpoint Info**
```
✅ DO: Note which epoch had best SMAPE
❌ DON'T: Forget which file to use for inference
```

---

## 🎯 **Quick Reference:**

| Task | Command |
|------|---------|
| Resume from latest | `resume_from='models/checkpoint_{model}_latest.pth'` |
| Resume from epoch 10 | `resume_from='models/checkpoint_{model}_epoch_10.pth'` |
| Resume from best | `resume_from='models/best_model_{model}.pth'` |
| List checkpoints | `!ls -lh models/checkpoint_*.pth` |
| Check checkpoint | `torch.load('models/checkpoint_{model}_latest.pth')` |
| Download best | `FileLink('models/best_model_{model}.pth')` |
| Remove old | `!rm models/checkpoint_{model}_epoch_1.pth` |

---

## 🚨 **Troubleshooting:**

### **Error: "No such file or directory: models/checkpoint_ultra_latest.pth"**

**Cause:** Training hasn't completed even one epoch yet.

**Solution:** Wait for epoch 1 to complete, or remove `resume_from` parameter.

---

### **Error: "KeyError: 'scheduler_state_dict'"**

**Cause:** Old checkpoint format (before checkpoint system update).

**Solution:** Start fresh without `resume_from`, or use newer checkpoint.

---

### **Error: "CUDA out of memory" when resuming**

**Cause:** Checkpoint was saved with different GPU memory config.

**Solution:** Use same batch size and gradient accumulation as original training.

---

## 📈 **Summary:**

✅ **Checkpoint system active** in all 3 models (ULTRA, FAST, BLAZING)  
✅ **Saves every epoch** automatically  
✅ **Resume training** from any checkpoint  
✅ **Download best model** at any time  
✅ **Protects against crashes** and interruptions  
✅ **Add more epochs** later if needed  

**Your training is now crash-proof!** 🛡️

---

**Last Updated:** October 13, 2025, 8:45 AM IST  
**Applies to:** train_improved.py, train_fast.py, train_blazing.py  
**Status:** ✅ **PRODUCTION READY**

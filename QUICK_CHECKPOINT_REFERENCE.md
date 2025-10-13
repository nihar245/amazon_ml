# 💾 Quick Checkpoint Reference

## 🎯 **What You Get:**

✅ **Automatic saving** every epoch  
✅ **Resume training** if it crashes  
✅ **Download best model** anytime  
✅ **Add more epochs** later  

---

## 🚀 **How to Resume Training:**

### **If training crashes, edit your training file:**

**For ORIGINAL model (`train.py`):**
```python
# Find this line near the end:
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=1e-3
)

# Add resume_from parameter:
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=1e-3,
    resume_from='models/checkpoint_original_latest.pth'  # ← Add this!
)
```

**For ULTRA model (`train_improved.py`):**
```python
# Find this line near the end:
model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=1e-4,
    gradient_accumulation_steps=4
)

# Add resume_from parameter:
model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=1e-4,
    gradient_accumulation_steps=4,
    resume_from='models/checkpoint_ultra_latest.pth'  # ← Add this!
)
```

**For FAST model (`train_fast.py`):**
```python
model = train_fast_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,  # You changed this to 25
    lr=2e-4,
    gradient_accumulation_steps=4,
    resume_from='models/checkpoint_fast_latest.pth'  # ← Add this!
)
```

**For BLAZING model (`train_blazing.py`):**
```python
model = train_blazing(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=3e-4,
    grad_accum=8,
    resume_from='models/checkpoint_blazing_latest.pth'  # ← Add this!
)
```

---

## 📥 **How to Download Best Model from Kaggle:**

**Add this cell at the end of your Kaggle notebook:**

```python
# Show best model info
import torch
checkpoint = torch.load('amazon_ai_chall_1/models/best_model_ultra.pth')
print(f"✓ Best model - Epoch: {checkpoint['epoch']+1}")
print(f"✓ Validation SMAPE: {checkpoint['val_smape']:.2f}%")

# Create download link
from IPython.display import FileLink
FileLink('amazon_ai_chall_1/models/best_model_ultra.pth')
```

**Click the link to download!**

---

## 📊 **Files Saved (Per Model):**

| File | Purpose |
|------|---------|
| `checkpoint_{model}_latest.pth` | Most recent epoch (for resuming) |
| `checkpoint_{model}_epoch_N.pth` | Each epoch (optional backup) |
| `best_model_{model}.pth` | **Best model** ⭐ (use for inference) |

**Replace `{model}` with:** `original`, `ultra`, `fast`, or `blazing`

**Example:**
- Original model: `checkpoint_original_latest.pth`
- ULTRA model: `checkpoint_ultra_latest.pth`
- FAST model: `checkpoint_fast_latest.pth`
- BLAZING model: `checkpoint_blazing_latest.pth`

---

## 🔄 **Resume from Specific Epoch:**

```python
# Resume from epoch 10 instead of latest
resume_from='models/checkpoint_ultra_epoch_10.pth'
```

---

## ➕ **Add More Epochs Later:**

```python
# Original: Trained 20 epochs
# Now: Want 5 more (total 25)

model = train_model_improved(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,  # ← Increase from 20 to 25
    lr=1e-4,
    gradient_accumulation_steps=4,
    resume_from='models/best_model_ultra.pth'  # ← Start from best
)
```

---

## 🔍 **Check Training Progress:**

```python
# List all checkpoints
!ls -lh models/checkpoint_*.pth

# Check latest checkpoint
import torch
checkpoint = torch.load('models/checkpoint_ultra_latest.pth')
print(f"Current epoch: {checkpoint['epoch'] + 1}")
print(f"Val SMAPE: {checkpoint['val_smape']:.2f}%")
print(f"Best SMAPE so far: {checkpoint['best_val_smape']:.2f}%")
```

---

## 🧹 **Clean Old Checkpoints (Optional):**

```python
# Keep only latest + best (remove epoch-specific)
!rm models/checkpoint_ultra_epoch_*.pth

# Or keep only last 5 epochs
!ls models/checkpoint_ultra_epoch_*.pth | head -n -5 | xargs rm
```

---

## ⚡ **Quick Examples:**

### **Example 1: Training Crashed at Epoch 8**

```python
# Just add resume_from and run again!
resume_from='models/checkpoint_ultra_latest.pth'
```

**Output:**
```
📂 Resuming from checkpoint: models/checkpoint_ultra_latest.pth
✓ Resumed from epoch 9, best SMAPE: 43.12%

Epoch 9/20 - Train Loss: 0.3521 - Val SMAPE: 42.85%
💾 Checkpoint saved: epoch 9
⭐ Best model saved with Val SMAPE: 42.85%
```

### **Example 2: Want to Train 10 More Epochs**

```python
# Change epochs from 20 to 30, add resume_from
epochs=30
resume_from='models/best_model_ultra.pth'
```

### **Example 3: Try Different Learning Rate**

```python
# Resume from epoch 15, change LR
epochs=25
lr=5e-5  # Lower LR for fine-tuning
resume_from='models/checkpoint_ultra_epoch_15.pth'
```

---

## 🎯 **What You'll See During Training:**

```
Epoch 1/20 - Train Loss: 0.4821 - Val SMAPE: 48.12%
💾 Checkpoint saved: epoch 1
⭐ Best model saved with Val SMAPE: 48.12%

Epoch 2/20 - Train Loss: 0.4156 - Val SMAPE: 45.31%
💾 Checkpoint saved: epoch 2
⭐ Best model saved with Val SMAPE: 45.31%

Epoch 3/20 - Train Loss: 0.3821 - Val SMAPE: 46.02%
💾 Checkpoint saved: epoch 3
(No ⭐ because SMAPE didn't improve)
```

**Key:**
- 💾 = Checkpoint saved (every epoch)
- ⭐ = Best model updated (only when SMAPE improves)

---

## ✅ **Checklist Before Long Training:**

- [ ] Checkpoint system is in the code (it is! ✅)
- [ ] Tested resume with 2-3 epochs
- [ ] Know how to download best_model_{model}.pth
- [ ] Have backup plan if Kaggle crashes
- [ ] Ready to resume if needed

---

## 🚨 **Emergency: How to Recover After Crash:**

1. **Check what's saved:**
   ```python
   !ls -lh models/checkpoint_*.pth
   ```

2. **See latest checkpoint:**
   ```python
   checkpoint = torch.load('models/checkpoint_ultra_latest.pth')
   print(f"Can resume from epoch {checkpoint['epoch'] + 1}")
   ```

3. **Edit training script:**
   ```python
   # Add resume_from parameter
   resume_from='models/checkpoint_ultra_latest.pth'
   ```

4. **Run again:**
   ```python
   !python train_improved.py
   ```

5. **Training continues from where it stopped!** ✅

---

## 💡 **Pro Tips:**

1. **Download periodically:** Don't wait for all 20 epochs. Download best model after every 5-10 epochs.

2. **Test resume early:** After epoch 2-3, test that resume works before committing to 20 epochs.

3. **Monitor disk space:** Epoch-specific checkpoints accumulate. Clean old ones if needed.

4. **Keep best model:** `best_model_{model}.pth` is the most important file. Never delete it!

5. **Use latest for resume:** `checkpoint_{model}_latest.pth` is best for resuming after crash.

---

## 📖 **Full Documentation:**

See `CHECKPOINT_GUIDE.md` for complete details, advanced usage, and troubleshooting.

---

**Status:** ✅ **Active in all 4 models** (ORIGINAL, ULTRA, FAST, BLAZING)  
**Benefit:** 🛡️ **Crash protection + Resume capability**  
**Usage:** 🎯 **Just add `resume_from` parameter!**

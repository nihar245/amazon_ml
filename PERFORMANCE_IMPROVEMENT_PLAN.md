# 🎯 MODEL PERFORMANCE IMPROVEMENT PLAN
## From 47.5% → Below 42% SMAPE

---

## 📊 Current Status

```
Model: train_improved.py with MiniLM-L6-v2
Train SMAPE: 44.71%
Val SMAPE: 47.99%
Test SMAPE: 47.55%
Gap: 3.28% (low overfitting ✅)
```

**Analysis:** Model is not overfitting (small gap), so we can increase capacity!

---

## 🚀 IMPLEMENTATION PLAN (5 Key Changes)

### Priority 1: Switch to DeBERTa-v3-base ⭐⭐⭐⭐⭐
**Expected Impact:** -5 to -7% SMAPE reduction

**Files to modify:**
1. `train_improved.py` line 475
2. `train_improved.py` line 557
3. `sample_code_improved.py` line 324 & 341

**Changes:**
```python
# Replace all instances of:
'sentence-transformers/all-MiniLM-L6-v2'
# With:
'microsoft/deberta-v3-base'

# Also update hidden_dim from 512 to 768
```

---

### Priority 2: Fine-tune ALL layers ⭐⭐⭐⭐⭐
**Expected Impact:** -2 to -3% SMAPE reduction

**File:** `train_improved.py` lines 322-329

**Action:** Remove or comment out the layer freezing code

**Reason:** Currently only last 2 layers train. Full fine-tuning adapts model better to price prediction task.

---

### Priority 3: Increase sequence length ⭐⭐⭐⭐
**Expected Impact:** -1 to -2% SMAPE reduction

**Files:** `train_improved.py` lines 498, 503

**Changes:**
```python
# Change from:
max_length=256
# To:
max_length=384
```

**Reason:** Longer sequences capture more product information

---

### Priority 4: More training epochs ⭐⭐⭐
**Expected Impact:** -0.5 to -1% SMAPE reduction

**File:** `train_improved.py` line 583

**Changes:**
```python
# Change from:
epochs=25
# To:
epochs=35
```

---

### Priority 5: Reduce batch size ⭐⭐⭐
**Expected Impact:** Prevents OOM, enables larger model

**Files:** `train_improved.py` lines 507-508

**Changes:**
```python
# Change from:
batch_size=16
# To:
batch_size=8
```

**Reason:** DeBERTa needs more GPU memory

---

## 📈 Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Text Encoder | MiniLM (22M) | DeBERTa (184M) |
| Sequence Length | 256 tokens | 384 tokens |
| Trainable Layers | Last 2 | ALL layers |
| Epochs | 25 | 35 |
| Batch Size | 16 | 8 |
| **Train SMAPE** | 44.71% | 33-38% |
| **Val SMAPE** | 47.99% | 36-41% |
| **Test SMAPE** | 47.55% | 37-42% ✅ |
| **Training Time** | 50-70 min | 80-100 min |

---

## ⚡ QUICK IMPLEMENTATION GUIDE

### Step 1: Apply Changes to train_improved.py

```bash
# Make a backup
cp train_improved.py train_improved_backup.py

# Edit train_improved.py with changes:
# 1. Line 475: Change tokenizer to deberta-v3-base
# 2. Line 557: Change model to deberta-v3-base, hidden_dim=768
# 3. Lines 322-329: Comment out freezing code
# 4. Lines 498, 503: Change max_length=384
# 5. Lines 507-508: Change batch_size=8
# 6. Line 583: Change epochs=35
```

### Step 2: Apply Changes to sample_code_improved.py

```bash
# Edit sample_code_improved.py:
# 1. Line 324: Change tokenizer to deberta-v3-base
# 2. Line 341-344: Change model to deberta-v3-base, hidden_dim=768
```

### Step 3: Train on Kaggle

```python
# On Kaggle:
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1

!mkdir -p student_resource/dataset models
!cp /kaggle/input/amazon-ml-challenge-dataset/*.csv student_resource/dataset/

!pip install transformers>=4.30.0 -q

# Train (80-100 minutes)
!python train_improved.py
```

### Step 4: Generate Predictions

```python
# After training:
!python sample_code_improved.py
```

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
```
Solution 1: Reduce batch_size to 6 or 4
Solution 2: Reduce max_length to 320 or 256
Solution 3: Use gradient_checkpointing (advanced)
```

### Issue: Training too slow
```
Ensure GPU P100 is enabled (not T4)
Expected: ~2.5-3 min per epoch
Total: 80-100 minutes for 35 epochs
```

### Issue: DeBERTa model not loading
```
!pip install transformers>=4.30.0 --upgrade
Restart Kaggle session
```

---

## 📊 Validation Strategy

### After Training, Check:
1. **Val SMAPE < 42%** ✅ Target achieved
2. **Train-Val gap < 5%** ✅ Not overfitting
3. **Training curve smooth** ✅ Good convergence
4. **Model file saved** ✅ best_model_ultra.pth exists

### If Val SMAPE still > 42%:
- Try Phase 2 improvements (see IMPROVEMENT_STRATEGY.md)
- Consider ensemble (3-5 models)
- Add more features
- Use focal loss

---

## 💾 File Modifications Required

### Primary Files:
1. **train_improved.py** - 7 changes
2. **sample_code_improved.py** - 2 changes

### Documentation Updates:
1. DEPLOYMENT_GUIDE.md - Update model specs
2. DATA_FLOW.md - Update architecture
3. CHANGES_IMPLEMENTED.txt - Add ultra improvements

---

## ✅ Implementation Checklist

```
[ ] Backup train_improved.py
[ ] Change 1: DeBERTa tokenizer (line 475)
[ ] Change 2: DeBERTa model (line 557)
[ ] Change 3: hidden_dim=768 (line 557)
[ ] Change 4: Remove freezing (lines 322-329)
[ ] Change 5: max_length=384 (lines 498, 503)
[ ] Change 6: batch_size=8 (lines 507-508)
[ ] Change 7: epochs=35 (line 583)
[ ] Update sample_code_improved.py (2 places)
[ ] Test locally (optional)
[ ] Push to GitHub
[ ] Run on Kaggle
[ ] Verify SMAPE < 42%
```

---

## 🎯 Success Metrics

### Must Achieve:
- ✅ Val SMAPE < 42%
- ✅ Test SMAPE < 42%
- ✅ No OOM errors
- ✅ Training completes successfully

### Nice to Have:
- ⭐ Val SMAPE < 40%
- ⭐ Test SMAPE < 40%
- ⭐ Train-Val gap < 3%

---

## 📞 Next Steps After Implementation

1. **If SMAPE 37-42%:** ✅ SUCCESS! Generate final predictions
2. **If SMAPE 42-45%:** Try ensemble (3 models), averaging predictions
3. **If SMAPE >45%:** Check implementation, verify all changes applied

---

## 🚀 Alternative: Pre-made train_ultra.py

I can create a complete `train_ultra.py` with all changes pre-applied.

**Advantages:**
- No manual editing
- All changes guaranteed
- Ready to use

**To create:** Just ask and I'll generate the complete file!

---

## 📚 Additional Resources

- `ULTRA_UPGRADE_GUIDE.md` - Detailed change guide
- `IMPROVEMENT_STRATEGY.md` - Phase 2 & 3 improvements
- `BUGFIXES_APPLIED.txt` - Previous fixes applied
- `VERIFICATION_CHECKLIST.txt` - Pre-flight checks

---

**Status:** Ready to implement  
**Expected Time:** 2-3 hours total (editing + training)  
**Expected Result:** SMAPE 37-42% ✅ Below target!

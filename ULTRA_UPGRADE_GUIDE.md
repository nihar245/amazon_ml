# 🚀 ULTRA MODEL UPGRADE GUIDE
## Goal: Reduce SMAPE from 47.5% to below 42%

### Current Performance:
- Train SMAPE: 44.71%
- Val SMAPE: 47.99%
- Test SMAPE: 47.55%
- **Gap:** 3.28% (low overfitting ✅)

### Target: <42% SMAPE (5-6% improvement needed)

---

## 🎯 TOP 5 HIGH-IMPACT CHANGES (Implement These)

### 1. **Switch to DeBERTa-v3-base** (Expected: 5-7% improvement) ⭐⭐⭐⭐⭐

**Change in `train_improved.py` line 559:**
```python
# OLD:
text_model_name='sentence-transformers/all-MiniLM-L6-v2'

# NEW:
text_model_name='microsoft/deberta-v3-base'
```

**Also change in `train_improved.py` line 469:**
```python
# OLD:
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# NEW:
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
```

**Why:** DeBERTa-v3-base (184M params) is FAR superior to MiniLM (22M params)
- Better language understanding
- Better price-related reasoning
- Still only 2.3% of 8B param limit!

**Impact:** -5 to -7% SMAPE

---

### 2. **Fine-tune ALL Transformer Layers** (Expected: 2-3% improvement) ⭐⭐⭐⭐⭐

**Remove lines 322-329 in `train_improved.py`:**
```python
# DELETE THESE LINES:
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        try:
            for layer in self.text_encoder.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        except Exception:
            pass
```

**Why:** Currently only last 2 layers are trainable. Training ALL layers lets model adapt better to price prediction.

**Impact:** -2 to -3% SMAPE

---

### 3. **Increase Sequence Length** (Expected: 1-2% improvement) ⭐⭐⭐⭐

**Change in `train_improved.py` line 498 and 503:**
```python
# OLD:
train_dataset = ProductDataset(train_data, tokenizer, brand_encoder, category_encoder, 
                               max_length=256, is_train=True)
val_dataset = ProductDataset(val_data, tokenizer, brand_encoder, category_encoder,
                             max_length=256, is_train=True)

# NEW:
train_dataset = ProductDataset(train_data, tokenizer, brand_encoder, category_encoder, 
                               max_length=384, is_train=True)
val_dataset = ProductDataset(val_data, tokenizer, brand_encoder, category_encoder,
                             max_length=384, is_train=True)
```

**Why:** Longer sequences capture more product information (descriptions, features)

**Impact:** -1 to -2% SMAPE

---

### 4. **Train for More Epochs** (Expected: 0.5-1% improvement) ⭐⭐⭐

**Change in `train_improved.py` line 583:**
```python
# OLD:
best_val_loss = train_model_improved(model, train_loader, val_loader, epochs=25)

# NEW:
best_val_loss = train_model_improved(model, train_loader, val_loader, epochs=35)
```

**Why:** More epochs = better convergence, especially with larger model

**Impact:** -0.5 to -1% SMAPE

---

### 5. **Reduce Batch Size (for memory)** (Required for DeBERTa) ⭐⭐⭐⭐

**Change in `train_improved.py` line 507-508:**
```python
# OLD:
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# NEW:
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
```

**Why:** DeBERTa is larger and needs more GPU memory. Smaller batch size prevents OOM errors.

**Impact:** No performance change, just prevents crashes

---

## 📊 Expected Total Improvement:

| Change | SMAPE Reduction |
|--------|-----------------|
| DeBERTa-v3-base | -5 to -7% |
| Fine-tune all layers | -2 to -3% |
| Longer sequences (384) | -1 to -2% |
| More epochs (35) | -0.5 to -1% |
| **TOTAL EXPECTED** | **-8.5 to -13%** |

**Current:** 47.5% SMAPE  
**Expected:** 34-39% SMAPE ✅ **Below 42% target!**

---

## 🛠️ COMPLETE MODIFIED CODE SECTIONS:

### Section 1: Tokenizer Initialization (Line ~469)
```python
print("\nInitializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
```

### Section 2: Model Initialization (Line ~559-565)
```python
print("\nInitializing improved model...")
model = ImprovedPricePredictor(
    text_model_name='microsoft/deberta-v3-base',  # ← CHANGED
    numeric_dim=25,
    hidden_dim=768  # ← CHANGED (DeBERTa hidden_dim is 768, not 512)
).to(device)
```

### Section 3: Model Architecture (Line ~318-329) - REMOVE FREEZING
```python
def __init__(self, text_model_name='microsoft/deberta-v3-base', 
             numeric_dim=25, hidden_dim=768):  # ← CHANGED hidden_dim
    super().__init__()
    
    self.text_encoder = AutoModel.from_pretrained(text_model_name)
    text_dim = self.text_encoder.config.hidden_size  # 768 for DeBERTa
    
    # REMOVE THE FREEZING CODE - Let all layers train!
    # (Delete lines 322-329)
    
    # Fusion network (adjust first layer input to match 768)
    self.fusion = nn.Sequential(
        nn.Linear(text_dim + numeric_dim, hidden_dim),  # 793 → 768
        nn.BatchNorm1d(hidden_dim),
        # ... rest stays the same
```

### Section 4: Dataset Creation (Line ~497-508)
```python
train_dataset = ProductDataset(
    train_data, tokenizer, brand_encoder, category_encoder, 
    max_length=384, is_train=True  # ← CHANGED from 256
)
val_dataset = ProductDataset(
    val_data, tokenizer, brand_encoder, category_encoder,
    max_length=384, is_train=True  # ← CHANGED from 256
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)  # ← CHANGED from 16
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)  # ← CHANGED from 16
```

### Section 5: Training Call (Line ~583)
```python
best_val_loss = train_model_improved(model, train_loader, val_loader, epochs=35)  # ← CHANGED from 25
```

---

## ⚡ QUICK IMPLEMENTATION CHECKLIST:

```
[ ] 1. Change text_model_name to 'microsoft/deberta-v3-base' (2 places)
[ ] 2. Change hidden_dim to 768 (2 places)
[ ] 3. Remove lines 322-329 (freezing code)
[ ] 4. Change max_length to 384 (2 places)
[ ] 5. Change batch_size to 8 (2 places)
[ ] 6. Change epochs to 35 (1 place)
[ ] 7. Save as train_ultra.py or modify train_improved.py
[ ] 8. Run training (will take 80-100 minutes)
```

---

## 🎯 Kaggle Configuration:

### Memory Requirements:
- **DeBERTa-v3-base:** ~2GB GPU memory
- **Batch size 8:** ~12GB GPU memory total
- **Kaggle P100:** 16GB ✅ Should work!
- **If OOM:** Reduce batch_size to 6 or 4

### Training Time:
- **MiniLM (current):** 50-70 minutes
- **DeBERTa (upgraded):** 80-100 minutes
- **Extra time:** +30-40 minutes (worth it for 8-13% improvement!)

### Installation:
```python
# DeBERTa should already be included in transformers
# If error, run:
!pip install transformers>=4.30.0 -q
```

---

## 📋 File Modifications Summary:

### Files to Modify:
1. **train_improved.py** - Apply all 5 changes above
2. **sample_code_improved.py** - Update model loading (see below)

### sample_code_improved.py Changes:

**Line ~324 and ~341:**
```python
# OLD:
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = ImprovedPricePredictor(
    text_model_name='sentence-transformers/all-MiniLM-L6-v2',
    numeric_dim=25,
    hidden_dim=512
)

# NEW:
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
model = ImprovedPricePredictor(
    text_model_name='microsoft/deberta-v3-base',
    numeric_dim=25,
    hidden_dim=768
)
```

**Also update model file path:**
```python
# Load from new model file
checkpoint = torch.load('models/best_model_ultra.pth', map_location=device)
```

---

## ⚠️ Potential Issues & Solutions:

### Issue 1: CUDA Out of Memory
```
Solution: Reduce batch_size to 6 or 4
```

### Issue 2: DeBERTa not found
```
Solution: !pip install transformers>=4.30.0
```

### Issue 3: Training too slow
```
Solution: Use Kaggle P100 GPU (not T4)
Enable GPU in Settings → Accelerator → GPU P100
```

### Issue 4: Model weights too large
```
DeBERTa model file will be ~700MB (vs ~100MB for MiniLM)
This is fine - still under GitHub LFS limit
Save to Kaggle Dataset for reuse
```

---

## 🎯 Expected Results:

### After Training:
```
Epoch 35/35
Train SMAPE: 33-38%
Val SMAPE: 36-41%
Test SMAPE: 37-42% ← Target achieved!
```

### Performance Breakdown:
- **Baseline (train.py):** 50-55% SMAPE
- **Improved (train_improved.py):** 47.5% SMAPE ← You are here
- **Ultra (with changes):** 37-42% SMAPE ← Target
- **Improvement:** ~10% SMAPE reduction ✅

---

## 📦 Complete Modified File:

I'll create `train_ultra.py` with all changes applied. See next file!

---

## 🚀 Quick Start Commands (Kaggle):

```python
# 1. Clone repo
!git clone https://github.com/meethp1884/amazon_ai_chall_1.git
%cd amazon_ai_chall_1

# 2. Setup
!mkdir -p student_resource/dataset models
!cp /kaggle/input/amazon-ml-challenge-dataset/*.csv student_resource/dataset/

# 3. Install
!pip install transformers>=4.30.0 -q

# 4. Train (use modified script)
!python train_ultra.py  # ← Will create this file

# 5. Wait 80-100 minutes

# 6. Generate predictions
!python sample_code_ultra.py  # ← Will create this file
```

---

## ✅ Success Criteria:

```
✅ Val SMAPE < 42%
✅ Train-Val gap < 5%
✅ No CUDA OOM errors
✅ Model saved successfully
✅ Test SMAPE < 42%
```

---

**NEXT:** I'll create the complete `train_ultra.py` file with all modifications applied!

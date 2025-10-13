# 🎯 Model Improvement Strategy - Reduce SMAPE from 55% to 30-40%

**Current Performance:** Validation SMAPE = 55%  
**Target Performance:** SMAPE = 30-40%  
**Required Improvement:** ~20-25% SMAPE reduction  

---

## 📊 Current Issues Analysis

### Why is SMAPE at 55%?

1. **Limited Feature Engineering:**
   - Only basic numeric features (8 features)
   - No brand/category extraction
   - No price range indicators
   - Missing product type classification

2. **Shallow Model:**
   - Only 3 hidden layers
   - Limited model capacity
   - No attention mechanisms
   - Minimal text encoder fine-tuning (only 2 layers)

3. **Basic Training Strategy:**
   - Standard learning rate
   - No learning rate warmup
   - Limited epochs (15)
   - Basic regularization only

4. **No Data Augmentation:**
   - No text augmentation
   - No feature perturbation
   - Single model (no ensemble)

---

## 🚀 Implementation Plan - Priority Order

### 🔥 Phase 1: Quick Wins (Expected: 10-15% SMAPE reduction)

**1. Advanced Feature Engineering** ⭐ HIGHEST IMPACT
   - Extract brand names (NER or regex)
   - Extract product categories
   - Add price range buckets
   - Text quality features (has brand, has details, etc.)
   - N-gram features (TF-IDF top terms)
   
   **Expected Impact:** 5-8% SMAPE reduction

**2. Better Text Processing**
   - Increase max_length from 128 → 256 tokens
   - Better text cleaning (remove noise but keep structure)
   - Separate title and description processing
   
   **Expected Impact:** 2-3% SMAPE reduction

**3. Improved Model Architecture**
   - Deeper network (4-5 layers instead of 3)
   - Increase hidden dimensions (256 → 512)
   - Add Batch Normalization
   - Better dropout schedule
   
   **Expected Impact:** 3-5% SMAPE reduction

**4. Training Improvements**
   - Longer training (20-25 epochs with early stopping)
   - Learning rate warmup
   - Cosine annealing scheduler
   - Gradient accumulation for larger effective batch size
   
   **Expected Impact:** 2-4% SMAPE reduction

---

### 🔥 Phase 2: Advanced Techniques (Expected: 5-10% SMAPE reduction)

**5. Full Transformer Fine-tuning**
   - Fine-tune all MiniLM layers (not just last 2)
   - Use discriminative learning rates
   - Layer-wise learning rate decay
   
   **Expected Impact:** 3-5% SMAPE reduction

**6. Multi-head Attention**
   - Add attention over bullet points
   - Feature-wise attention
   - Self-attention on numeric features
   
   **Expected Impact:** 2-4% SMAPE reduction

**7. Better Loss Function**
   - Weighted SMAPE (weight by price ranges)
   - Focal loss variant for hard examples
   - Combined SMAPE + MAE loss
   
   **Expected Impact:** 1-3% SMAPE reduction

---

### 🔥 Phase 3: Ensemble & Advanced (Expected: 5-8% SMAPE reduction)

**8. Model Ensemble**
   - Train multiple models with different seeds
   - Combine with XGBoost/LightGBM
   - Different architectures (BERT, RoBERTa)
   
   **Expected Impact:** 3-5% SMAPE reduction

**9. Image Features (CLIP)**
   - Add visual product information
   - CLIP ViT-B/32 embeddings
   - Multimodal fusion
   
   **Expected Impact:** 2-5% SMAPE reduction (if images are informative)

---

## 📝 Detailed Implementation Guide

### 1. Advanced Feature Engineering (HIGHEST PRIORITY)

#### A. Brand Name Extraction

```python
def extract_brand(text):
    """Extract brand name from product text"""
    if pd.isna(text):
        return "unknown"
    
    text = str(text).lower()
    
    # Common brands (add more as needed)
    brands = ['amazon', 'kraft', 'nestle', 'pepsi', 'coca-cola', 'heinz',
              'campbell', 'general mills', 'kellogg', 'unilever', 'procter',
              'johnson', 'colgate', 'palmolive', 'gillette', 'dove', 'axe']
    
    for brand in brands:
        if brand in text:
            return brand
    
    # Extract first capitalized word (often brand)
    words = text.split()
    for word in words:
        if word[0].isupper() and len(word) > 2:
            return word.lower()
    
    return "unknown"

# Brand encoding (one-hot or label encode)
brand = extract_brand(catalog_content)
brand_encoded = label_encoder.transform([brand])[0]
```

#### B. Product Category Classification

```python
def classify_category(text):
    """Classify product into categories"""
    if pd.isna(text):
        return "other"
    
    text = str(text).lower()
    
    # Category keywords
    categories = {
        'food': ['food', 'snack', 'sauce', 'pasta', 'rice', 'cereal', 'bread'],
        'beverage': ['coffee', 'tea', 'juice', 'drink', 'water', 'soda'],
        'health': ['vitamin', 'supplement', 'medicine', 'health', 'protein'],
        'beauty': ['shampoo', 'soap', 'lotion', 'cream', 'cosmetic'],
        'household': ['cleaner', 'detergent', 'paper', 'towel', 'bag'],
        'baby': ['baby', 'diaper', 'formula', 'wipes'],
        'pet': ['pet', 'dog', 'cat', 'animal']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category
    
    return "other"

category = classify_category(catalog_content)
category_encoded = label_encoder.transform([category])[0]
```

#### C. Price Range Buckets

```python
# During training, create price buckets
def get_price_bucket(price):
    if price < 5:
        return 0  # very_cheap
    elif price < 10:
        return 1  # cheap
    elif price < 20:
        return 2  # moderate
    elif price < 50:
        return 3  # expensive
    else:
        return 4  # very_expensive

# Use as auxiliary task during training
```

#### D. Text Quality Features

```python
def extract_quality_features(text):
    """Extract text quality indicators"""
    if pd.isna(text):
        return [0, 0, 0, 0, 0]
    
    text = str(text)
    
    features = [
        1 if any(char.isupper() for char in text) else 0,  # has_capitals (brand)
        len(text.split()),  # word_count
        text.count(','),  # has_list
        1 if any(num.isdigit() for num in text) else 0,  # has_numbers
        len(set(text.split())) / max(len(text.split()), 1)  # vocabulary_richness
    ]
    
    return features
```

#### E. Enhanced Numeric Features (Expand from 8 to 20+ features)

```python
def extract_enhanced_features(catalog_content):
    """Extract comprehensive feature set"""
    features = {
        # Existing features (8)
        'value': 0.0,
        'pack_size': 1,
        'total_quantity': 0.0,
        'is_pack': 0,
        'text_length': 0,
        'bullet_points': 0,
        'has_value': 0,
        'unit_type': 0,
        
        # NEW: Brand & Category (2)
        'brand_encoded': 0,
        'category_encoded': 0,
        
        # NEW: Text Quality (5)
        'has_capitals': 0,
        'word_count': 0,
        'comma_count': 0,
        'has_numbers': 0,
        'vocab_richness': 0.0,
        
        # NEW: Advanced Quantity (5)
        'value_per_unit': 0.0,
        'log_total_quantity': 0.0,
        'is_large_pack': 0,  # pack_size > 4
        'has_fraction': 0,  # like "0.5 oz"
        'quantity_category': 0,  # 0-4 (tiny to huge)
        
        # NEW: Text Structure (3)
        'num_sentences': 0,
        'avg_word_length': 0.0,
        'has_description': 0,  # length > 100 chars
    }
    
    # ... implement extraction logic ...
    
    return features  # 23 features total
```

---

### 2. Improved Model Architecture

#### A. Deeper Network with Batch Normalization

```python
class ImprovedPricePredictor(nn.Module):
    def __init__(self, text_model_name, numeric_dim=23, hidden_dim=512):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 384
        
        # Deeper network with batch normalization
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),  # 407 → 512
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),  # 512 → 512
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(hidden_dim, hidden_dim // 2),  # 512 → 256
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 256 → 128
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim // 4, 64),  # 128 → 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)  # 64 → 1
        )
        
    def forward(self, input_ids, attention_mask, numeric_features):
        # Text encoding
        text_output = self.text_encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        
        # Feature fusion
        combined = torch.cat([text_features, numeric_features], dim=1)
        price = self.fusion(combined).squeeze(-1)
        
        return price
```

#### B. With Attention Mechanism (Alternative)

```python
class AttentionPricePredictor(nn.Module):
    def __init__(self, text_model_name, numeric_dim=23, hidden_dim=512):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        # Attention over token embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # ... rest of network ...
        )
    
    def forward(self, input_ids, attention_mask, numeric_features):
        # Get all token embeddings
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        token_embeddings = text_output.last_hidden_state  # (batch, seq_len, 384)
        
        # Apply attention
        attn_output, _ = self.attention(
            token_embeddings, 
            token_embeddings, 
            token_embeddings,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Use [CLS] token with attention
        text_features = attn_output[:, 0, :]
        
        # Fusion
        combined = torch.cat([text_features, numeric_features], dim=1)
        price = self.fusion(combined).squeeze(-1)
        
        return price
```

---

### 3. Improved Training Strategy

#### A. Learning Rate Warmup + Cosine Annealing

```python
def train_model_improved(model, train_loader, val_loader, epochs=25, lr=2e-4):
    """Improved training with better scheduling"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Warmup + Cosine Annealing
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * 2  # 2 epochs warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # ... training loop with scheduler.step() after each batch ...
```

#### B. Gradient Accumulation

```python
# For effective batch size = 64 (with actual batch size = 16)
accumulation_steps = 4

for epoch in range(epochs):
    optimizer.zero_grad()
    
    for i, batch in enumerate(train_loader):
        # Forward pass
        predictions = model(input_ids, attention_mask, numeric_features)
        loss = criterion(predictions, targets)
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
```

#### C. Mixed Precision Training (Faster)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        predictions = model(input_ids, attention_mask, numeric_features)
        loss = criterion(predictions, targets)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

### 4. Better Hyperparameters

```python
# Current (suboptimal)
batch_size = 32
learning_rate = 2e-4
hidden_dim = 256
max_length = 128
epochs = 15

# Improved
batch_size = 16  # Smaller for better gradients
accumulation_steps = 4  # Effective batch size = 64
learning_rate = 5e-5  # Lower for stability
hidden_dim = 512  # Larger capacity
max_length = 256  # Capture more context
epochs = 25  # More training
warmup_epochs = 2  # Gradual warmup
```

---

### 5. Data Augmentation Techniques

#### A. Text Augmentation

```python
import random

def augment_text(text):
    """Augment text data"""
    if random.random() > 0.5:
        return text  # 50% no augmentation
    
    words = text.split()
    
    # Random deletion (remove 10% of words)
    if random.random() > 0.5:
        keep_indices = random.sample(range(len(words)), int(len(words) * 0.9))
        words = [words[i] for i in sorted(keep_indices)]
    
    # Random swap (swap 2 adjacent words)
    if random.random() > 0.5 and len(words) > 2:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    
    return ' '.join(words)
```

#### B. Feature Perturbation

```python
def perturb_numeric_features(features, training=True):
    """Add small noise to numeric features during training"""
    if not training:
        return features
    
    noise = torch.randn_like(features) * 0.05  # 5% noise
    return features + noise
```

---

### 6. Ensemble Methods

#### A. Simple Ensemble (Average Predictions)

```python
# Train 5 models with different seeds
models = []
for seed in [42, 123, 456, 789, 2023]:
    set_seed(seed)
    model = train_model(...)
    models.append(model)

# Ensemble prediction
def ensemble_predict(models, input_ids, attention_mask, numeric_features):
    predictions = []
    for model in models:
        pred = model(input_ids, attention_mask, numeric_features)
        predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

#### B. Stacking with XGBoost

```python
# Step 1: Get predictions from neural network
nn_predictions = []
for batch in data_loader:
    with torch.no_grad():
        pred = model(...)
        nn_predictions.append(pred)

# Step 2: Train XGBoost on top
import xgboost as xgb

# Features for XGBoost: numeric features + NN predictions
X_meta = np.concatenate([numeric_features, nn_predictions], axis=1)
y_meta = actual_prices

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8
)

xgb_model.fit(X_meta, y_meta)

# Step 3: Final prediction = XGBoost(NN predictions + features)
final_prediction = xgb_model.predict(X_meta_test)
```

---

## 📊 Expected Results After Implementation

### Phase 1 (Quick Wins) - Expected SMAPE: 40-45%
- Advanced feature engineering: 23 features
- Deeper model: 5 layers
- Better training: 25 epochs with warmup
- **Time:** 2-3 hours implementation
- **Training:** 45-60 minutes

### Phase 2 (Advanced) - Expected SMAPE: 35-40%
- Full transformer fine-tuning
- Attention mechanisms
- Better loss function
- **Time:** 4-6 hours implementation
- **Training:** 60-90 minutes

### Phase 3 (Ensemble) - Expected SMAPE: 30-35%
- 5-model ensemble
- XGBoost stacking
- Image features (optional)
- **Time:** 6-8 hours implementation
- **Training:** 3-4 hours (5 models)

---

## 🎯 Recommended Immediate Actions (Priority Order)

### Action 1: Enhanced Feature Engineering ⭐⭐⭐⭐⭐
**Impact:** HIGH | **Effort:** MEDIUM | **Time:** 2 hours

Implement all 23 features (brand, category, quality indicators).

### Action 2: Deeper Model Architecture ⭐⭐⭐⭐
**Impact:** HIGH | **Effort:** LOW | **Time:** 30 mins

Add more layers and batch normalization.

### Action 3: Better Training Strategy ⭐⭐⭐⭐
**Impact:** MEDIUM-HIGH | **Effort:** MEDIUM | **Time:** 1 hour

Warmup + cosine annealing + longer training.

### Action 4: Increase Model Capacity ⭐⭐⭐
**Impact:** MEDIUM | **Effort:** LOW | **Time:** 15 mins

hidden_dim: 256 → 512, max_length: 128 → 256

### Action 5: Full Transformer Fine-tuning ⭐⭐⭐
**Impact:** MEDIUM | **Effort:** LOW | **Time:** 30 mins

Unfreeze all layers with discriminative learning rates.

---

## 🚫 What NOT to Do (Common Mistakes)

1. ❌ **Don't overfit to validation set** - Use cross-validation
2. ❌ **Don't use huge batch sizes** - Smaller batches (16-32) often better
3. ❌ **Don't skip warmup** - Causes training instability
4. ❌ **Don't train too long** - Use early stopping (patience=7-10)
5. ❌ **Don't ignore data leakage** - Ensure proper train/val split
6. ❌ **Don't use outdated models** - MiniLM is good, but try others too

---

## ✅ Success Metrics

Track these during training:

```python
metrics = {
    'train_smape': [],
    'val_smape': [],
    'train_val_gap': [],  # Should be <5%
    'learning_rate': [],
    'gradient_norm': [],  # Should be stable
}
```

**Target Metrics:**
- Train SMAPE: 28-35%
- Val SMAPE: 30-40%
- Train-Val Gap: <5%
- Training Time: <90 minutes per model

---

## 📋 Implementation Checklist

### Quick Wins (Do First)
- [ ] Extract brand names (add brand_encoded feature)
- [ ] Extract categories (add category_encoded feature)
- [ ] Add text quality features (+5 features)
- [ ] Add advanced quantity features (+5 features)
- [ ] Total: 8 → 23 features
- [ ] Increase hidden_dim: 256 → 512
- [ ] Add 2 more hidden layers (3 → 5 layers)
- [ ] Add BatchNorm after each layer
- [ ] Increase max_length: 128 → 256
- [ ] Implement learning rate warmup
- [ ] Use cosine annealing scheduler
- [ ] Train for 25 epochs (vs 15)
- [ ] Lower learning rate: 2e-4 → 5e-5

### Advanced (Do Second)
- [ ] Fine-tune all transformer layers
- [ ] Implement gradient accumulation
- [ ] Add mixed precision training
- [ ] Implement attention mechanism
- [ ] Try different loss functions

### Ensemble (Do Last)
- [ ] Train 5 models with different seeds
- [ ] Implement ensemble averaging
- [ ] Train XGBoost on top
- [ ] (Optional) Add CLIP image features

---

**Total Expected Improvement:** 20-25% SMAPE reduction (55% → 30-35%)

**Implementation Time:**
- Phase 1: 2-3 hours
- Phase 2: 4-6 hours
- Phase 3: 6-8 hours
- **Total: 12-17 hours of work**

**Training Time:**
- Single model: ~60 minutes
- Ensemble (5 models): ~300 minutes (5 hours)

---

## 💡 Pro Tips

1. **Test incrementally** - Implement one change at a time, measure impact
2. **Use test_sample.py** - Quick validation before full training
3. **Save all models** - You might want to ensemble later
4. **Monitor gradients** - Use TensorBoard or Weights & Biases
5. **Try different architectures** - DeBERTa, RoBERTa might work better
6. **Cross-validate** - Don't trust single train/val split

---

**Next Steps:** I will now implement Phase 1 improvements (highest impact) in your training script!

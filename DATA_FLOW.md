# 📊 Data Flow Documentation

## Complete Data Pipeline - From Dataset to Predictions

---

## 🗂️ File Structure and Data Movement

```
Amazon_ai_challenge/
├── student_resource/
│   ├── dataset/
│   │   ├── train.csv          [INPUT]  → Training data with prices
│   │   ├── test.csv           [INPUT]  → Test data without prices
│   │   └── test_out.csv       [OUTPUT] → Final predictions
│   └── sample_code.py         [OLD]    → Original baseline code
├── train.py                   [SCRIPT] → Training pipeline
├── sample_code.py             [SCRIPT] → Inference pipeline
├── models/
│   ├── best_model.pth         [MODEL]  → Trained model weights
│   └── feature_scaler.pkl     [MODEL]  → Feature normalization scaler
└── requirements.txt           [CONFIG] → Dependencies
```

---

## 📈 PHASE 1: TRAINING PIPELINE (train.py)

### Step 1: Data Loading
**File:** `student_resource/dataset/train.csv`  
**Destination:** `train_df` DataFrame in memory

**Data Format:**
- **sample_id**: Unique identifier (e.g., 33127)
- **catalog_content**: Product description text
- **image_link**: Product image URL
- **price**: Target variable (ground truth price)

**Number of Samples:** 75,000 training products

---

### Step 2: Feature Extraction

#### 2.1 Numeric Features Extraction
**Function:** `extract_numeric_features(catalog_content)`  
**Input:** Raw catalog_content string  
**Output:** 25 features

**BASELINE MODEL (train.py) - 8 features:**
1. `value` - Product quantity (e.g., 12.0 oz)
2. `pack_size` - Number of items in pack (e.g., 6)
3. `total_quantity` - value × pack_size (e.g., 72.0)
4. `is_pack` - Binary: 1 if multi-pack, 0 if single
5. `text_length` - Character count of catalog_content
6. `bullet_points` - Number of bullet points in description
7. `has_value` - Binary: 1 if value extracted, 0 otherwise
8. `unit_type` - Categorical: 0=other, 1=weight, 2=volume, 3=count

**IMPROVED MODEL (train_improved.py) - 25 features:**

**Basic (8):**
1-8. Same as baseline above

**Brand & Category (2):**
9. `brand_encoded` - Extracted brand name (e.g., "kraft", "heinz") → label encoded
10. `category_encoded` - Product category (food, beverage, health, beauty, etc.) → label encoded

**Text Quality (10):**
11. `has_capitals` - Has uppercase letters (brand indicator)
12. `word_count` - Number of words
13. `comma_count` - List indicator
14. `has_numbers` - Contains numeric characters
15. `vocab_richness` - Unique words / total words
16. `num_sentences` - Number of sentences
17. `avg_word_length` - Average word length
18. `has_description` - Detailed description (length > 100)
19. `price_keywords` - Premium indicators (organic, deluxe, etc.)
20. `bulk_indicators` - Bulk purchase keywords (family, value, etc.)

**Advanced Quantity (5):**
21. `value_per_unit` - value / pack_size (efficiency metric)
22. `log_total_quantity` - log(1 + total_quantity) (normalized scale)
23. `is_large_pack` - Binary: 1 if pack_size > 4
24. `has_fraction` - Binary: 1 if value has decimal (e.g., 10.5 oz)
25. `quantity_category` - 0-4 scale (tiny to huge)

**Example (Improved Model):**
```
Input: "Item Name: Kraft Organic Tomato Sauce, 12.5 Ounce (Pack of 6)
        Value: 12.5
        Unit: Fl Oz
        Bullet Point: Premium Quality"

Output: [12.5, 6, 75.0, 1, 180, 1, 1, 2,  ← Basic (8)
         15, 0,                              ← Brand: kraft(15), Category: food(0)
         1, 12, 3, 1, 0.83, 2, 8.5, 1, 1, 0, ← Text Quality (10)
         2.08, 4.33, 1, 1, 3]                ← Advanced Quantity (5)
         
Total: 25 features
```

#### 2.2 Text Feature Extraction
**Model:** `sentence-transformers/all-MiniLM-L6-v2`  
**Input:** catalog_content string  
**Output:** 384-dimensional embedding vector

**Process:**
1. Tokenize text using AutoTokenizer
2. Pass through MiniLM model
3. Extract [CLS] token representation
4. Result: 384-dim dense vector representing semantic content

#### 2.3 Image Features (Future Enhancement)
**Current:** Placeholder feature (image_available flag)  
**Future:** CLIP ViT-B/32 embeddings (512-dim vector)

---

### Step 3: Data Preprocessing

**Operations:**
1. **Train-Validation Split**
   - Training: 85% (63,750 samples)
   - Validation: 15% (11,250 samples)
   - Random seed: 42

2. **Feature Scaling**
   - Method: StandardScaler (mean=0, std=1)
   - Applied to: **8 numeric features (baseline)** OR **25 numeric features (improved)**
   - Saved to: `models/feature_scaler.pkl` (baseline) OR `models/feature_scaler_improved.pkl` (improved)
   - Also saves: `models/brand_encoder.pkl` and `models/category_encoder.pkl` (for improved model)

3. **Dataset Creation**
   - PyTorch Dataset class
   - Batch size: 32
   - Shuffle: True (training), False (validation)

---

### Step 4: Model Architecture

**BASELINE MODEL (train.py):**

**Model Name:** `MultimodalPricePredictor`

**Input Dimensions:**
- Text features: 384 (from MiniLM)
- Numeric features: 8 (extracted + engineered)
- **Total input: 392 dimensions**

**Architecture:**
```
Input Layer:        392 dimensions (384 text + 8 numeric)
   ↓
Hidden Layer 1:     256 neurons (ReLU, Dropout 0.3)
   ↓
Hidden Layer 2:     128 neurons (ReLU, Dropout 0.2)
   ↓
Hidden Layer 3:     64 neurons (ReLU, Dropout 0.1)
   ↓
Output Layer:       1 neuron (price prediction)
```

**Trainable Parameters:** ~150,000 (lightweight model)

---

**IMPROVED MODEL (train_improved.py):**

**Model Name:** `ImprovedPricePredictor`

**Input Dimensions:**
- Text features: 384 (from MiniLM)
- Numeric features: **25** (enhanced feature engineering)
- **Total input: 409 dimensions**

**Architecture:**
```
Input Layer:        409 dimensions (384 text + 25 numeric)
   ↓
Hidden Layer 1:     512 neurons (BatchNorm, ReLU, Dropout 0.3)
   ↓
Hidden Layer 2:     512 neurons (BatchNorm, ReLU, Dropout 0.25)
   ↓
Hidden Layer 3:     256 neurons (BatchNorm, ReLU, Dropout 0.2)
   ↓
Hidden Layer 4:     128 neurons (BatchNorm, ReLU, Dropout 0.15)
   ↓
Hidden Layer 5:     64 neurons (BatchNorm, ReLU, Dropout 0.1)
   ↓
Output Layer:       1 neuron (price prediction)
```

**Key Improvements:**
- ✅ Deeper network: 5 hidden layers (vs 3)
- ✅ Larger capacity: 512 hidden dim (vs 256)
- ✅ BatchNormalization: Stabilizes training
- ✅ Progressive dropout: 0.3 → 0.1
- ✅ More features: 25 (vs 8)

**Trainable Parameters:** ~400,000 (still lightweight)

---

### Step 5: Training Process

**BASELINE MODEL (train.py):**

**Loss Function:** SMAPE (Symmetric Mean Absolute Percentage Error) - Competition metric  
**Optimizer:** AdamW with weight decay 0.01  
**Learning Rate:** 2e-4 with ReduceLROnPlateau scheduler

**Training Configuration:**
- **Epochs:** 15 (with early stopping)
- **Early Stopping:** Patience = 5 epochs
- **Gradient Clipping:** Max norm = 1.0
- **Batch Size:** 32
- **Max Sequence Length:** 128 tokens

---

**IMPROVED MODEL (train_improved.py):**

**Loss Function:** SMAPE (Symmetric Mean Absolute Percentage Error) - Competition metric  
**Optimizer:** AdamW with weight decay 0.01  
**Learning Rate:** 1e-4 with **Warmup + Cosine Annealing**

**Training Configuration:**
- **Epochs:** 25 (with early stopping)
- **Warmup:** 2 epochs (gradual learning rate increase)
- **Cosine Annealing:** Smooth learning rate decay
- **Early Stopping:** Patience = 7 epochs (more patience)
- **Gradient Clipping:** Max norm = 1.0
- **Batch Size:** 16 (with gradient accumulation = 4, effective batch = 64)
- **Max Sequence Length:** 256 tokens (captures more context)
- **Mixed Precision Training:** Faster training, less memory

**Key Training Improvements:**
- ✅ Learning rate warmup: Prevents early instability
- ✅ Cosine annealing: Smooth LR decay
- ✅ Gradient accumulation: Larger effective batch size
- ✅ Mixed precision: 2× faster training
- ✅ Longer sequences: More product information captured

**Training Loop (per epoch):**
1. Forward pass: Predict prices
2. Calculate SMAPE loss
3. Backward pass: Compute gradients
4. Gradient accumulation (every 4 steps for improved model)
5. Update weights via AdamW + scheduler
6. Validate on validation set
7. Save best model based on validation SMAPE

**Checkpointing:**
- Best model saved to: `models/best_model.pth`
- Contains: model weights, optimizer state, epoch number, validation loss

---

### Step 6: Output Artifacts

**Saved Files:**
1. `models/best_model.pth` → Trained neural network weights (5-10 MB)
2. `models/feature_scaler.pkl` → StandardScaler object for feature normalization

---

## 🔮 PHASE 2: INFERENCE PIPELINE (sample_code.py)

### Step 1: Model Initialization
**Function:** `initialize_models()`  
**Loaded Components:**
1. Tokenizer (sentence-transformers/all-MiniLM-L6-v2)
2. Trained model weights from `models/best_model.pth`
3. Feature scaler from `models/feature_scaler.pkl`
4. Device setup (CUDA/CPU)

---

### Step 2: Test Data Loading
**File:** `student_resource/dataset/test.csv`  
**Columns:** sample_id, catalog_content, image_link  
**Samples:** 75,000 test products (no price labels)

---

### Step 3: Prediction Process (per sample)

**Function:** `predictor(sample_id, catalog_content, image_link)`

**Step-by-Step Flow:**

1. **Text Processing**
   ```
   catalog_content → Tokenizer → input_ids, attention_mask
   ```

2. **Numeric Feature Extraction**
   ```
   catalog_content → extract_numeric_features() → 8 features
   ```

3. **Feature Normalization**
   ```
   8 numeric features → StandardScaler.transform() → normalized features
   ```

4. **Feature Fusion**
   ```
   Text embedding (384-dim) + Numeric features (8-dim) → 392-dim vector
   ```

5. **Model Inference**
   ```
   392-dim vector → MultimodalPricePredictor → predicted price (float)
   ```

6. **Post-processing**
   ```
   Ensure price > 0.01, round to 2 decimal places
   ```

---

### Step 4: Batch Predictions

**Process:**
1. Load all 75,000 test samples
2. Apply `predictor()` function to each row
3. Collect predictions in DataFrame
4. Format output with columns: [sample_id, price]

---

### Step 5: Output Generation

**File:** `student_resource/dataset/test_out.csv`

**Format:**
```csv
sample_id,price
100179,15.67
100180,8.99
100181,23.45
...
```

**Validation:**
- Exactly 75,000 predictions
- All prices are positive floats
- All sample_ids match test.csv

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (train.py)                    │
└─────────────────────────────────────────────────────────────────┘

train.csv (75K samples)
    │
    ├─→ Extract Numeric Features (8 features)
    ├─→ Tokenize Text (MiniLM tokenizer)
    └─→ Create PyTorch Dataset
         │
         ├─→ Train Split (85%) ──┐
         └─→ Val Split (15%)      │
                                  ▼
                         Train MultimodalPricePredictor
                           (10-15 epochs, Early Stopping)
                                  │
                                  ├─→ Save best_model.pth
                                  └─→ Save feature_scaler.pkl


┌─────────────────────────────────────────────────────────────────┐
│                  INFERENCE PHASE (sample_code.py)               │
└─────────────────────────────────────────────────────────────────┘

test.csv (75K samples, no prices)
    │
    └─→ Load Models (best_model.pth, feature_scaler.pkl)
         │
         └─→ For each sample:
              │
              ├─→ Extract Numeric Features (8)
              ├─→ Tokenize Text (384-dim)
              ├─→ Normalize Numeric Features
              ├─→ Concatenate Features (392-dim)
              ├─→ Model Inference
              └─→ Predicted Price
                   │
                   └─→ Collect all predictions
                        │
                        └─→ test_out.csv (75K predictions)
```

---

## 📌 Key Data Transformations

### Example: Single Product Flow

**Input (test.csv row):**
```
sample_id: 100179
catalog_content: "Rani 14-Spice Mango Chutney, 10.5oz, Value: 10.5, Unit: Ounce"
image_link: https://m.media-amazon.com/images/I/example.jpg
```

**Step 1: Numeric Features**
```python
[10.5, 1, 10.5, 0, 250, 5, 1, 1]
# [value, pack_size, total_qty, is_pack, text_len, bullets, has_val, unit_type]
```

**Step 2: Normalized Numeric Features**
```python
[0.23, -0.45, 0.18, -0.89, 0.56, 0.12, 1.34, 0.67]
# After StandardScaler transformation
```

**Step 3: Text Embedding**
```python
[0.023, -0.145, 0.678, ..., 0.234]  # 384 dimensions
```

**Step 4: Combined Features**
```python
[0.023, -0.145, ..., 0.234, 0.23, -0.45, 0.18, -0.89, 0.56, 0.12, 1.34, 0.67]
# 392 dimensions total
```

**Step 5: Model Prediction**
```python
15.67  # Predicted price in USD
```

**Output (test_out.csv row):**
```
sample_id,price
100179,15.67
```

---

## 🎯 Performance Considerations

### Training Time (CPU)
- Per epoch: ~30-45 minutes
- Total (15 epochs): ~7-10 hours
- **Recommendation:** Use Kaggle GPU (reduces to ~30-60 minutes total)

### Training Time (GPU - P100/T4)
- Per epoch: ~2-3 minutes
- Total (15 epochs): ~30-45 minutes

### Inference Time
- Per sample: ~50-100ms (CPU)
- Total (75K samples): ~1-2 hours (CPU)
- Total (75K samples): ~15-30 minutes (GPU)

---

## 🚨 Error Handling & Fallbacks

### Missing Catalog Content
→ Use empty string, numeric features = zeros

### Missing Image Link
→ Set image_available = 0, continue with text features

### Model Loading Failure
→ Fallback to heuristic: price = total_quantity × 0.15

### Prediction Error
→ Return random price between $5-$500

---

## 📁 Data Persistence

**Intermediate Files (Created During Training):**
- `models/best_model.pth` (5-10 MB) - Keep for inference
- `models/feature_scaler.pkl` (<1 MB) - Keep for inference

**Final Output:**
- `student_resource/dataset/test_out.csv` (~2 MB) - Submit to competition

**Git Repository:**
- Include: train.py, sample_code.py, requirements.txt, documentation
- Exclude: models/, dataset/, test_out.csv (use .gitignore)

---

## ✅ Verification Checklist

- [ ] train.csv loaded successfully (75,000 rows)
- [ ] Numeric features extracted (8 features per sample)
- [ ] Model trained (best_model.pth saved)
- [ ] Feature scaler saved (feature_scaler.pkl)
- [ ] test.csv loaded successfully (75,000 rows)
- [ ] All 75,000 predictions generated
- [ ] test_out.csv format matches sample_test_out.csv
- [ ] All prices are positive floats
- [ ] No missing sample_ids

---

**End of Data Flow Documentation**

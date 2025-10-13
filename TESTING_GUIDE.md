# 🧪 Testing Guide - Validate Before Full Inference

## Why Test on Sample Data First?

Before running predictions on all 75,000 test samples, it's smart to validate on the smaller `sample_test.csv`:

✅ **Fast validation** - 100 samples vs 75,000 samples (seconds vs minutes)
✅ **Early SMAPE estimate** - Know your expected score before full run
✅ **Catch errors** - Debug issues on small dataset first
✅ **Verify model** - Confirm model loaded and works correctly
✅ **Performance insights** - See best/worst predictions, error patterns

---

## 🚀 Quick Start

### Step 1: Train Your Model

```bash
python train.py
```

Wait for training to complete (~30-45 min on GPU)

### Step 2: Test on Sample Data

```bash
python test_sample.py
```

This runs in **seconds** and shows:
- SMAPE score (competition metric)
- MAE, RMSE, MAPE (additional metrics)
- Best and worst predictions
- Price distribution comparison
- Performance verdict

### Step 3: Review Results

Check the output:

```
📈 Performance Metrics:
==================================================================
MAE (Mean Absolute Error):        $3.45
SMSE (Root Mean Squared Error):   $5.23
MAPE (Mean Abs Percentage Error): 18.50%
SMAPE (Competition Metric):       22.15%

🎯 Performance Verdict:
==================================================================
✅ GOOD! SMAPE < 25% - Strong baseline performance!
```

### Step 4: Decide Next Step

**If SMAPE < 25%:**
```bash
# Great! Run full predictions
python sample_code.py
```

**If SMAPE > 35%:**
```bash
# Re-train or tune hyperparameters
# Edit train.py, then re-run
python train.py
python test_sample.py  # Test again
```

---

## 📊 Understanding the Output

### Files Generated

1. **`sample_test_predictions.csv`** - Your model's predictions
   ```csv
   sample_id,price
   123,15.67
   124,8.99
   ...
   ```

2. **`sample_test_comparison.csv`** - Detailed comparison (if ground truth exists)
   ```csv
   sample_id,price_pred,price_actual,error,abs_error,pct_error
   123,15.67,14.50,1.17,1.17,8.07
   124,8.99,9.50,-0.51,0.51,5.37
   ...
   ```

### Metrics Explained

**MAE (Mean Absolute Error)**
- Average dollar amount of error
- Example: MAE = $3.45 means average error is $3.45

**RMSE (Root Mean Squared Error)**
- Like MAE but penalizes large errors more
- Higher = some predictions are way off

**MAPE (Mean Absolute Percentage Error)**
- Average percentage error
- Example: MAPE = 18.5% means average 18.5% off

**SMAPE (Symmetric Mean Absolute Percentage Error)** ⭐
- **Competition metric!**
- Treats over/under prediction equally
- Bounded: 0% (perfect) to 200% (worst)

### Performance Targets

| SMAPE Score | Performance | Action |
|-------------|-------------|--------|
| < 15% | 🌟 Excellent | Submit immediately! |
| 15-20% | ✅ Very Good | Strong competitive score |
| 20-25% | ✅ Good | Solid baseline |
| 25-35% | ⚠️ Acceptable | Room for improvement |
| > 35% | ❌ Needs Work | Re-train or tune |

---

## 🔍 Detailed Analysis

### Sample Output Sections

#### 1. Sample Predictions Table

```
sample_id  price_actual  price_pred   error  pct_error
------------------------------------------------------
123        14.50         15.67        1.17   8.07
124        9.50          8.99        -0.51   5.37
125        23.00         24.50        1.50   6.52
...
```

Shows first 10 predictions with errors.

#### 2. Performance Metrics

All key metrics calculated automatically:
- MAE, RMSE, MAPE, SMAPE

#### 3. Best Prediction

```
🏆 Best Prediction:
Sample ID: 456
Actual Price: $12.99
Predicted Price: $12.95
Error: $-0.04 (0.31%)
```

Your most accurate prediction!

#### 4. Worst Prediction

```
❌ Worst Prediction:
Sample ID: 789
Actual Price: $8.50
Predicted Price: $15.23
Error: $6.73 (79.18%)
```

Helps identify patterns in errors.

#### 5. Price Distribution

```
Metric               Actual          Predicted
--------------------------------------------------
Mean                 $25.45          $26.12
Median               $18.90          $19.45
Min                  $2.50           $3.15
Max                  $149.99         $145.67
Std Dev              $22.15          $21.89
```

Check if predictions match actual distribution.

---

## 🐛 Troubleshooting

### Issue: Model not found

```
⚠ Warning: Trained model not found. Using random predictions.
```

**Solution:**
```bash
# Train the model first
python train.py

# Then test
python test_sample.py
```

### Issue: sample_test.csv not found

```
FileNotFoundError: sample_test.csv
```

**Solution:**
- Ensure you're in the correct directory
- Check `student_resource/dataset/sample_test.csv` exists
- Dataset files should be in the correct folder

### Issue: No ground truth found

```
💡 No ground truth found (sample_test_out.csv)
   Predictions saved but can't calculate SMAPE
```

**Solution:**
- This is OK if you don't have `sample_test_out.csv`
- You'll see predictions but no accuracy metrics
- Still useful to verify model runs

### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Sample test uses very little memory
- If this happens, something else is using GPU
- Restart Python kernel or use CPU:
  ```python
  DEVICE = torch.device('cpu')
  ```

---

## 💡 Tips for Better Results

### Analyzing Errors

**If errors are consistent (all predictions too high/low):**
- Model has bias
- Check feature scaling
- Verify training converged

**If errors vary wildly:**
- Model uncertainty
- May need more training
- Check feature extraction

**If errors on specific price ranges:**
- Feature engineering opportunity
- Add price range indicators
- Use stratified sampling

### Quick Iterations

```bash
# Workflow for tuning
python train.py              # Train
python test_sample.py        # Quick test
# Adjust hyperparameters
python train.py              # Re-train
python test_sample.py        # Test again
# Repeat until SMAPE < 25%
python sample_code.py        # Full run
```

---

## 📋 Checklist Before Full Inference

Run through this checklist after `python test_sample.py`:

- [ ] Model loaded successfully (no warnings)
- [ ] SMAPE calculated (if ground truth exists)
- [ ] SMAPE < 30% (acceptable performance)
- [ ] No systematic bias (predictions not all high/low)
- [ ] Price distribution similar to actual
- [ ] No errors/exceptions during prediction

**If all checked:** ✅ Ready for full inference!

```bash
python sample_code.py
```

**If any issues:** ⚠️ Debug and re-train before full run

---

## 🎯 Example Workflow

### Complete Testing Workflow

```bash
# 1. Initial training
python train.py
# Wait ~30-45 min

# 2. Quick validation
python test_sample.py
# Output: SMAPE = 28%

# 3. Tune hyperparameters (if needed)
# Edit train.py: learning_rate = 1e-4
python train.py

# 4. Test again
python test_sample.py
# Output: SMAPE = 23%

# 5. Looks good! Full inference
python sample_code.py
# Generates test_out.csv

# 6. Submit to competition
# Upload test_out.csv to competition portal
```

---

## 🚀 Advanced Usage

### Custom Analysis

You can modify `test_sample.py` to add custom analysis:

```python
# Add after line 280
# Custom: Check predictions by price range
low_price = comparison[comparison['price_actual'] < 10]
mid_price = comparison[(comparison['price_actual'] >= 10) & (comparison['price_actual'] < 50)]
high_price = comparison[comparison['price_actual'] >= 50]

print(f"Low price SMAPE: {calculate_smape(low_price['price_actual'], low_price['price_pred']):.2f}%")
print(f"Mid price SMAPE: {calculate_smape(mid_price['price_actual'], mid_price['price_pred']):.2f}%")
print(f"High price SMAPE: {calculate_smape(high_price['price_actual'], high_price['price_pred']):.2f}%")
```

### Batch Testing

Test multiple models:

```bash
# Train model 1
python train.py  # with default params
python test_sample.py > results_model1.txt

# Train model 2
# Edit train.py: hidden_dim = 512
python train.py
python test_sample.py > results_model2.txt

# Compare results
diff results_model1.txt results_model2.txt
```

---

## 📊 Expected vs Actual

### What to Expect

**On sample_test.csv (100 samples):**
- Runtime: 5-30 seconds
- SMAPE: 18-30% (depending on model quality)
- Some predictions very accurate, some off
- Overall trend should match actual prices

**On full test.csv (75,000 samples):**
- Runtime: 15-30 minutes (GPU) or 1-2 hours (CPU)
- SMAPE: Usually within 2-5% of sample test SMAPE
- More stable statistics
- Better confidence in final score

---

## ✅ Summary

**test_sample.py is your validation tool before committing to full inference.**

**Benefits:**
- ⚡ Fast feedback (seconds)
- 📊 SMAPE estimation
- 🐛 Early error detection
- 💡 Performance insights
- 🎯 Confidence building

**Workflow:**
1. Train model (`train.py`)
2. Test on sample (`test_sample.py`)
3. Review SMAPE and metrics
4. If good (< 25%), run full inference (`sample_code.py`)
5. If not, tune and repeat

**Always test before full inference!**

---

**Created:** October 11, 2025  
**Purpose:** Validate model performance before full test.csv inference

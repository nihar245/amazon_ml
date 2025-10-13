"""
Amazon ML Challenge 2025 - Sample Test Script
Test model on sample_test.csv before running on full test.csv
"""

import os
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==================== GLOBAL MODEL INITIALIZATION ====================

TOKENIZER = None
MODEL = None
SCALER = None
DEVICE = None

def initialize_models():
    """Initialize models once at startup"""
    global TOKENIZER, MODEL, SCALER, DEVICE
    
    if MODEL is not None:
        return  # Already initialized
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    print("Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    scaler_path = 'models/feature_scaler.pkl'
    if os.path.exists(scaler_path):
        print("Loading feature scaler...")
        with open(scaler_path, 'rb') as f:
            SCALER = pickle.load(f)
    else:
        print("⚠ Warning: Feature scaler not found. Using dummy scaler.")
        SCALER = None
    
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        print("Loading trained model...")
        MODEL = MultimodalPricePredictor(
            text_model_name='sentence-transformers/all-MiniLM-L6-v2',
            numeric_dim=8,
            hidden_dim=256
        ).to(DEVICE)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()
        print("✓ Model loaded successfully")
    else:
        print("⚠ Warning: Trained model not found. Using random predictions.")
        MODEL = None

# ==================== MODEL ARCHITECTURE ====================

class MultimodalPricePredictor(nn.Module):
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 numeric_dim=8, hidden_dim=256):
        super(MultimodalPricePredictor, self).__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask, numeric_features):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat([text_features, numeric_features], dim=1)
        price = self.fusion(combined).squeeze(-1)
        return price

# ==================== FEATURE EXTRACTION ====================

def extract_numeric_features(catalog_content):
    """Extract numeric features from catalog content"""
    features = {
        'value': 0.0,
        'pack_size': 1,
        'total_quantity': 0.0,
        'is_pack': 0,
        'text_length': 0,
        'bullet_points': 0,
        'has_value': 0,
        'unit_type': 0
    }
    
    if pd.isna(catalog_content) or not catalog_content:
        return features
    
    text = str(catalog_content)
    features['text_length'] = len(text)
    features['bullet_points'] = text.count('Bullet Point')
    
    value_match = re.search(r'Value:\s*([\d.]+)', text, re.IGNORECASE)
    if value_match:
        features['value'] = float(value_match.group(1))
        features['has_value'] = 1
    
    unit_match = re.search(r'Unit:\s*(\w+(?:\s+\w+)?)', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).lower()
        if 'oz' in unit or 'ounce' in unit or 'lb' in unit or 'pound' in unit:
            if 'fl' in unit or 'fluid' in unit:
                features['unit_type'] = 2
            else:
                features['unit_type'] = 1
        elif 'count' in unit or 'piece' in unit:
            features['unit_type'] = 3
    
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
    
    features['total_quantity'] = features['value'] * features['pack_size']
    
    return features

# ==================== PREDICTOR FUNCTION ====================

def predictor(sample_id, catalog_content, image_link):
    '''
    Predict product price
    '''
    if MODEL is None or TOKENIZER is None:
        return round(random.uniform(5.0, 500.0), 2)
    
    try:
        catalog_text = str(catalog_content) if not pd.isna(catalog_content) else ""
        
        encoding = TOKENIZER(
            catalog_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        numeric_dict = extract_numeric_features(catalog_content)
        numeric_features = np.array([[
            numeric_dict['value'],
            numeric_dict['pack_size'],
            numeric_dict['total_quantity'],
            numeric_dict['is_pack'],
            numeric_dict['text_length'],
            numeric_dict['bullet_points'],
            numeric_dict['has_value'],
            numeric_dict['unit_type']
        ]], dtype=np.float32)
        
        if SCALER is not None:
            numeric_features = SCALER.transform(numeric_features)
        
        numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            MODEL.eval()
            prediction = MODEL(input_ids, attention_mask, numeric_tensor)
            price = prediction.item()
        
        price = max(0.01, price)
        return round(price, 2)
    
    except Exception as e:
        print(f"Error predicting for sample {sample_id}: {str(e)}")
        try:
            features = extract_numeric_features(catalog_content)
            if features['total_quantity'] > 0:
                base_price = features['total_quantity'] * 0.15
                return round(max(1.0, base_price), 2)
        except:
            pass
        return round(random.uniform(5.0, 500.0), 2)

# ==================== SMAPE CALCULATION ====================

def calculate_smape(actual, predicted, epsilon=1e-8):
    """Calculate SMAPE between actual and predicted prices"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0 + epsilon
    smape = np.abs(predicted - actual) / denominator
    return np.mean(smape) * 100

# ==================== MAIN TEST SCRIPT ====================

if __name__ == "__main__":
    DATASET_FOLDER = 'student_resource/dataset/'
    
    print("="*70)
    print("Amazon ML Challenge 2025 - SAMPLE TEST (Validation)")
    print("="*70)
    
    # Initialize models
    initialize_models()
    print("="*70)
    
    # Read sample test data
    print("\n📂 Loading sample_test.csv...")
    sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
    print(f"✓ Sample test samples: {len(sample_test)}")
    
    # Check if sample_test_out.csv exists (for SMAPE calculation)
    sample_out_path = os.path.join(DATASET_FOLDER, 'sample_test_out.csv')
    has_ground_truth = os.path.exists(sample_out_path)
    
    if has_ground_truth:
        print("✓ Found sample_test_out.csv (ground truth) - will calculate SMAPE")
        ground_truth = pd.read_csv(sample_out_path)
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    sample_test['price'] = sample_test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Prepare output
    output_df = sample_test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'sample_test_predictions.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n✓ Predictions saved to: sample_test_predictions.csv")
    print(f"✓ Total predictions: {len(output_df)}")
    
    # Display sample predictions
    print("\n" + "="*70)
    print("📊 Sample Predictions:")
    print("="*70)
    
    if has_ground_truth:
        # Merge with ground truth
        comparison = output_df.merge(ground_truth, on='sample_id', suffixes=('_pred', '_actual'))
        comparison['error'] = comparison['price_pred'] - comparison['price_actual']
        comparison['abs_error'] = abs(comparison['error'])
        comparison['pct_error'] = (comparison['abs_error'] / comparison['price_actual']) * 100
        
        print(comparison[['sample_id', 'price_actual', 'price_pred', 'error', 'pct_error']].head(10).to_string(index=False))
        
        # Calculate metrics
        print("\n" + "="*70)
        print("📈 Performance Metrics:")
        print("="*70)
        
        mae = comparison['abs_error'].mean()
        rmse = np.sqrt((comparison['error'] ** 2).mean())
        mape = comparison['pct_error'].mean()
        smape = calculate_smape(comparison['price_actual'], comparison['price_pred'])
        
        print(f"MAE (Mean Absolute Error):        ${mae:.2f}")
        print(f"RMSE (Root Mean Squared Error):   ${rmse:.2f}")
        print(f"MAPE (Mean Abs Percentage Error): {mape:.2f}%")
        print(f"SMAPE (Competition Metric):       {smape:.2f}%")
        
        # Best and worst predictions
        best_idx = comparison['abs_error'].idxmin()
        worst_idx = comparison['abs_error'].idxmax()
        
        print("\n" + "="*70)
        print("🏆 Best Prediction:")
        print("="*70)
        best = comparison.loc[best_idx]
        print(f"Sample ID: {best['sample_id']}")
        print(f"Actual Price: ${best['price_actual']:.2f}")
        print(f"Predicted Price: ${best['price_pred']:.2f}")
        print(f"Error: ${best['error']:.2f} ({best['pct_error']:.2f}%)")
        
        print("\n" + "="*70)
        print("❌ Worst Prediction:")
        print("="*70)
        worst = comparison.loc[worst_idx]
        print(f"Sample ID: {worst['sample_id']}")
        print(f"Actual Price: ${worst['price_actual']:.2f}")
        print(f"Predicted Price: ${worst['price_pred']:.2f}")
        print(f"Error: ${worst['error']:.2f} ({worst['pct_error']:.2f}%)")
        
        # Price distribution comparison
        print("\n" + "="*70)
        print("📊 Price Distribution:")
        print("="*70)
        print(f"{'Metric':<20} {'Actual':<15} {'Predicted':<15}")
        print("-"*50)
        print(f"{'Mean':<20} ${comparison['price_actual'].mean():<14.2f} ${comparison['price_pred'].mean():<14.2f}")
        print(f"{'Median':<20} ${comparison['price_actual'].median():<14.2f} ${comparison['price_pred'].median():<14.2f}")
        print(f"{'Min':<20} ${comparison['price_actual'].min():<14.2f} ${comparison['price_pred'].min():<14.2f}")
        print(f"{'Max':<20} ${comparison['price_actual'].max():<14.2f} ${comparison['price_pred'].max():<14.2f}")
        print(f"{'Std Dev':<20} ${comparison['price_actual'].std():<14.2f} ${comparison['price_pred'].std():<14.2f}")
        
        # Save comparison
        comparison_file = os.path.join(DATASET_FOLDER, 'sample_test_comparison.csv')
        comparison.to_csv(comparison_file, index=False)
        print(f"\n✓ Detailed comparison saved to: sample_test_comparison.csv")
        
        # Performance verdict
        print("\n" + "="*70)
        print("🎯 Performance Verdict:")
        print("="*70)
        if smape < 20:
            print("🌟 EXCELLENT! SMAPE < 20% - Competitive performance!")
        elif smape < 25:
            print("✅ GOOD! SMAPE < 25% - Strong baseline performance!")
        elif smape < 35:
            print("⚠️  ACCEPTABLE! SMAPE < 35% - Room for improvement")
        else:
            print("❌ NEEDS IMPROVEMENT! SMAPE > 35% - Consider retraining or feature engineering")
        
    else:
        # No ground truth - just show predictions
        print(output_df.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("📊 Prediction Statistics:")
        print("="*70)
        print(f"Mean Price:   ${output_df['price'].mean():.2f}")
        print(f"Median Price: ${output_df['price'].median():.2f}")
        print(f"Min Price:    ${output_df['price'].min():.2f}")
        print(f"Max Price:    ${output_df['price'].max():.2f}")
        print(f"Std Dev:      ${output_df['price'].std():.2f}")
    
    print("\n" + "="*70)
    print("✅ Sample test complete!")
    print("="*70)
    
    if has_ground_truth:
        print(f"\n💡 SMAPE on sample_test.csv: {smape:.2f}%")
        print("   This gives you an estimate of performance on test.csv")
        print("\n🚀 Next step: Run 'python sample_code.py' for full test.csv predictions")
    else:
        print("\n💡 No ground truth found (sample_test_out.csv)")
        print("   Predictions saved but can't calculate SMAPE")
        print("\n🚀 Next step: Run 'python sample_code.py' for full test.csv predictions")

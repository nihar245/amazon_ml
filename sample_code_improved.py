"""
Amazon ML Challenge 2025 - Inference Script for IMPROVED Model
This script loads the improved model (25 features) and generates predictions
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*60)
print("Amazon ML Challenge 2025 - Improved Model Inference")
print("="*60)
print(f"Using device: {device}")

# ==================== FEATURE EXTRACTION (Same as train_improved.py) ====================

def extract_brand(text):
    """Extract brand name from product text"""
    if pd.isna(text) or not text:
        return "unknown"
    
    original_text = str(text)
    text_lower = original_text.lower()
    
    common_brands = [
        'amazon', 'kraft', 'nestle', 'pepsi', 'coca', 'heinz', 'campbell',
        'general mills', 'kellogg', 'unilever', 'procter', 'johnson', 'colgate',
        'palmolive', 'gillette', 'dove', 'axe', 'tide', 'gain', 'bounty',
        'charmin', 'pampers', 'huggies', 'kleenex', 'scotts', 'lysol',
        'clorox', 'ajax', 'cascade', 'dawn', 'pledge', 'windex', 'ziploc'
    ]
    
    for brand in common_brands:
        if brand in text_lower:
            return brand
    
    words = original_text.split()
    for i, word in enumerate(words):
        if len(word) > 2 and len(word) < 20 and word[0].isupper():
            if i + 1 < len(words) and len(words[i + 1]) > 2 and words[i + 1][0].isupper():
                return (word + ' ' + words[i + 1]).lower()
            return word.lower()
    
    return "unknown"

def classify_category(text):
    """Classify product into categories"""
    if pd.isna(text) or not text:
        return "other"
    text = str(text).lower()
    categories = {
        'food': ['food', 'snack', 'sauce', 'pasta', 'rice', 'cereal', 'bread', 
                 'cookie', 'cracker', 'chip', 'nut', 'candy', 'chocolate', 'granola'],
        'beverage': ['drink', 'juice', 'soda', 'water', 'tea', 'coffee', 'beverage'],
        'health': ['vitamin', 'supplement', 'medicine', 'health', 'protein', 'fitness'],
        'beauty': ['shampoo', 'soap', 'lotion', 'cream', 'cosmetic', 'makeup', 'skincare'],
        'household': ['cleaner', 'detergent', 'paper', 'towel', 'tissue', 'trash', 'bag'],
        'baby': ['baby', 'diaper', 'formula', 'wipes', 'infant'],
        'pet': ['pet', 'dog', 'cat', 'animal', 'puppy', 'kitten']
    }
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "other"

def extract_text_quality_features(text):
    """Extract text quality features"""
    if pd.isna(text) or not text:
        return {
            'has_capitals': 0, 'word_count': 0, 'comma_count': 0, 'has_numbers': 0,
            'vocab_richness': 0.0, 'num_sentences': 0, 'avg_word_length': 0.0,
            'has_description': 0, 'price_keywords': 0, 'bulk_indicators': 0
        }
    
    text_str = str(text)
    words = text_str.split()
    
    features = {}
    features['has_capitals'] = 1 if any(c.isupper() for c in text_str) else 0
    features['word_count'] = len(words)
    features['comma_count'] = text_str.count(',')
    features['has_numbers'] = 1 if any(c.isdigit() for c in text_str) else 0
    
    unique_words = len(set(words))
    features['vocab_richness'] = unique_words / max(len(words), 1)
    
    sentences = text_str.split('.')
    features['num_sentences'] = len([s for s in sentences if len(s.strip()) > 0])
    
    word_lengths = [len(w) for w in words if len(w) > 0]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0.0
    
    features['has_description'] = 1 if len(text_str) > 100 else 0
    
    price_keywords = ['premium', 'deluxe', 'gourmet', 'organic', 'natural', 
                      'artisan', 'professional', 'luxury', 'quality']
    features['price_keywords'] = sum(1 for kw in price_keywords if kw in text_str.lower())
    
    bulk_keywords = ['bulk', 'family', 'value', 'economy', 'jumbo', 'super', 'mega']
    features['bulk_indicators'] = sum(1 for kw in bulk_keywords if kw in text_str.lower())
    
    return features

def extract_enhanced_features(catalog_content):
    """Extract comprehensive feature set (25 features)"""
    features = {
        'value': 0.0, 'pack_size': 1, 'total_quantity': 0.0, 'is_pack': 0,
        'text_length': 0, 'bullet_points': 0, 'has_value': 0, 'unit_type': 0,
        'brand': 'unknown', 'category': 'other',
        'brand_encoded': 0, 'category_encoded': 0,
        'has_capitals': 0, 'word_count': 0, 'comma_count': 0, 'has_numbers': 0,
        'vocab_richness': 0.0, 'num_sentences': 0, 'avg_word_length': 0.0,
        'has_description': 0, 'price_keywords': 0, 'bulk_indicators': 0,
        'value_per_unit': 0.0, 'log_total_quantity': 0.0, 'is_large_pack': 0,
        'has_fraction': 0, 'quantity_category': 0
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
        features['has_fraction'] = 1 if '.' in value_match.group(1) else 0
    
    unit_match = re.search(r'Unit:\s*(\w+(?:\s+\w+)?)', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).lower()
        if 'oz' in unit or 'lb' in unit or 'pound' in unit or 'gram' in unit:
            features['unit_type'] = 2 if 'fl' in unit or 'fluid' in unit else 1
        elif 'count' in unit or 'piece' in unit:
            features['unit_type'] = 3
    
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
        features['is_large_pack'] = 1 if features['pack_size'] > 4 else 0
    
    features['total_quantity'] = features['value'] * features['pack_size']
    features['log_total_quantity'] = np.log1p(features['total_quantity'])
    features['value_per_unit'] = features['value'] / max(features['pack_size'], 1)
    
    tq = features['total_quantity']
    if tq == 0:
        features['quantity_category'] = 0
    elif tq < 5:
        features['quantity_category'] = 1
    elif tq < 20:
        features['quantity_category'] = 2
    elif tq < 50:
        features['quantity_category'] = 3
    else:
        features['quantity_category'] = 4
    
    brand = extract_brand(text)
    category = classify_category(text)
    features['brand'] = brand
    features['category'] = category
    
    quality_features = extract_text_quality_features(text)
    features.update(quality_features)
    
    return features

# ==================== DATASET CLASS ====================

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, brand_encoder, category_encoder, scaler, max_length=384):
        self.df = df
        self.tokenizer = tokenizer
        self.brand_encoder = brand_encoder
        self.category_encoder = category_encoder
        self.scaler = scaler
        self.max_length = max_length
        
        print(f"Extracting features for {len(df)} test samples...")
        self.features_list = []
        for idx in tqdm(range(len(df)), desc="Feature extraction"):
            catalog_content = df.iloc[idx]['catalog_content']
            features = extract_enhanced_features(catalog_content)
            self.features_list.append(features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = dict(self.features_list[idx])
        
        catalog_text = str(row['catalog_content']) if not pd.isna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            catalog_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode brand and category
        brand_val = features.get('brand', 'unknown')
        cat_val = features.get('category', 'other')
        
        try:
            brand_encoded = self.brand_encoder.transform([brand_val])[0]
        except:
            brand_encoded = 0  # Unknown brand
        
        try:
            category_encoded = self.category_encoder.transform([cat_val])[0]
        except:
            category_encoded = 0  # Unknown category
        
        # Create numeric feature vector (25 features)
        numeric_features = np.array([
            features['value'],
            features['pack_size'],
            features['total_quantity'],
            features['is_pack'],
            features['text_length'],
            features['bullet_points'],
            features['has_value'],
            features['unit_type'],
            brand_encoded,
            category_encoded,
            features['has_capitals'],
            features['word_count'],
            features['comma_count'],
            features['has_numbers'],
            features['vocab_richness'],
            features['num_sentences'],
            features['avg_word_length'],
            features['has_description'],
            features['price_keywords'],
            features['bulk_indicators'],
            features['value_per_unit'],
            features['log_total_quantity'],
            features['is_large_pack'],
            features['has_fraction'],
            features['quantity_category']
        ], dtype=np.float32)
        
        # Apply scaling
        numeric_features = self.scaler.transform(numeric_features.reshape(1, -1))[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(numeric_features, dtype=torch.float32),
            'sample_id': row['sample_id']
        }

# ==================== MODEL CLASS ====================

class ImprovedPricePredictor(nn.Module):
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 numeric_dim=25, hidden_dim=512):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim // 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask, numeric_features):
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat([text_features, numeric_features], dim=1)
        price = self.fusion(combined).squeeze(-1)
        return price

# ==================== MAIN INFERENCE ====================

def main():
    # Load tokenizer
    print("\nLoading tokenizer (DeBERTa-v3-base)...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    # Load scaler and encoders
    print("Loading scaler and encoders...")
    try:
        with open('models/feature_scaler_improved.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/brand_encoder.pkl', 'rb') as f:
            brand_encoder = pickle.load(f)
        with open('models/category_encoder.pkl', 'rb') as f:
            category_encoder = pickle.load(f)
        print("✓ Scaler and encoders loaded successfully")
    except Exception as e:
        print(f"✗ Error loading scaler/encoders: {e}")
        return
    
    # Load model
    print("Loading trained model...")
    try:
        model = ImprovedPricePredictor(
            text_model_name='microsoft/deberta-v3-base',
            numeric_dim=25,
            hidden_dim=768
        ).to(device)
        
        checkpoint = torch.load('models/best_model_ultra.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model loaded successfully")
        
        # Display validation SMAPE if available
        val_loss = checkpoint.get('val_loss', None)
        if val_loss is not None:
            print(f"  Validation SMAPE from training: {val_loss:.2f}%")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv('student_resource/dataset/test.csv')
    print(f"Test samples: {len(test_df)}")
    
    # Create dataset
    print("\nCreating test dataset...")
    test_dataset = TestDataset(
        test_df, tokenizer, brand_encoder, category_encoder, scaler, max_length=384
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            
            preds = model(input_ids, attention_mask, numeric_features)
            predictions.extend(preds.cpu().numpy())
            sample_ids.extend(batch['sample_id'].numpy())
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Save predictions
    output_path = 'student_resource/dataset/test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to {output_path}")
    print(f"✓ Total predictions: {len(output_df)}")
    
    print("\nSample predictions:")
    print(output_df.head(10))
    
    print("\nPrice statistics:")
    print(f"  Mean: ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print(f"  Min: ${output_df['price'].min():.2f}")
    print(f"  Max: ${output_df['price'].max():.2f}")
    print("="*60)

if __name__ == "__main__":
    main()

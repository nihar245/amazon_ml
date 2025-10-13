"""
Amazon ML Challenge 2025 - BLAZING Model Inference
Uses DistilRoBERTa model trained with train_blazing.py
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*60)
print("Amazon ML Challenge 2025 - BLAZING Inference")
print("="*60)
print(f"Using device: {device}")

# ==================== FEATURES ====================

def extract_enhanced_features(text):
    features = {
        'value': 0.0, 'pack_size': 1, 'total_quantity': 0.0,
        'text_length': 0, 'has_value': 0, 'brand': 'unknown',
        'category': 'other', 'word_count': 0, 'has_numbers': 0,
        'has_capitals': 0, 'log_quantity': 0.0, 'value_per_unit': 0.0,
        'is_pack': 0, 'bullet_points': 0, 'quantity_category': 0,
        'has_description': 0, 'num_sentences': 0, 'avg_word_length': 0.0,
        'price_keywords': 0, 'unit_type': 'unknown'
    }
    
    if pd.isna(text):
        return features
    
    text = str(text)
    features['text_length'] = len(text)
    features['bullet_points'] = text.count('Bullet Point')
    features['word_count'] = len(text.split())
    features['has_numbers'] = 1 if any(c.isdigit() for c in text) else 0
    features['has_capitals'] = 1 if any(c.isupper() for c in text) else 0
    features['has_description'] = 1 if 'Product Description' in text else 0
    
    features['num_sentences'] = len([s for s in text.split('.') if len(s.strip()) > 10])
    
    words = text.split()
    if words:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
    
    price_kw = ['pack', 'bundle', 'set', 'count', 'oz', 'lb', 'kg', 'ml', 'liter']
    features['price_keywords'] = sum(1 for kw in price_kw if kw in text.lower())
    
    value_match = re.search(r'Value:\s*([\d.]+)', text, re.IGNORECASE)
    if value_match:
        features['value'] = float(value_match.group(1))
        features['has_value'] = 1
    
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
    
    if 'ounce' in text.lower() or 'oz' in text.lower():
        features['unit_type'] = 'ounce'
    elif 'pound' in text.lower() or 'lb' in text.lower():
        features['unit_type'] = 'pound'
    elif 'gram' in text.lower() or 'kg' in text.lower():
        features['unit_type'] = 'gram'
    elif 'liter' in text.lower() or 'ml' in text.lower():
        features['unit_type'] = 'liter'
    
    features['total_quantity'] = features['value'] * features['pack_size']
    features['log_quantity'] = np.log1p(features['total_quantity'])
    features['value_per_unit'] = features['value'] / max(features['pack_size'], 1)
    
    tq = features['total_quantity']
    if tq < 5:
        features['quantity_category'] = 0
    elif tq < 20:
        features['quantity_category'] = 1
    elif tq < 50:
        features['quantity_category'] = 2
    else:
        features['quantity_category'] = 3
    
    text_lower = text.lower()
    brands = ['amazon', 'kraft', 'nestle', 'pepsi', 'coca', 'heinz', 'campbell',
              'kellogg', 'unilever', 'procter', 'colgate', 'dove', 'tide']
    for brand in brands:
        if brand in text_lower:
            features['brand'] = brand
            break
    else:
        words = text.split()
        for word in words[:3]:
            if len(word) > 2 and word[0].isupper():
                features['brand'] = word.lower()
                break
    
    categories = {
        'food': ['food', 'snack', 'sauce', 'pasta', 'cereal', 'cookie'],
        'beverage': ['coffee', 'tea', 'juice', 'drink', 'water', 'soda'],
        'health': ['vitamin', 'supplement', 'protein', 'health'],
        'beauty': ['shampoo', 'soap', 'lotion', 'cream'],
        'household': ['cleaner', 'detergent', 'paper', 'towel'],
        'baby': ['baby', 'diaper', 'formula', 'wipes'],
        'pet': ['pet', 'dog', 'cat', 'animal']
    }
    for cat, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            features['category'] = cat
            break
    
    return features

# ==================== DATASET ====================

class TestDataset(Dataset):
    def __init__(self, df, tokenizer, brand_enc, category_enc, unit_enc, scaler, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.brand_enc = brand_enc
        self.category_enc = category_enc
        self.unit_enc = unit_enc
        self.scaler = scaler
        self.max_length = max_length
        
        print(f"Extracting features for {len(df)} test samples...")
        self.features_list = []
        for idx in tqdm(range(len(df)), desc="Features"):
            features = extract_enhanced_features(df.iloc[idx]['catalog_content'])
            self.features_list.append(features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat = self.features_list[idx]
        
        text = str(row['catalog_content']) if not pd.isna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        try:
            brand_encoded = self.brand_enc.transform([feat['brand']])[0]
        except:
            brand_encoded = 0
        
        try:
            cat_encoded = self.category_enc.transform([feat['category']])[0]
        except:
            cat_encoded = 0
        
        try:
            unit_encoded = self.unit_enc.transform([feat['unit_type']])[0]
        except:
            unit_encoded = 0
        
        numeric = np.array([
            feat['value'], feat['pack_size'], feat['total_quantity'],
            feat['text_length'], feat['has_value'], brand_encoded,
            cat_encoded, feat['word_count'], feat['has_numbers'],
            feat['has_capitals'], feat['log_quantity'], feat['value_per_unit'],
            feat['is_pack'], feat['bullet_points'], feat['quantity_category'],
            feat['has_description'], feat['num_sentences'], feat['avg_word_length'],
            feat['price_keywords'], unit_encoded
        ], dtype=np.float32)
        
        numeric = self.scaler.transform(numeric.reshape(1, -1))[0]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(numeric, dtype=torch.float32),
            'sample_id': row['sample_id']
        }

# ==================== MODEL ====================

class BlazingPredictor(nn.Module):
    def __init__(self, text_model_name='distilroberta-base', numeric_dim=20, hidden_dim=384):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim // 2, 1)
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

# ==================== MAIN ====================

def main():
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    
    print("Loading scaler and encoders...")
    try:
        with open('models/scaler_blazing.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/brand_encoder_blazing.pkl', 'rb') as f:
            brand_enc = pickle.load(f)
        with open('models/category_encoder_blazing.pkl', 'rb') as f:
            category_enc = pickle.load(f)
        with open('models/unit_encoder_blazing.pkl', 'rb') as f:
            unit_enc = pickle.load(f)
        print("✓ Loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("Loading model...")
    try:
        model = BlazingPredictor(
            text_model_name='distilroberta-base',
            numeric_dim=20,
            hidden_dim=384
        ).to(device)
        
        checkpoint = torch.load('models/best_model_blazing.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✓ Model loaded")
        
        val_smape = checkpoint.get('val_smape', None)
        if val_smape:
            print(f"  Validation SMAPE: {val_smape:.2f}%")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("="*60)
    
    print("\nLoading test data...")
    test_df = pd.read_csv('student_resource/dataset/test.csv')
    print(f"Test samples: {len(test_df)}")
    
    print("\nCreating test dataset...")
    test_dataset = TestDataset(
        test_df, tokenizer, brand_enc, category_enc, unit_enc, scaler, max_length=256
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
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
    
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
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

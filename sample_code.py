"""
Amazon ML Challenge 2025 - Multimodal Price Prediction Inference
Enhanced predictor using trained multimodal model
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

# Global variables to store loaded models
TOKENIZER = None
MODEL = None
SCALER = None
DEVICE = None

def initialize_models():
    """Initialize models once at startup"""
    global TOKENIZER, MODEL, SCALER, DEVICE
    
    if MODEL is not None:
        return  # Already initialized
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load scaler
    scaler_path = 'models/feature_scaler.pkl'
    if os.path.exists(scaler_path):
        print("Loading feature scaler...")
        with open(scaler_path, 'rb') as f:
            SCALER = pickle.load(f)
    else:
        print("⚠ Warning: Feature scaler not found. Using dummy scaler.")
        SCALER = None
    
    # Load model
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

# ==================== MODEL ARCHITECTURE (same as training) ====================

class MultimodalPricePredictor(nn.Module):
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 numeric_dim=8, hidden_dim=256):
        super(MultimodalPricePredictor, self).__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        # Feature fusion
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
        # Text features
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Concatenate features
        combined = torch.cat([text_features, numeric_features], dim=1)
        
        # Predict price
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
        'unit_type': 0  # 0=other, 1=weight(oz,lb), 2=volume(fl oz), 3=count
    }
    
    if pd.isna(catalog_content) or not catalog_content:
        return features
    
    text = str(catalog_content)
    features['text_length'] = len(text)
    features['bullet_points'] = text.count('Bullet Point')
    
    # Extract value
    value_match = re.search(r'Value:\s*([\d.]+)', text, re.IGNORECASE)
    if value_match:
        features['value'] = float(value_match.group(1))
        features['has_value'] = 1
    
    # Extract unit type
    unit_match = re.search(r'Unit:\s*(\w+(?:\s+\w+)?)', text, re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1).lower()
        if 'oz' in unit or 'ounce' in unit or 'lb' in unit or 'pound' in unit:
            if 'fl' in unit or 'fluid' in unit:
                features['unit_type'] = 2  # volume
            else:
                features['unit_type'] = 1  # weight
        elif 'count' in unit or 'piece' in unit:
            features['unit_type'] = 3
    
    # Extract pack size
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
    
    # Calculate total quantity
    features['total_quantity'] = features['value'] * features['pack_size']
    
    return features

# ==================== PREDICTOR FUNCTION ====================

def predictor(sample_id, catalog_content, image_link):
    '''
    Call your model/approach here
    
    Parameters:
    - sample_id: Unique identifier for the sample
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    
    # Fallback to random if model not loaded
    if MODEL is None or TOKENIZER is None:
        return round(random.uniform(5.0, 500.0), 2)
    
    try:
        # Extract text features
        catalog_text = str(catalog_content) if not pd.isna(catalog_content) else ""
        
        # Tokenize text
        encoding = TOKENIZER(
            catalog_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        # Extract numeric features
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
        
        # Scale numeric features
        if SCALER is not None:
            numeric_features = SCALER.transform(numeric_features)
        
        numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            MODEL.eval()
            prediction = MODEL(input_ids, attention_mask, numeric_tensor)
            price = prediction.item()
        
        # Ensure positive price
        price = max(0.01, price)
        
        return round(price, 2)
    
    except Exception as e:
        print(f"Error predicting for sample {sample_id}: {str(e)}")
        # Fallback to simple heuristic
        try:
            features = extract_numeric_features(catalog_content)
            if features['total_quantity'] > 0:
                # Simple heuristic: ~$0.10-0.30 per unit
                base_price = features['total_quantity'] * 0.15
                return round(max(1.0, base_price), 2)
        except:
            pass
        return round(random.uniform(5.0, 500.0), 2)

if __name__ == "__main__":
    DATASET_FOLDER = 'student_resource/dataset/'
    
    # Initialize models once at startup
    print("="*60)
    print("Amazon ML Challenge 2025 - Price Prediction Inference")
    print("="*60)
    initialize_models()
    print("="*60)
    
    # Read test data
    print("\nLoading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Test samples: {len(test)}")
    
    # Apply predictor function to each row
    print("\nGenerating predictions...")
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']), 
        axis=1
    )
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    
    # Save predictions
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n✓ Predictions saved to {output_filename}")
    print(f"✓ Total predictions: {len(output_df)}")
    print(f"\nSample predictions:")
    print(output_df.head(10))
    print(f"\nPrice statistics:")
    print(f"  Mean: ${output_df['price'].mean():.2f}")
    print(f"  Median: ${output_df['price'].median():.2f}")
    print(f"  Min: ${output_df['price'].min():.2f}")
    print(f"  Max: ${output_df['price'].max():.2f}")
    print("\n" + "="*60)

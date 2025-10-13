"""
Amazon ML Challenge 2025 - FAST Model Training Script
Model: RoBERTa-base (125M params, 2x faster than DeBERTa)
Expected: 40-45% SMAPE in 8-10 hours on T4, 4-5 hours on P100
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable optimizations
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✓ CUDA optimizations enabled")

# ==================== FEATURE EXTRACTION ====================

def extract_brand(text):
    """Extract brand name"""
    if pd.isna(text) or not text:
        return "unknown"
    text_str = str(text).lower()
    brands = ['amazon', 'kraft', 'nestle', 'pepsi', 'coca', 'heinz', 'campbell',
              'kellogg', 'unilever', 'procter', 'colgate', 'dove', 'tide']
    for brand in brands:
        if brand in text_str:
            return brand
    words = str(text).split()
    for word in words[:3]:
        if len(word) > 2 and word[0].isupper():
            return word.lower()
    return "unknown"

def classify_category(text):
    """Classify into categories"""
    if pd.isna(text) or not text:
        return "other"
    text = str(text).lower()
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
        if any(kw in text for kw in keywords):
            return cat
    return "other"

def extract_features(catalog_content):
    """Extract 15 core features (simplified for speed)"""
    features = {
        'value': 0.0, 'pack_size': 1, 'total_quantity': 0.0,
        'text_length': 0, 'has_value': 0, 'brand': 'unknown',
        'category': 'other', 'word_count': 0, 'has_numbers': 0,
        'has_capitals': 0, 'log_quantity': 0.0, 'value_per_unit': 0.0,
        'is_pack': 0, 'bullet_points': 0, 'quantity_category': 0
    }
    
    if pd.isna(catalog_content):
        return features
    
    text = str(catalog_content)
    features['text_length'] = len(text)
    features['bullet_points'] = text.count('Bullet Point')
    features['word_count'] = len(text.split())
    features['has_numbers'] = 1 if any(c.isdigit() for c in text) else 0
    features['has_capitals'] = 1 if any(c.isupper() for c in text) else 0
    
    # Extract value
    value_match = re.search(r'Value:\s*([\d.]+)', text, re.IGNORECASE)
    if value_match:
        features['value'] = float(value_match.group(1))
        features['has_value'] = 1
    
    # Extract pack size
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
    
    features['total_quantity'] = features['value'] * features['pack_size']
    features['log_quantity'] = np.log1p(features['total_quantity'])
    features['value_per_unit'] = features['value'] / max(features['pack_size'], 1)
    
    # Quantity category
    tq = features['total_quantity']
    if tq < 5:
        features['quantity_category'] = 0
    elif tq < 20:
        features['quantity_category'] = 1
    elif tq < 50:
        features['quantity_category'] = 2
    else:
        features['quantity_category'] = 3
    
    features['brand'] = extract_brand(text)
    features['category'] = classify_category(text)
    
    return features

# ==================== DATASET ====================

class FastDataset(Dataset):
    def __init__(self, df, tokenizer, brand_encoder=None, category_encoder=None, 
                 max_length=256, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.brand_encoder = brand_encoder
        self.category_encoder = category_encoder
        
        # Extract features
        print(f"Extracting features for {len(df)} samples...")
        self.features_list = []
        for idx in tqdm(range(len(df)), desc="Feature extraction"):
            features = extract_features(df.iloc[idx]['catalog_content'])
            # Create numeric array (15 features)
            numeric = np.array([
                features['value'],
                features['pack_size'],
                features['total_quantity'],
                features['text_length'],
                features['has_value'],
                0,  # brand_encoded
                0,  # category_encoded
                features['word_count'],
                features['has_numbers'],
                features['has_capitals'],
                features['log_quantity'],
                features['value_per_unit'],
                features['is_pack'],
                features['bullet_points'],
                features['quantity_category']
            ], dtype=np.float32)
            features['numeric'] = numeric
            self.features_list.append(features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = self.features_list[idx]
        
        # Tokenize
        text = str(row['catalog_content']) if not pd.isna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode brand/category
        brand_encoded = self.brand_encoder.transform([features['brand']])[0] if self.brand_encoder else 0
        cat_encoded = self.category_encoder.transform([features['category']])[0] if self.category_encoder else 0
        
        # Get numeric features (use scaled if available)
        if '_scaled' in features:
            numeric = features['_scaled']
        else:
            numeric = np.copy(features['numeric'])
            numeric[5] = float(brand_encoded)
            numeric[6] = float(cat_encoded)
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(numeric, dtype=torch.float32)
        }
        
        if self.is_train:
            item['price'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return item

# ==================== FAST MODEL ====================

class FastPricePredictor(nn.Module):
    def __init__(self, text_model_name='roberta-base', numeric_dim=15, hidden_dim=512):
        super().__init__()
        
        # RoBERTa encoder (faster than DeBERTa)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Simpler fusion network for speed
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(hidden_dim // 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1)
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

# ==================== LOSS & METRICS ====================

def smape_loss(predictions, targets, epsilon=1e-8):
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0 + epsilon
    smape = torch.abs(predictions - targets) / denominator
    return torch.mean(smape) * 100

def smape_metric(y_true, y_pred, epsilon=1e-8):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    smape = np.abs(y_pred - y_true) / denominator
    return np.mean(smape) * 100

# ==================== TRAINING ====================

def train_fast_model(model, train_loader, val_loader, epochs=15, lr=2e-4, 
                     gradient_accumulation_steps=4, resume_from=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = smape_loss
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler()
    best_val_smape = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_smape = checkpoint.get('best_val_smape', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}, best SMAPE: {best_val_smape:.2f}%\n")
    
    print(f"\n⚡ FAST MODEL OPTIMIZATIONS:")
    print(f"  - Model: RoBERTa-base (125M params, 2x faster)")
    print(f"  - Features: 15 (vs 25 in ULTRA)")
    print(f"  - Sequence length: 256 (vs 384)")
    print(f"  - Gradient accumulation: {gradient_accumulation_steps}x")
    print(f"  - Epochs: {epochs} (starting from {start_epoch + 1})")
    print(f"  - Checkpoint saving: Every epoch")
    print(f"  - Expected: 40-45% SMAPE in 4-5 hours (P100)")
    print(f"  - Expected: 40-45% SMAPE in 8-10 hours (T4)\n")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            numeric_features = batch['numeric_features'].to(device, non_blocking=True)
            labels = batch['price'].to(device, non_blocking=True)
            
            with autocast():
                outputs = model(input_ids, attention_mask, numeric_features)
                loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * gradient_accumulation_steps
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numeric_features = batch['numeric_features'].to(device)
                labels = batch['price'].to(device)
                
                outputs = model(input_ids, attention_mask, numeric_features)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_smape = smape_metric(np.array(val_targets), np.array(val_predictions))
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val SMAPE: {val_smape:.2f}%')
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_smape': val_smape,
            'train_loss': avg_train_loss,
            'best_val_smape': best_val_smape
        }
        
        # Save latest checkpoint (overwrites each epoch)
        torch.save(checkpoint, 'models/checkpoint_fast_latest.pth', _use_new_zipfile_serialization=False)
        
        # Save epoch-specific checkpoint
        torch.save(checkpoint, f'models/checkpoint_fast_epoch_{epoch+1}.pth', _use_new_zipfile_serialization=False)
        print(f'💾 Checkpoint saved: epoch {epoch+1}')
        
        # Save best model
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            torch.save(checkpoint, 'models/best_model_fast.pth', _use_new_zipfile_serialization=False)
            print(f'⭐ Best model saved with Val SMAPE: {val_smape:.2f}%')
    
    return model

# ==================== MAIN ====================

def main():
    print("="*70)
    print("Amazon ML Challenge 2025 - FAST Model Training")
    print("RoBERTa-base: 125M params, 2x faster than DeBERTa")
    print("="*70)
    
    # Tokenizer
    print("\nLoading RoBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('student_resource/dataset/train.csv')
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Extract features and fit encoders
    print("\nExtracting features for encoder fitting...")
    train_features = [extract_features(row['catalog_content']) 
                     for _, row in tqdm(train_data.iterrows(), total=len(train_data))]
    
    brand_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    brand_encoder.fit([f['brand'] for f in train_features])
    category_encoder.fit([f['category'] for f in train_features])
    
    print(f"Brands: {len(brand_encoder.classes_)}, Categories: {len(category_encoder.classes_)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FastDataset(train_data, tokenizer, brand_encoder, category_encoder, 
                                max_length=256, is_train=True)
    val_dataset = FastDataset(val_data, tokenizer, brand_encoder, category_encoder,
                              max_length=256, is_train=True)
    
    # Optimized dataloaders (num_workers=0 for Kaggle compatibility)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                           num_workers=0, pin_memory=True)
    
    # Fit scaler
    print("\nFitting scaler...")
    all_features = []
    for i in range(len(train_dataset.features_list)):
        feat = train_dataset.features_list[i]
        numeric = np.copy(feat['numeric'])
        numeric[5] = brand_encoder.transform([feat['brand']])[0]
        numeric[6] = category_encoder.transform([feat['category']])[0]
        all_features.append(numeric)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Save scaler and encoders
    os.makedirs('models', exist_ok=True)
    with open('models/scaler_fast.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/brand_encoder_fast.pkl', 'wb') as f:
        pickle.dump(brand_encoder, f)
    with open('models/category_encoder_fast.pkl', 'wb') as f:
        pickle.dump(category_encoder, f)
    print("✓ Scaler and encoders saved")
    
    # Apply scaling
    print("Applying scaling...")
    for i in range(len(train_dataset.features_list)):
        feat = train_dataset.features_list[i]
        numeric = np.copy(feat['numeric'])
        numeric[5] = brand_encoder.transform([feat['brand']])[0]
        numeric[6] = category_encoder.transform([feat['category']])[0]
        scaled = scaler.transform(numeric.reshape(1, -1))[0]
        train_dataset.features_list[i]['_scaled'] = scaled
    
    for i in range(len(val_dataset.features_list)):
        feat = val_dataset.features_list[i]
        numeric = np.copy(feat['numeric'])
        numeric[5] = brand_encoder.transform([feat['brand']])[0]
        numeric[6] = category_encoder.transform([feat['category']])[0]
        scaled = scaler.transform(numeric.reshape(1, -1))[0]
        val_dataset.features_list[i]['_scaled'] = scaled
    
    # Initialize model
    print("\nInitializing FAST model (RoBERTa-base)...")
    model = FastPricePredictor(
        text_model_name='roberta-base',
        numeric_dim=15,
        hidden_dim=512
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n" + "="*70)
    print("Starting FAST training...")
    print("="*70)
    
    model = train_fast_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        lr=2e-4,
        gradient_accumulation_steps=4
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print("✓ Model saved to: models/best_model_fast.pth")
    print("✓ Scaler saved to: models/scaler_fast.pkl")
    print("✓ Encoders saved to: models/*_encoder_fast.pkl")
    print("\nUse sample_code_fast.py for inference")
    print("Expected SMAPE: 40-45% (good balance of speed/accuracy)")

if __name__ == "__main__":
    main()

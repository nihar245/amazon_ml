"""
Amazon ML Challenge 2025 - OPTIMIZED Model Training
Target: SMAPE 30-40% on test set, ultra-fast training on Kaggle GPU
Model: DeBERTa-v3-small (optimized for speed and accuracy)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# EXTREME optimizations for Kaggle GPU
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print("✓ CUDA extreme optimizations enabled")

# ==================== ADVANCED FEATURE EXTRACTION ====================

def extract_optimized_features(text):
    """Extract 22 highly predictive features for better accuracy"""
    features = {
        'value': 0.0, 'pack_size': 1, 'total_quantity': 0.0,
        'text_length': 0, 'has_value': 0, 'brand': 'unknown',
        'category': 'other', 'word_count': 0, 'has_numbers': 0,
        'has_capitals': 0, 'log_quantity': 0.0, 'value_per_unit': 0.0,
        'is_pack': 0, 'bullet_points': 0, 'quantity_category': 0,
        'has_description': 0, 'num_sentences': 0, 'avg_word_length': 0.0,
        'price_keywords': 0, 'unit_type': 'unknown',
        'vocab_richness': 0.0, 'bulk_indicators': 0
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
    
    # Sentences
    features['num_sentences'] = len([s for s in text.split('.') if len(s.strip()) > 10])
    
    # Avg word length
    words = text.split()
    if words:
        features['avg_word_length'] = sum(len(w) for w in words) / len(words)
    
    # Vocabulary richness (unique words ratio)
    if words:
        features['vocab_richness'] = len(set(words)) / len(words)
    
    # Price keywords and bulk indicators
    price_kw = ['pack', 'bundle', 'set', 'count', 'oz', 'lb', 'kg', 'ml', 'liter']
    bulk_kw = ['bulk', 'family', 'economy', 'large', 'value pack']
    features['price_keywords'] = sum(1 for kw in price_kw if kw in text.lower())
    features['bulk_indicators'] = sum(1 for kw in bulk_kw if kw in text.lower())
    
    # Extract value
    value_match = re.search(r'Value:\s*([\d.]+)', text, re.IGNORECASE)
    if value_match:
        features['value'] = float(value_match.group(1))
        features['has_value'] = 1
    
    # Pack size
    pack_match = re.search(r'Pack\s+of\s+(\d+)', text, re.IGNORECASE)
    if pack_match:
        features['pack_size'] = int(pack_match.group(1))
        features['is_pack'] = 1
    
    # Unit type
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
    
    # Brand
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
    
    # Category
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

# ==================== OPTIMIZED DATASET ====================

class OptimizedDataset(Dataset):
    def __init__(self, df, tokenizer, brand_enc, category_enc, unit_enc, 
                 max_length=256, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.brand_enc = brand_enc
        self.category_enc = category_enc
        self.unit_enc = unit_enc
        
        print(f"Extracting features for {len(df)} samples...")
        self.features_list = []
        for idx in tqdm(range(len(df)), desc="Features"):
            feat = extract_optimized_features(df.iloc[idx]['catalog_content'])
            self.features_list.append(feat)
    
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
        
        # Encode categoricals
        brand_encoded = self.brand_enc.transform([feat['brand']])[0] if self.brand_enc else 0
        cat_encoded = self.category_enc.transform([feat['category']])[0] if self.category_enc else 0
        unit_encoded = self.unit_enc.transform([feat['unit_type']])[0] if self.unit_enc else 0
        
        # Get scaled features
        if '_scaled' in feat:
            numeric = feat['_scaled']
        else:
            numeric = np.array([
                feat['value'], feat['pack_size'], feat['total_quantity'],
                feat['text_length'], feat['has_value'], brand_encoded,
                cat_encoded, feat['word_count'], feat['has_numbers'],
                feat['has_capitals'], feat['log_quantity'], feat['value_per_unit'],
                feat['is_pack'], feat['bullet_points'], feat['quantity_category'],
                feat['has_description'], feat['num_sentences'], feat['avg_word_length'],
                feat['price_keywords'], unit_encoded, feat['vocab_richness'],
                feat['bulk_indicators']
            ], dtype=np.float32)
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(numeric, dtype=torch.float32)
        }
        
        if self.is_train:
            item['price'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return item

# ==================== OPTIMIZED MODEL ====================

class OptimizedPredictor(nn.Module):
    def __init__(self, text_model_name='microsoft/deberta-v3-small', numeric_dim=22, hidden_dim=384):
        super().__init__()
        
        # DeBERTa-v3-small: 44M params, much faster than base, still accurate
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 768 for small
        
        # Streamlined fusion for speed with enough depth for accuracy
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 4, 1)
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

# ==================== CUSTOM LOSS FOR ACCURACY ====================

def smape_loss(predictions, targets, epsilon=1e-8):
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0 + epsilon
    smape = torch.abs(predictions - targets) / denominator
    return torch.mean(smape) * 100

def smape_metric(y_true, y_pred, epsilon=1e-8):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    smape = np.abs(y_pred - y_true) / denominator
    return np.mean(smape) * 100

# Custom focal SMAPE loss to focus on hard examples
def focal_smape_loss(predictions, targets, epsilon=1e-8, gamma=2.0):
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0 + epsilon
    smape = torch.abs(predictions - targets) / denominator
    # Focal factor: higher weight to harder examples
    focal_factor = torch.pow(smape, gamma)
    return torch.mean(smape * focal_factor) * 100

# ==================== OPTIMIZED TRAINING ====================

def train_optimized(model, train_loader, val_loader, epochs=25, lr=3e-5, grad_accum=8, resume_from=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    criterion = focal_smape_loss  # Use focal loss for better accuracy
    
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = len(train_loader) * 2
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
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
    
    print(f"\n⚡ OPTIMIZED MODEL CONFIG:")
    print(f"  - Model: DeBERTa-v3-small (44M params, 3x faster than base)")
    print(f"  - Features: 22 (highly predictive)")
    print(f"  - Sequence length: 256 (optimized)")
    print(f"  - Batch: 16 x {grad_accum} = {16*grad_accum} effective")
    print(f"  - Epochs: {epochs} (starting from {start_epoch + 1})")
    print(f"  - Checkpoint saving: Every epoch")
    print(f"  - Learning rate: {lr} (fine-tuned)")
    print(f"  - Custom loss: Focal SMAPE (focus on hard examples)")
    print(f"  - Expected: SMAPE 30-40% in 3-4 hours (P100)\n")
    
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
                loss = criterion(outputs, labels) / grad_accum
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * grad_accum
        
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
        torch.save(checkpoint, 'models/checkpoint_optimized_latest.pth', _use_new_zipfile_serialization=False)
        
        # Save epoch-specific checkpoint
        torch.save(checkpoint, f'models/checkpoint_optimized_epoch_{epoch+1}.pth', _use_new_zipfile_serialization=False)
        print(f'💾 Checkpoint saved: epoch {epoch+1}')
        
        # Save best model
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            torch.save(checkpoint, 'models/best_model_optimized.pth', _use_new_zipfile_serialization=False)
            print(f'⭐ Best model saved with Val SMAPE: {val_smape:.2f}%')
    
    return model

# ==================== MAIN ====================

def main():
    print("="*70)
    print("Amazon ML Challenge 2025 - OPTIMIZED Model Training")
    print("DeBERTa-v3-small: 44M params, targeting SMAPE 30-40%")
    print("="*70)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('student_resource/dataset/train.csv')
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Extract features
    print("\nExtracting features...")
    train_features = [extract_optimized_features(row['catalog_content']) 
                     for _, row in tqdm(train_data.iterrows(), total=len(train_data))]
    
    # Fit encoders
    brand_enc = LabelEncoder()
    category_enc = LabelEncoder()
    unit_enc = LabelEncoder()
    brand_enc.fit([f['brand'] for f in train_features])
    category_enc.fit([f['category'] for f in train_features])
    unit_enc.fit([f['unit_type'] for f in train_features])
    
    print(f"Brands: {len(brand_enc.classes_)}, Categories: {len(category_enc.classes_)}, Units: {len(unit_enc.classes_)}")
    
    # Datasets
    print("\nCreating datasets...")
    train_dataset = OptimizedDataset(train_data, tokenizer, brand_enc, category_enc, unit_enc, 256, True)
    val_dataset = OptimizedDataset(val_data, tokenizer, brand_enc, category_enc, unit_enc, 256, True)
    
    # Dataloaders (aggressive batch size for speed)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # Fit scaler
    print("\nFitting scaler...")
    all_features = []
    for feat in train_dataset.features_list:
        numeric = np.array([
            feat['value'], feat['pack_size'], feat['total_quantity'],
            feat['text_length'], feat['has_value'], 
            brand_enc.transform([feat['brand']])[0],
            category_enc.transform([feat['category']])[0],
            feat['word_count'], feat['has_numbers'], feat['has_capitals'],
            feat['log_quantity'], feat['value_per_unit'], feat['is_pack'],
            feat['bullet_points'], feat['quantity_category'], feat['has_description'],
            feat['num_sentences'], feat['avg_word_length'], feat['price_keywords'],
            unit_enc.transform([feat['unit_type']])[0], feat['vocab_richness'],
            feat['bulk_indicators']
        ])
        all_features.append(numeric)
    
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Save
    os.makedirs('models', exist_ok=True)
    with open('models/scaler_optimized.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/brand_encoder_optimized.pkl', 'wb') as f:
        pickle.dump(brand_enc, f)
    with open('models/category_encoder_optimized.pkl', 'wb') as f:
        pickle.dump(category_enc, f)
    with open('models/unit_encoder_optimized.pkl', 'wb') as f:
        pickle.dump(unit_enc, f)
    print("✓ Scaler and encoders saved")
    
    # Apply scaling
    print("Scaling...")
    for feat in train_dataset.features_list:
        numeric = np.array([
            feat['value'], feat['pack_size'], feat['total_quantity'],
            feat['text_length'], feat['has_value'],
            brand_enc.transform([feat['brand']])[0],
            category_enc.transform([feat['category']])[0],
            feat['word_count'], feat['has_numbers'], feat['has_capitals'],
            feat['log_quantity'], feat['value_per_unit'], feat['is_pack'],
            feat['bullet_points'], feat['quantity_category'], feat['has_description'],
            feat['num_sentences'], feat['avg_word_length'], feat['price_keywords'],
            unit_enc.transform([feat['unit_type']])[0], feat['vocab_richness'],
            feat['bulk_indicators']
        ])
        feat['_scaled'] = scaler.transform(numeric.reshape(1, -1))[0]
    
    for feat in val_dataset.features_list:
        numeric = np.array([
            feat['value'], feat['pack_size'], feat['total_quantity'],
            feat['text_length'], feat['has_value'],
            brand_enc.transform([feat['brand']])[0],
            category_enc.transform([feat['category']])[0],
            feat['word_count'], feat['has_numbers'], feat['has_capitals'],
            feat['log_quantity'], feat['value_per_unit'], feat['is_pack'],
            feat['bullet_points'], feat['quantity_category'], feat['has_description'],
            feat['num_sentences'], feat['avg_word_length'], feat['price_keywords'],
            unit_enc.transform([feat['unit_type']])[0], feat['vocab_richness'],
            feat['bulk_indicators']
        ])
        feat['_scaled'] = scaler.transform(numeric.reshape(1, -1))[0]
    
    # Model
    print("\nInitializing OPTIMIZED model...")
    model = OptimizedPredictor(
        text_model_name='microsoft/deberta-v3-small',
        numeric_dim=22,
        hidden_dim=384
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n" + "="*70)
    print("Starting OPTIMIZED training...")
    print("="*70)
    
    model = train_optimized(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=25,
        lr=3e-5,
        grad_accum=8
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print("✓ Model: models/best_model_optimized.pth")
    print("✓ Use sample_code_optimized.py for inference")
    print("Expected SMAPE: 30-40% in just 3-4 hours on P100!")

if __name__ == "__main__":
    main()

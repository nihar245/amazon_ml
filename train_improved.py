"""
Amazon ML Challenge 2025 - IMPROVED Multimodal Price Prediction Training Script
Fixed version: avoids mutating stored features in ProductDataset.__getitem__
and precomputes numeric feature arrays for efficiency.
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

# Enable TF32 for faster computation on Ampere GPUs (A100, RTX 3090, etc.)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ TF32 enabled for faster training")

# ==================== ENHANCED FEATURE EXTRACTION ====================

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
        'beverage': ['coffee', 'tea', 'juice', 'drink', 'water', 'soda', 'cola',
                     'beverage', 'lemonade', 'smoothie'],
        'health': ['vitamin', 'supplement', 'medicine', 'health', 'protein',
                   'omega', 'probiotic', 'multivitamin', 'calcium'],
        'beauty': ['shampoo', 'conditioner', 'soap', 'lotion', 'cream', 'cosmetic',
                   'deodorant', 'perfume', 'moisturizer'],
        'household': ['cleaner', 'detergent', 'paper', 'towel', 'bag', 'tissue',
                      'trash', 'sponge', 'dish', 'laundry'],
        'baby': ['baby', 'diaper', 'formula', 'wipes', 'infant'],
        'pet': ['pet', 'dog', 'cat', 'animal', 'puppy', 'kitten']
    }
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                return category
    return "other"

def extract_text_quality_features(text):
    """Extract text quality and structure indicators"""
    if pd.isna(text) or not text:
        return {
            'has_capitals': 0,
            'word_count': 0,
            'comma_count': 0,
            'has_numbers': 0,
            'vocab_richness': 0.0,
            'num_sentences': 0,
            'avg_word_length': 0.0,
            'has_description': 0,
            'price_keywords': 0,
            'bulk_indicators': 0
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
        'value': 0.0,
        'pack_size': 1,
        'total_quantity': 0.0,
        'is_pack': 0,
        'text_length': 0,
        'bullet_points': 0,
        'has_value': 0,
        'unit_type': 0,
        'brand': 'unknown',
        'category': 'other',
        'brand_encoded': 0,
        'category_encoded': 0,
        'has_capitals': 0,
        'word_count': 0,
        'comma_count': 0,
        'has_numbers': 0,
        'vocab_richness': 0.0,
        'num_sentences': 0,
        'avg_word_length': 0.0,
        'has_description': 0,
        'price_keywords': 0,
        'bulk_indicators': 0,
        'value_per_unit': 0.0,
        'log_total_quantity': 0.0,
        'is_large_pack': 0,
        'has_fraction': 0,
        'quantity_category': 0
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

class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, brand_encoder=None, category_encoder=None, 
                 max_length=256, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.brand_encoder = brand_encoder
        self.category_encoder = category_encoder
        
        # Pre-extract all features and numeric np arrays
        print(f"Extracting features for {len(df)} samples...")
        self.features_list = []
        for idx in tqdm(range(len(df)), desc="Feature extraction"):
            catalog_content = df.iloc[idx]['catalog_content']
            features = extract_enhanced_features(catalog_content)
            # Precompute the numeric vector to avoid recomputing in __getitem__
            numeric_np = np.array([
                features['value'],
                features['pack_size'],
                features['total_quantity'],
                features['is_pack'],
                features['text_length'],
                features['bullet_points'],
                features['has_value'],
                features['unit_type'],
                0,  # brand_encoded placeholder
                0,  # category_encoded placeholder
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
            # store both the original features (including brand/category) and numeric array
            features['numeric_np'] = numeric_np
            self.features_list.append(features)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Work on a shallow copy so we never mutate stored dicts
        features = dict(self.features_list[idx])
        
        # Tokenize text
        catalog_text = str(row['catalog_content']) if not pd.isna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            catalog_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode brand and category using the encoders (if provided)
        brand_val = features.get('brand', 'unknown')
        cat_val = features.get('category', 'other')
        brand_encoded = self.brand_encoder.transform([brand_val])[0] if self.brand_encoder is not None else 0
        category_encoded = self.category_encoder.transform([cat_val])[0] if self.category_encoder is not None else 0
        
        # If scaler has been applied, the scaled vector is saved in features_list as '_scaled'
        if '_scaled' in self.features_list[idx]:
            numeric_array = np.array(self.features_list[idx]['_scaled'], dtype=np.float32)
        else:
            # use the precomputed numeric_np and insert encoded brand/cat at positions 8 and 9
            numeric_array = np.array(self.features_list[idx]['numeric_np'], dtype=np.float32)
            numeric_array[8] = float(brand_encoded)
            numeric_array[9] = float(category_encoded)
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': torch.tensor(numeric_array, dtype=torch.float32)
        }
        
        if self.is_train:
            item['price'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return item

# ==================== IMPROVED MODEL ARCHITECTURE ====================

class ImprovedPricePredictor(nn.Module):
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 numeric_dim=25, hidden_dim=512):
        super(ImprovedPricePredictor, self).__init__()
        
        # Text encoder (pre-trained transformer)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 384 for MiniLM
        
        # Fine-tune ALL layers for better adaptation (ULTRA mode)
        # All parameters trainable by default
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        
        # Deeper fusion network with BatchNormalization
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
        # Many transformers return last_hidden_state; [CLS] is first token
        text_features = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat([text_features, numeric_features], dim=1)
        price = self.fusion(combined).squeeze(-1)
        return price

# ==================== SMAPE LOSS ====================

def smape_loss(predictions, targets, epsilon=1e-8):
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0 + epsilon
    smape = torch.abs(predictions - targets) / denominator
    return torch.mean(smape) * 100

def smape_metric(y_true, y_pred, epsilon=1e-8):
    """Calculate SMAPE metric for numpy arrays"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    smape = np.abs(y_pred - y_true) / denominator
    return np.mean(smape) * 100

# ==================== OPTIMIZED TRAINING LOOP ====================

def train_model_improved(model, train_loader, val_loader, epochs=20, lr=1e-4, 
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
    
    scaler_amp = torch.cuda.amp.GradScaler()
    best_val_smape = float('inf')
    start_epoch = 0
    
    # Print optimization settings
    print("\n🚀 SPEED OPTIMIZATIONS:")
    print(f"  - Gradient accumulation: {gradient_accumulation_steps}x (effective batch size: {train_loader.batch_size * gradient_accumulation_steps})")
    print(f"  - Mixed precision: Enabled")
    print(f"  - Epochs: {epochs} (starting from {start_epoch+1})")
    print(f"  - Checkpoint saving: Only best model (saves disk space)")
    print(f"  - Expected time per epoch: 15-20 minutes")
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_smape = checkpoint.get('best_val_smape', float('inf'))
        print(f"✓ Resumed from epoch {start_epoch}, best SMAPE: {best_val_smape:.2f}%\n")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()  # Zero grad once at the start
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            numeric_features = batch['numeric_features'].to(device, non_blocking=True)
            labels = batch['price'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask, numeric_features)
                loss = criterion(outputs, labels)
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
            
            scaler_amp.scale(loss).backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler_amp.step(optimizer)
                scaler_amp.update()
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
        
        # Save checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler_amp.state_dict(),
            'val_smape': val_smape,
            'train_loss': avg_train_loss,
            'best_val_smape': best_val_smape
        }
        
        # Save only the latest checkpoint (overwrites each epoch - saves space)
        torch.save(checkpoint, 'models/checkpoint_ultra_latest.pth')
        
        # Save best model and delete old epoch checkpoints to save disk space
        if val_smape < best_val_smape:
            best_val_smape = val_smape
            torch.save(checkpoint, 'models/best_model_ultra.pth')
            print(f'⭐ Best model saved with Val SMAPE: {val_smape:.2f}%')
            
            # Clean up old epoch-specific checkpoints to save space
            import glob
            old_checkpoints = glob.glob('models/checkpoint_ultra_epoch_*.pth')
            for old_cp in old_checkpoints:
                try:
                    os.remove(old_cp)
                except:
                    pass
        else:
            print(f'💾 Checkpoint saved (latest only)')
    
    return model


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("="*60)
    print("Amazon ML Challenge 2025 - ULTRA Model Training")
    print("DeBERTa-v3-base with 25 Engineered Features")
    print("="*60)
    print(f"Using device: {device}")
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage('.')
    free_gb = free / (1024**3)
    print(f"\n💾 Disk Space: {free_gb:.1f} GB available")
    if free_gb < 5:
        print("⚠️  WARNING: Low disk space! Training may fail.")
        print("   Recommendation: Only best model will be saved to save space.")
    
    # Initialize tokenizer
    print("\nLoading DeBERTa-v3-base tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    
    # Load data
    print("\nLoading data...")
    train_data = pd.read_csv('student_resource/dataset/train.csv')
    print(f"Training samples: {len(train_data)}")
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Extract features and fit encoders
    print("\nExtracting features...")
    train_features = [extract_enhanced_features(row.to_dict()['catalog_content']) for _, row in tqdm(train_data.iterrows(), total=len(train_data))]
    
    # Fit encoders on training data
    brands = [f['brand'] for f in train_features]
    categories = [f['category'] for f in train_features]
    
    brand_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    brand_encoder.fit(brands)
    category_encoder.fit(categories)
    
    print(f"Unique brands: {len(brand_encoder.classes_)}")
    print(f"Unique categories: {len(category_encoder.classes_)}")
    
    print("\nCreating datasets...")
    train_dataset = ProductDataset(
        train_data, tokenizer, brand_encoder, category_encoder, 
        max_length=128, is_train=True  # Reduced from 384 for 3x speed boost
    )
    val_dataset = ProductDataset(
        val_data, tokenizer, brand_encoder, category_encoder,
        max_length=128, is_train=True  # Reduced from 384 for 3x speed boost
    )
    
    # Optimized dataloaders with multi-worker prefetching (P100 optimized)
    num_workers = 4 if torch.cuda.is_available() else 0  # Increased from 2
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Increased from 8 (P100 can handle this)
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3 if num_workers > 0 else None,  # Increased prefetch
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,  # Increased from 16 (no gradients = more memory)
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=3 if num_workers > 0 else None,  # Increased prefetch
        persistent_workers=True if num_workers > 0 else False
    )
    print(f"✓ Dataloaders optimized for P100 (batch={16}/{32}, workers={num_workers})")
    
    print("\nFitting feature scaler...")
    all_numeric_features = []
    for i in range(len(train_dataset.features_list)):
        feat = dict(train_dataset.features_list[i])  # shallow copy
        # encode brand/category into numeric positions 8 and 9
        brand_val = feat.get('brand', 'unknown')
        cat_val = feat.get('category', 'other')
        brand_encoded = brand_encoder.transform([brand_val])[0]
        category_encoded = category_encoder.transform([cat_val])[0]
        numeric = np.array(feat['numeric_np'], dtype=np.float32)
        numeric[8] = float(brand_encoded)
        numeric[9] = float(category_encoded)
        all_numeric_features.append(numeric)
    
    scaler = StandardScaler()
    scaler.fit(all_numeric_features)
    
    # Save scaler and encoders
    with open('models/feature_scaler_improved.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/brand_encoder.pkl', 'wb') as f:
        pickle.dump(brand_encoder, f)
    with open('models/category_encoder.pkl', 'wb') as f:
        pickle.dump(category_encoder, f)
    
    print("✓ Scaler and encoders saved")
    
    # Apply scaling: store scaled vector into features_list[i]['_scaled'] (no mutation when used in __getitem__)
    print("\nApplying feature scaling...")
    for i in range(len(train_dataset.features_list)):
        feat = train_dataset.features_list[i]  # original dict reference
        numeric = np.array(feat['numeric_np'], dtype=np.float32)
        # set brand/category encoded values
        brand_encoded = brand_encoder.transform([feat.get('brand', 'unknown')])[0]
        category_encoded = category_encoder.transform([feat.get('category', 'other')])[0]
        numeric[8] = float(brand_encoded)
        numeric[9] = float(category_encoded)
        scaled = scaler.transform(numeric.reshape(1, -1))[0]
        # store scaled vector as new key (this doesn't break anything)
        train_dataset.features_list[i]['_scaled'] = scaled
    
    for i in range(len(val_dataset.features_list)):
        feat = val_dataset.features_list[i]
        numeric = np.array(feat['numeric_np'], dtype=np.float32)
        brand_encoded = brand_encoder.transform([feat.get('brand', 'unknown')])[0]
        category_encoded = category_encoder.transform([feat.get('category', 'other')])[0]
        numeric[8] = float(brand_encoded)
        numeric[9] = float(category_encoded)
        scaled = scaler.transform(numeric.reshape(1, -1))[0]
        val_dataset.features_list[i]['_scaled'] = scaled
    
    print("✓ Feature scaling applied")
    
    print("\nInitializing ULTRA model (DeBERTa-v3-base)...")
    model = ImprovedPricePredictor(
        text_model_name='microsoft/deberta-v3-base',
        numeric_dim=25,
        hidden_dim=768
    ).to(device)
    
    # Enable gradient checkpointing to save memory (allows larger batch sizes)
    if hasattr(model.text_encoder, 'gradient_checkpointing_enable'):
        model.text_encoder.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("Starting training with improved strategy...")
    print("="*70)
    
    model = train_model_improved(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=18,  # Optimized for better convergence
        lr=3e-4,  # Increased learning rate for better SMAPE
        gradient_accumulation_steps=2  # Reduced from 4 (effective batch size: 32)
    )
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)
    print("✓ Best model saved to: models/best_model_ultra.pth")
    print("✓ Feature scaler saved to: models/feature_scaler_improved.pkl")
    print("✓ Brand encoder saved to: models/brand_encoder.pkl")
    print("✓ Category encoder saved to: models/category_encoder.pkl")
    print("\nYou can now use sample_code_improved.py for inference on test data.")
    print("Expected SMAPE: 37-42% (from 47.5% baseline)")

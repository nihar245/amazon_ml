"""
Amazon ML Challenge 2025 - Multimodal Price Prediction Training Script
This script trains a multimodal model using text, image, and numeric features
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import requests
from io import BytesIO
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

def download_image_with_retry(url, max_retries=3, timeout=10):
    """Download image with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            continue
    return None

# ==================== DATASET CLASS ====================

class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Pre-extract numeric features
        print("Extracting numeric features...")
        self.numeric_features = []
        for idx in tqdm(range(len(df))):
            catalog_content = df.iloc[idx]['catalog_content']
            features = extract_numeric_features(catalog_content)
            self.numeric_features.append([
                features['value'],
                features['pack_size'],
                features['total_quantity'],
                features['is_pack'],
                features['text_length'],
                features['bullet_points'],
                features['has_value'],
                features['unit_type']
            ])
        self.numeric_features = np.array(self.numeric_features, dtype=np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text encoding
        catalog_content = str(row['catalog_content']) if not pd.isna(row['catalog_content']) else ""
        encoding = self.tokenizer(
            catalog_content,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Numeric features
        numeric = torch.tensor(self.numeric_features[idx], dtype=torch.float32)
        
        # Image placeholder (will be handled separately for efficiency)
        # In real training, you'd download and process images
        # For now, we'll create a dummy image feature
        image_available = 0 if pd.isna(row['image_link']) else 1
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numeric_features': numeric,
            'image_available': torch.tensor(image_available, dtype=torch.float32),
            'image_link': str(row['image_link']) if not pd.isna(row['image_link']) else ""
        }
        
        if self.is_train:
            result['price'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return result

# ==================== MODEL ARCHITECTURE ====================

class MultimodalPricePredictor(nn.Module):
    def __init__(self, text_model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 numeric_dim=8, hidden_dim=256):
        super(MultimodalPricePredictor, self).__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze most of the text encoder (fine-tune only last 2 layers)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
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

# ==================== TRAINING LOOP ====================

def smape_loss(predictions, targets, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) loss
    Competition evaluation metric - used as loss function
    """
    denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0 + epsilon
    smape = torch.abs(predictions - targets) / denominator
    return torch.mean(smape) * 100  # Return as percentage

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, resume_from=None):
    """Train the multimodal price prediction model"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Using SMAPE as loss (competition metric)
    criterion = smape_loss
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_smape', checkpoint.get('val_smape', float('inf')))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"✓ Resumed from epoch {start_epoch}, best SMAPE: {best_val_loss:.2f}%\n")
    
    print(f"\n🚀 TRAINING CONFIGURATION:")
    print(f"  - Epochs: {epochs} (starting from {start_epoch + 1})")
    print(f"  - Checkpoint saving: Every epoch")
    print(f"  - Early stopping patience: {max_patience}")
    print(f"  - Learning rate: {lr}\n")
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            prices = batch['price'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask, numeric_features)
            loss = criterion(predictions, prices)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'SMAPE': f'{loss.item():.2f}%'})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numeric_features = batch['numeric_features'].to(device)
                prices = batch['price'].to(device)
                
                predictions = model(input_ids, attention_mask, numeric_features)
                loss = criterion(predictions, prices)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train SMAPE: {train_loss:.2f}%")
        print(f"Val SMAPE: {val_loss:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_smape': val_loss,
            'train_smape': train_loss,
            'best_val_smape': best_val_loss,
            'patience_counter': patience_counter
        }
        
        # Save latest checkpoint (overwrites each epoch)
        torch.save(checkpoint, 'models/checkpoint_original_latest.pth')
        
        # Save epoch-specific checkpoint
        torch.save(checkpoint, f'models/checkpoint_original_epoch_{epoch+1}.pth')
        print(f"💾 Checkpoint saved: epoch {epoch+1}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, 'models/best_model.pth')
            print(f"⭐ Best model saved with Val SMAPE: {val_loss:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return model

# ==================== MAIN TRAINING SCRIPT ====================

def main():
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv('student_resource/dataset/train.csv')
    print(f"Training samples: {len(train_df)}")
    
    # Data preprocessing
    print("\nPreprocessing data...")
    # No need for log transform - SMAPE works well with raw prices
    
    # Split data
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=42, shuffle=True
    )
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ProductDataset(train_data, tokenizer, is_train=True)
    val_dataset = ProductDataset(val_data, tokenizer, is_train=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Fit scaler on training numeric features
    print("\nFitting feature scaler...")
    scaler = StandardScaler()
    scaler.fit(train_dataset.numeric_features)
    
    # Save scaler
    with open('models/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Feature scaler saved")
    
    # Initialize model
    print("\nInitializing model...")
    model = MultimodalPricePredictor(
        text_model_name='sentence-transformers/all-MiniLM-L6-v2',
        numeric_dim=8,
        hidden_dim=256
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=15,  # Recommended: 10-15 epochs to avoid overfitting
        lr=2e-4
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"✓ Best model saved to: models/best_model.pth")
    print(f"✓ Feature scaler saved to: models/feature_scaler.pkl")
    print("\nYou can now use sample_code.py for inference on test data.")

if __name__ == "__main__":
    main()

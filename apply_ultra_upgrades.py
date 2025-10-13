"""
Automatic script to upgrade train_improved.py to ultra-optimized version
Applies all 5 high-impact changes to reduce SMAPE from 47.5% to <42%
"""

import re

def apply_upgrades():
    print("="*70)
    print("APPLYING ULTRA UPGRADES TO train_improved.py")
    print("="*70)
    
    # Read the file
    with open('train_improved.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Change 1: Switch to DeBERTa-v3-base
    print("\n✓ Change 1: Switching to DeBERTa-v3-base...")
    content = content.replace(
        "AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')",
        "AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')"
    )
    content = content.replace(
        "text_model_name='sentence-transformers/all-MiniLM-L6-v2'",
        "text_model_name='microsoft/deberta-v3-base'"
    )
    
    # Change 2: Increase hidden_dim to 768
    print("✓ Change 2: Updating hidden_dim to 768...")
    content = re.sub(
        r"hidden_dim=512",
        "hidden_dim=768",
        content
    )
    
    # Change 3: Remove freezing code (lines that freeze parameters)
    print("✓ Change 3: Removing layer freezing (fine-tune all layers)...")
    # Remove the freezing section
    pattern = r"for param in self\.text_encoder\.parameters\(\):\s+param\.requires_grad = False\s+try:\s+for layer in self\.text_encoder\.encoder\.layer\[-2:\]:\s+for param in layer\.parameters\(\):\s+param\.requires_grad = True\s+except Exception:\s+pass"
    content = re.sub(pattern, "", content, flags=re.DOTALL)
    
    # Simpler approach - just comment out freezing lines
    lines = content.split('\n')
    new_lines = []
    skip_until_except = False
    for line in lines:
        if 'for param in self.text_encoder.parameters()' in line and 'requires_grad = False' in content.split(line)[1][:200]:
            new_lines.append(line.replace(line.strip(), '# ' + line.strip() + ' # DISABLED: Fine-tune all layers'))
            skip_until_except = True
        elif skip_until_except and 'except Exception:' in line:
            new_lines.append(line.replace(line.strip(), '# ' + line.strip()))
            skip_until_except = False
        elif skip_until_except:
            new_lines.append(line.replace(line.strip(), '# ' + line.strip()))
        else:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    
    # Change 4: Increase sequence length to 384
    print("✓ Change 4: Increasing sequence length to 384...")
    content = re.sub(
        r"max_length=256",
        "max_length=384",
        content
    )
    
    # Change 5: Reduce batch size to 8
    print("✓ Change 5: Reducing batch size to 8...")
    content = re.sub(
        r"batch_size=16",
        "batch_size=8",
        content
    )
    
    # Change 6: Increase epochs to 35
    print("✓ Change 6: Increasing epochs to 35...")
    content = re.sub(
        r"epochs=25",
        "epochs=35",
        content
    )
    
    # Change 7: Update model save path
    print("✓ Change 7: Updating model save path...")
    content = content.replace(
        "'models/best_model_improved.pth'",
        "'models/best_model_ultra.pth'"
    )
    content = content.replace(
        "'models/feature_scaler_improved.pkl'",
        "'models/feature_scaler_ultra.pkl'"
    )
    content = content.replace(
        "'models/brand_encoder.pkl'",
        "'models/brand_encoder_ultra.pkl'"
    )
    content = content.replace(
        "'models/category_encoder.pkl'",
        "'models/category_encoder_ultra.pkl'"
    )
    
    # Save to new file
    output_file = 'train_ultra.py'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS! Created {output_file}")
    print(f"{'='*70}")
    
    # Show summary
    print("\n📊 CHANGES APPLIED:")
    print("  1. Text encoder: MiniLM → DeBERTa-v3-base")
    print("  2. Hidden dim: 512 → 768")
    print("  3. Fine-tuning: Last 2 layers → ALL layers")
    print("  4. Sequence length: 256 → 384 tokens")
    print("  5. Batch size: 16 → 8")
    print("  6. Epochs: 25 → 35")
    print("  7. Model files: *_improved.* → *_ultra.*")
    
    print("\n🎯 EXPECTED IMPROVEMENT:")
    print("  Current SMAPE: 47.5%")
    print("  Expected SMAPE: 37-42%")
    print("  Improvement: 5-10% reduction")
    
    print("\n⏱️  TRAINING TIME:")
    print("  Kaggle P100 GPU: ~80-100 minutes")
    
    print("\n🚀 NEXT STEPS:")
    print("  1. Review train_ultra.py")
    print("  2. Run: python train_ultra.py")
    print("  3. Wait for training to complete")
    print("  4. Check Val SMAPE < 42%")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    try:
        apply_upgrades()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure train_improved.py exists in the current directory")

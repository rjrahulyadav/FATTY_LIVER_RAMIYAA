"""
Diagnostic script to test model and data pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from src.data_loader import get_data_loaders, get_data_transforms
from models.siamese_net import SiameseNetwork
import numpy as np

def diagnose():
    print("=" * 60)
    print("🔍 MODEL & DATA DIAGNOSTIC")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")
    
    # 1. Check data loading
    print(f"\n📊 Testing Data Loading...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(
            data_dir='data',
            batch_size=32,
            num_workers=0
        )
        print(f"   ✅ Train batches: {len(train_loader)}")
        print(f"   ✅ Val batches: {len(val_loader)}")
        print(f"   ✅ Test batches: {len(test_loader)}")
        
        # Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"   ✅ Batch shape: {images.shape}")
        print(f"   ✅ Labels shape: {labels.shape}")
        print(f"   ✅ Unique labels: {torch.unique(labels).tolist()}")
        print(f"   ✅ Label distribution: {torch.bincount(labels).tolist()}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False
    
    # 2. Check model
    print(f"\n🧠 Testing Model...")
    try:
        model = SiameseNetwork().to(device)
        model.eval()
        print(f"   ✅ Model created successfully")
        
        # Test forward pass
        with torch.no_grad():
            test_images, _ = next(iter(train_loader))
            test_images = test_images.to(device)
            logits = model(test_images)
            probs = torch.softmax(logits, dim=1)
            
            print(f"   ✅ Output shape: {logits.shape}")
            print(f"   ✅ Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"   ✅ Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
            print(f"   ✅ Prob sum: {probs[0].sum():.4f}")
            print(f"   ✅ Sample predictions: {logits[0].tolist()}")
            print(f"   ✅ Sample probabilities: {probs[0].tolist()}")
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False
    
    # 3. Check saved model
    print(f"\n💾 Testing Saved Model...")
    model_path = 'best_model.pth'
    if Path(model_path).exists():
        try:
            state = torch.load(model_path, map_location=device)
            
            # Check if it's a state_dict or wrapped model
            if isinstance(state, dict):
                if 'model' in state:
                    print(f"   ℹ️  State contains 'model' key")
                    state_to_load = state['model']
                else:
                    state_to_load = state
            else:
                state_to_load = state
            
            model_loaded = SiameseNetwork().to(device)
            model_loaded.load_state_dict(state_to_load, strict=False)
            model_loaded.eval()
            print(f"   ✅ Model loaded successfully")
            
            # Test loaded model
            with torch.no_grad():
                test_images, _ = next(iter(train_loader))
                test_images = test_images.to(device)
                logits_loaded = model_loaded(test_images)
                probs_loaded = torch.softmax(logits_loaded, dim=1)
                
                print(f"   ✅ Loaded model output shape: {logits_loaded.shape}")
                print(f"   ✅ Loaded model predictions: {logits_loaded[0].tolist()}")
                print(f"   ✅ Loaded model probabilities: {probs_loaded[0].tolist()}")
                print(f"   ✅ Confidence on sample: {probs_loaded[0].max().item():.4f}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"   ⚠️  Model file not found: {model_path}")
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic complete!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    diagnose()

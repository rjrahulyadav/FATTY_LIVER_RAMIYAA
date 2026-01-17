"""
Optimized training script with proven techniques for medical image classification
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_loaders
import warnings
warnings.filterwarnings('ignore')

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Train')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accuracy
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{total_correct/total_samples:.2%}'})
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            pbar.update(1)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def calculate_class_weights(train_loader, device):
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(5)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Smooth weights to avoid extreme values
    total = class_counts.sum()
    weights = total / (class_counts + 1e-5)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum() * 5  # Normalize
    print(f"📊 Class weights: {weights.tolist()}")
    
    return weights.to(device)

def main():
    """Main training function"""
    
    # Configuration - optimized for medical imaging
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001  # Very conservative initial LR
    NUM_EPOCHS = 200
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Training on {DEVICE}")
    print(f"📊 Batch Size: {BATCH_SIZE}")
    print(f"📈 Learning Rate: {LEARNING_RATE}")
    print(f"⏱️  Epochs: {NUM_EPOCHS}")
    
    # Load data
    print("\n📁 Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir='data',
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    print(f"✅ Test samples: {len(test_loader.dataset)}")
    
    # Calculate class weights
    print("\n📊 Calculating class weights...")
    class_weights = calculate_class_weights(train_loader, DEVICE)
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = SiameseNetwork().to(DEVICE)
    print(f"✅ Model initialized on {DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - step decay
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=0.5
    )
    
    # Training loop
    best_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = 'best_model.pth'
    patience = 30
    patience_counter = 0
    
    print("\n🎯 Starting training...\n")
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step()
        
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ Best model saved! Accuracy: {val_acc:.2%}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Test on test set
    print(f"\n🔍 Testing on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"📈 Best validation accuracy: {best_acc:.2%}")
    print(f"📈 Test accuracy: {test_acc:.2%}")
    print(f"💾 Best model saved to: {best_model_path}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

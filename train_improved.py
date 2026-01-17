"""
Improved training script with better hyperparameters and techniques
to boost model confidence and accuracy
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
        
        # Forward pass - get classification logits
        logits = model(images)
        
        # Classification loss
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

def calculate_class_weights(train_loader):
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(5)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    total = class_counts.sum()
    weights = total / (class_counts * 5)
    return weights

def main():
    """Main training function with improved hyperparameters"""
    
    # Configuration
    BATCH_SIZE = 16  # Smaller batch size for better gradient updates
    LEARNING_RATE = 0.0005  # Lower learning rate for stability
    NUM_EPOCHS = 150  # More epochs for convergence
    WEIGHT_DECAY = 1e-5  # Reduced regularization
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
        num_workers=0  # Set to 0 for Windows compatibility
    )
    print(f"✅ Training samples: {len(train_loader.dataset)}")
    print(f"✅ Validation samples: {len(val_loader.dataset)}")
    
    # Calculate class weights for imbalanced data
    print("\n📊 Calculating class weights...")
    class_weights = calculate_class_weights(train_loader).to(DEVICE)
    print(f"✅ Class weights: {class_weights}")
    
    # Initialize model
    print("\n🧠 Initializing model...")
    model = SiameseNetwork().to(DEVICE)
    print(f"✅ Model initialized on {DEVICE}")
    
    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    
    # Optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler - more conservative
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )
    
    # Training loop
    best_acc = 0.0
    best_model_path = 'best_model_improved.pth'
    patience = 15
    patience_counter = 0
    
    print("\n🎯 Starting training...\n")
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        print(f"\n📊 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ Best model saved! Accuracy: {val_acc:.2%}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Save final model
    print(f"\n✅ Training complete!")
    print(f"📈 Best validation accuracy: {best_acc:.2%}")
    print(f"💾 Best model saved to: {best_model_path}")
    
    # Load and test on validation set
    print(f"\n🔍 Loading best model for final validation...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    final_loss, final_acc = validate(model, val_loader, criterion, DEVICE)
    print(f"✅ Final Validation Accuracy: {final_acc:.2%}")

if __name__ == '__main__':
    main()

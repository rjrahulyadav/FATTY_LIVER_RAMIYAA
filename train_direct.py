"""
Direct training script using the existing dataset
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_loaders, get_contrastive_data_loader
from utils.losses import NTXentLoss

def train_contrastive(model, data_loader, optimizer, criterion, device, epochs):
    """Train the model with contrastive learning"""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(data_loader):
            images, _ = batch
            if images.shape[0] == 0:
                continue
            images = images.to(device)

            # Create positive pairs (same image augmented differently)
            proj1, proj2 = model(images, images)

            # Check for NaNs in projections
            if torch.isnan(proj1).any() or torch.isnan(proj2).any():
                print(f"  Warning: NaN in projections, skipping batch {batch_idx+1}")
                continue

            loss = criterion(proj1, proj2)

            # Check for NaNs before backward
            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"  Warning: NaN/Inf loss detected, skipping batch {batch_idx+1}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}")

        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"✓ Contrastive Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

def train_classification(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_path):
    """Train the model for classification"""
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            images, labels = batch
            if images.shape[0] == 0:
                continue
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"  Warning: NaN/Inf in outputs, skipping batch {batch_idx+1}")
                continue

            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                print(f"  Warning: NaN/Inf loss, skipping batch {batch_idx+1}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            batch_count += 1

        train_acc = 100. * train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                if images.shape[0] == 0:
                    continue
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                val_batch_count += 1

        val_acc = 100. * val_correct / max(val_total, 1)

        print(f"✓ Classification Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/max(batch_count, 1):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss/max(val_batch_count, 1):.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FATTY LIVER CLASSIFICATION - SIAMESE NEURAL NETWORK")
    print("="*60 + "\n")
    
    parser = argparse.ArgumentParser(description='Train Siamese Network for Fatty Liver Classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of classification epochs')
    parser.add_argument('--contrastive_epochs', type=int, default=3, help='Number of contrastive pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}\n")

    # Load data
    try:
        print("Loading data...")
        train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
        contrastive_loader = get_contrastive_data_loader(args.data_dir, args.batch_size)
        print(f"✓ Data loaded successfully")
        print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}\n")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # Create model
    print("Creating model...")
    model = SiameseNetwork().to(device)
    print(f"✓ Model created (ResNet-50 + Siamese architecture)\n")

    # Contrastive pre-training
    if args.contrastive_epochs > 0:
        print(f"STAGE 1: Contrastive Pre-training ({args.contrastive_epochs} epochs)")
        print("-" * 60)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = NTXentLoss(temperature=0.5)
        train_contrastive(model, contrastive_loader, optimizer, criterion, device, args.contrastive_epochs)
        print()

    # Classification training
    print(f"STAGE 2: Classification Training ({args.epochs} epochs)")
    print("-" * 60)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_classification(model, train_loader, val_loader, optimizer, criterion, device, args.epochs, args.save_path)
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Best model saved to: {args.save_path}\n")

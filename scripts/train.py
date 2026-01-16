import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.siamese_net import SiameseNetwork
from src.data_loader import get_data_loaders, get_contrastive_data_loader
from utils.losses import NTXentLoss

def train_contrastive(model, data_loader, optimizer, criterion, device, epochs):
    """Train the model with contrastive learning"""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch in data_loader:
            images, _ = batch
            if images.shape[0] == 0:
                continue
            images = images.to(device)

            # Create positive pairs (same image augmented differently)
            # For simplicity, use same image as positive pair (identity)
            proj1, proj2 = model(images, images)

            # Check for NaNs in projections
            if torch.isnan(proj1).any() or torch.isnan(proj2).any():
                print(f"Warning: NaN in projections at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue

            loss = criterion(proj1, proj2)

            # Check for NaNs before backward
            if torch.isnan(loss) or loss.item() > 1e6:
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, value={loss.item():.6f}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent divergence
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        print(f"Contrastive Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

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

        for batch in train_loader:
            images, labels = batch
            if images.shape[0] == 0:
                continue
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            
            # Check for NaNs in outputs
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN/Inf in model outputs at epoch {epoch+1}, skipping batch")
                optimizer.zero_grad()
                continue

            loss = criterion(outputs, labels)

            # Check for NaNs before backward
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                print(f"Warning: NaN/Inf loss detected at epoch {epoch+1}, value={loss.item():.6f}, skipping batch")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent divergence
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
                
                # Skip batches with NaN/Inf outputs during validation
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

        print(f"Classification Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/max(batch_count, 1):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/max(val_batch_count, 1):.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val acc: {val_acc:.2f}%")
            
            # Validate checkpoint by loading and checking for NaNs
            loaded_state = torch.load(save_path, map_location=device)
            has_nans = False
            for k, v in loaded_state.items():
                if hasattr(v, 'dtype') and torch.isnan(v).any():
                    print(f"Warning: Saved checkpoint contains NaN in {k}")
                    has_nans = True
            if not has_nans:
                print(f"Checkpoint validated: no NaNs detected")

def main():
    parser = argparse.ArgumentParser(description='Train Siamese Network for Fatty Liver Classification')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of classification epochs')
    parser.add_argument('--contrastive_epochs', type=int, default=20, help='Number of contrastive pre-training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    try:
        train_loader, val_loader, _ = get_data_loaders(args.data_dir, args.batch_size)
        contrastive_loader = get_contrastive_data_loader(args.data_dir, args.batch_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure dataset exists at: {args.data_dir}")
        sys.exit(1)

    # Create model
    model = SiameseNetwork().to(device)

    # Contrastive pre-training
    if args.contrastive_epochs > 0:
        print("Starting contrastive pre-training...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = NTXentLoss(temperature=0.5)
        train_contrastive(model, contrastive_loader, optimizer, criterion, device, args.contrastive_epochs)

    # Classification training
    print("Starting classification training...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_classification(model, train_loader, val_loader, optimizer, criterion, device, args.epochs, args.save_path)

    print("Training completed!")
    
    # Final validation of checkpoint
    print("\nFinal checkpoint validation...")
    final_state = torch.load(args.save_path, map_location=device)
    has_nans = False
    nan_count = 0
    for k, v in final_state.items():
        if hasattr(v, 'dtype') and torch.isnan(v).any():
            nan_count += torch.isnan(v).sum().item()
            has_nans = True
    if has_nans:
        print(f"ERROR: Final checkpoint contains {nan_count} NaN values!")
    else:
        print(f"SUCCESS: Final checkpoint is clean (no NaNs detected)")

if __name__ == '__main__':
    main()
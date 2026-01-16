import argparse
import subprocess
import sys
from pathlib import Path
import os

def generate_data():
    """Generate synthetic dataset for testing"""
    try:
        from generate_dataset import generate_synthetic_dataset
        print("Generating synthetic dataset...")
        generate_synthetic_dataset()
        return True
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return False

def download_data():
    """Download dataset from Kaggle"""
    print("Please download the dataset manually from:")
    print("https://www.kaggle.com/code/nirmalgaud/b-mode-fatty-liverultrasound")
    print("Extract the images to the 'data/' directory with subfolders for each class:")
    print("- data/Normal/")
    print("- data/Grade-I/")
    print("- data/Grade-II/")
    print("- data/Grade-III/")
    print("- data/CLD/")

def train_model(args):
    """Train the model"""
    # Check if dataset exists, if not generate synthetic one
    data_path = Path(args.data_dir)
    mat_file = data_path / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
    
    if not mat_file.exists() and not any(data_path.glob('*/')):
        print("Dataset not found. Generating synthetic dataset...")
        if not generate_data():
            print("ERROR: Could not generate dataset")
            return
    
    cmd = [
        sys.executable, 'scripts/train.py',
        '--data_dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--contrastive_epochs', str(args.contrastive_epochs),
        '--save_path', args.save_path
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def evaluate_model(args):
    """Evaluate the model"""
    cmd = [
        sys.executable, 'scripts/evaluate.py',
        '--data_dir', args.data_dir,
        '--model_path', args.model_path,
        '--batch_size', str(args.batch_size)
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Fatty Liver Classification using Siamese Networks')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data
    subparsers.add_parser('generate_data', help='Generate synthetic dataset for testing')
    
    # Download data
    subparsers.add_parser('download_data', help='Instructions for downloading dataset')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of classification epochs')
    train_parser.add_argument('--contrastive_epochs', type=int, default=20, help='Number of contrastive pre-training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--save_path', type=str, default='best_model.pth', help='Path to save best model')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    eval_parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    eval_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Quick start (train + evaluate)
    quick_parser = subparsers.add_parser('quick_start', help='Quick start: generate data, train, and evaluate')
    quick_parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset')
    quick_parser.add_argument('--epochs', type=int, default=5, help='Number of classification epochs (default: 5 for quick test)')
    quick_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    quick_parser.add_argument('--contrastive_epochs', type=int, default=2, help='Number of contrastive epochs')
    
    args = parser.parse_args()
    
    # Default command
    if args.command is None:
        print("Fatty Liver Classification - Siamese Neural Network")
        print("=" * 50)
        print("\nAvailable commands:")
        print("  python main.py generate_data     - Generate synthetic dataset")
        print("  python main.py train             - Train the model")
        print("  python main.py evaluate          - Evaluate the model")
        print("  python main.py quick_start       - Generate data, train, and evaluate")
        print("  python main.py download_data     - Show Kaggle dataset link")
        print("\nExample usage:")
        print("  python main.py train --epochs 50 --batch_size 32")
        print("  python main.py evaluate --model_path best_model.pth")
        print("  python main.py quick_start --epochs 5")
        return

    if args.command == 'generate_data':
        generate_data()
    elif args.command == 'download_data':
        download_data()
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'quick_start':
        print("Starting quick start pipeline...")
        print("\n[Step 1/3] Generating synthetic dataset...")
        if generate_data():
            print("\n[Step 2/3] Training model...")
            train_args = argparse.Namespace(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                contrastive_epochs=args.contrastive_epochs,
                lr=1e-4,
                save_path='best_model.pth'
            )
            if train_model(train_args):
                print("\n[Step 3/3] Evaluating model...")
                eval_args = argparse.Namespace(
                    data_dir=args.data_dir,
                    model_path='best_model.pth',
                    batch_size=args.batch_size
                )
                evaluate_model(eval_args)
            else:
                print("Training failed")
        else:
            print("Dataset generation failed")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


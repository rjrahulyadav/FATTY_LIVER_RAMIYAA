import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat

class FattyLiverDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train

        # Class mapping
        self.class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load all image paths and labels
        self.image_paths = []
        self.labels = []

        # Check if .mat file exists
        mat_file = self.data_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
        if mat_file.exists():
            # Load from .mat file
            self.load_from_mat(mat_file)
        else:
            # Load from image folders
            for class_name in self.class_names:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def load_from_mat(self, mat_file):
        """Load dataset from MATLAB .mat file"""
        data = loadmat(str(mat_file))

        if 'data' in data:
            dataset = data['data'].flatten()  # Shape (55,)

            for sample in dataset:
                sample_id = sample['id'].item()
                class_val = sample['class'].item()
                fat_val = sample['fat'].item()
                images = sample['images']  # Shape (10, 434, 636)

                # Map fat percentage to 5-class label
                if class_val == self.class_to_idx['CLD']:
                    label = self.class_to_idx['CLD']
                elif fat_val < 5:
                    label = 0  # Normal
                elif 5 <= fat_val <= 35:
                    label = 1  # Grade-I
                elif 35 < fat_val <= 65:
                    label = 2  # Grade-II
                elif fat_val > 65:
                    label = 3  # Grade-III
                else:
                    label = 4  # CLD (fallback) - This line will now be redundant if class_val handles CLD.

                # For each of the 10 images in this sample
                for img_idx in range(images.shape[0]):
                    img = images[img_idx]  # Shape (434, 636)
                    # Convert to PIL Image
                    img_pil = Image.fromarray(img, mode='L').convert('RGB')
                    self.image_paths.append(img_pil)
                    self.labels.append(label)
        else:
            raise ValueError("MAT file must contain 'data' field with structured array")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]

        # If img is already a PIL Image (from .mat), use it directly
        if isinstance(img, Image.Image):
            image = img
        else:
            # Otherwise, it's a path, load it
            image = Image.open(img).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_transforms(is_train=True):
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """
    Returns train, val, test data loaders
    For this implementation, we'll split the data into train/val/test
    """
    # Create dataset
    train_transform = get_data_transforms(is_train=True)
    test_transform = get_data_transforms(is_train=False)
    
    full_dataset = FattyLiverDataset(data_dir, transform=None)
    
    # Split dataset (80% train, 10% val, 10% test)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform
    
    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

# For contrastive learning (self-supervised)
class ContrastiveDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Class mapping
        self.class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load all image paths and labels
        self.image_paths = []
        self.labels = []

        # Check if .mat file exists
        mat_file = self.data_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
        if mat_file.exists():
            # Load from .mat file
            self.load_from_mat(mat_file)
        else:
            # Load from image folders
            for class_name in self.class_names:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])

    def load_from_mat(self, mat_file):
        """Load dataset from MATLAB .mat file for contrastive learning"""
        data = loadmat(str(mat_file))

        if 'data' in data:
            dataset = data['data'].flatten()  # Shape (55,)

            for sample in dataset:
                sample_id = sample['id'].item()
                class_val = sample['class'].item()
                fat_val = sample['fat'].item()
                images = sample['images']  # Shape (10, 434, 636)

                # Map fat percentage to 5-class label
                if class_val == self.class_to_idx['CLD']:
                    label = self.class_to_idx['CLD']
                elif fat_val < 5:
                    label = 0  # Normal
                elif 5 <= fat_val <= 35:
                    label = 1  # Grade-I
                elif 35 < fat_val <= 65:
                    label = 2  # Grade-II
                elif fat_val > 65:
                    label = 3  # Grade-III
                else:
                    label = 4  # CLD (fallback) - This line will now be redundant if class_val handles CLD.

                # For each of the 10 images in this sample
                for img_idx in range(images.shape[0]):
                    img = images[img_idx]  # Shape (434, 636)
                    # Convert to PIL Image
                    img_pil = Image.fromarray(img, mode='L').convert('RGB')
                    self.image_paths.append(img_pil)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = self.image_paths[idx]
        label = self.labels[idx]

        # If img is already a PIL Image (from .mat), use it directly
        if isinstance(img, Image.Image):
            image = img
        else:
            # Otherwise, it's a path, load it
            image = Image.open(img).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_positive_pair(self, idx):
        """Get a positive pair (same class) for contrastive learning"""
        label = self.labels[idx]
        # Find another image with same label
        same_class_indices = [i for i, l in enumerate(self.labels) if l == label and i != idx]
        if same_class_indices:
            pos_idx = np.random.choice(same_class_indices)
            pos_image = Image.open(self.image_paths[pos_idx]).convert('RGB')
            if self.transform:
                pos_image = self.transform(pos_image)
            return pos_image
        else:
            # If no other image in same class, return the same image (edge case)
            return self[idx][0]
    
    def get_negative_pair(self, idx):
        """Get a negative pair (different class) for contrastive learning"""
        label = self.labels[idx]
        # Find image with different label
        diff_class_indices = [i for i, l in enumerate(self.labels) if l != label]
        neg_idx = np.random.choice(diff_class_indices)
        neg_image = Image.open(self.image_paths[neg_idx]).convert('RGB')
        if self.transform:
            neg_image = self.transform(neg_image)
        return neg_image

def get_contrastive_data_loader(data_dir, batch_size=32, num_workers=0):
    transform = get_data_transforms(is_train=True)
    dataset = ContrastiveDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

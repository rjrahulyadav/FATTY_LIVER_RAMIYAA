"""
Generate synthetic dataset for testing
"""
import numpy as np
from pathlib import Path
from PIL import Image
import os

def generate_synthetic_dataset(data_dir='data', num_samples_per_class=50):
    """
    Generate synthetic ultrasound images for testing
    Creates folders with synthetic grayscale images mimicking B-mode ultrasound
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    class_names = ['Normal', 'Grade-I', 'Grade-II', 'Grade-III', 'CLD']
    
    print(f"Generating synthetic dataset in {data_dir}/")
    
    for class_name in class_names:
        class_dir = data_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples_per_class):
            # Generate synthetic B-mode ultrasound image
            # Simulate typical ultrasound characteristics: speckle, gradients, varying intensity
            img = generate_ultrasound_image(class_name)
            
            # Save image
            img_path = class_dir / f'{class_name}_{i:03d}.jpg'
            img.save(img_path)
        
        print(f"  ✓ Generated {num_samples_per_class} images for {class_name}")
    
    print(f"✓ Dataset generation complete!")
    return True

def generate_ultrasound_image(class_name, size=(224, 224)):
    """
    Generate synthetic B-mode ultrasound image
    Different intensities and patterns for different classes
    """
    # Base image with typical ultrasound appearance
    img = np.random.uniform(30, 80, size)  # Background speckle (grayscale 30-80)
    
    # Add tissue patterns based on class
    if class_name == 'Normal':
        # Normal liver: uniform echotexture
        img += np.random.normal(0, 5, size)  # Small noise
        intensity_multiplier = 1.0
    elif class_name == 'Grade-I':
        # Mild steatosis: slightly increased echogenicity
        img += np.random.uniform(10, 20, size)
        intensity_multiplier = 1.15
    elif class_name == 'Grade-II':
        # Moderate steatosis: increased echogenicity with some heterogeneity
        img += np.random.uniform(20, 35, size)
        # Add gradient (brighter at top)
        for y in range(size[0]):
            img[y, :] += y / size[0] * 20
        intensity_multiplier = 1.3
    elif class_name == 'Grade-III':
        # Severe steatosis: markedly increased echogenicity
        img += np.random.uniform(35, 50, size)
        # Strong gradient
        for y in range(size[0]):
            img[y, :] += y / size[0] * 35
        intensity_multiplier = 1.5
    else:  # CLD
        # Cirrhosis: coarse pattern with heterogeneous echogenicity
        img += np.random.uniform(25, 45, size)
        # Add some blob-like structures
        num_blobs = np.random.randint(3, 8)
        for _ in range(num_blobs):
            y = np.random.randint(20, size[0] - 20)
            x = np.random.randint(20, size[1] - 20)
            yy, xx = np.ogrid[-20:20, -20:20]
            mask = yy**2 + xx**2 <= 100
            img[y-20:y+20, x-20:x+20][mask] += np.random.uniform(15, 30)
        intensity_multiplier = 1.4
    
    # Add realistic ultrasound artifacts
    # Speckle pattern
    speckle = np.random.exponential(scale=2.0, size=size)
    img = img * (1 + speckle * 0.05)
    
    # Normalize and apply intensity multiplier
    img = np.clip(img * intensity_multiplier, 0, 255)
    
    # Convert to PIL Image (grayscale then RGB)
    img = Image.fromarray(img.astype(np.uint8), mode='L').convert('RGB')
    
    return img

if __name__ == '__main__':
    generate_synthetic_dataset()

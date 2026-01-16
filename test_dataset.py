"""
Quick test to verify dataset loading
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from scipy.io import loadmat
import numpy as np

print("Testing dataset loading...")
mat_file = Path('data/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat')

if mat_file.exists():
    print(f"✓ Dataset file found: {mat_file}")
    
    try:
        data = loadmat(str(mat_file))
        print(f"✓ Dataset loaded successfully")
        print(f"  Keys: {list(data.keys())}")
        
        if 'data' in data:
            dataset = data['data'].flatten()
            print(f"✓ Data array shape: {dataset.shape}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✓ First sample loaded")
                print(f"  Sample keys: {sample.dtype.names if hasattr(sample, 'dtype') else 'N/A'}")
                
                if hasattr(sample, 'dtype') and hasattr(sample, '__getitem__'):
                    try:
                        sample_id = sample['id']
                        class_val = sample['class']
                        fat_val = sample['fat']
                        images = sample['images']
                        print(f"  ID: {sample_id}, Class: {class_val}, Fat: {fat_val:.1f}%")
                        print(f"  Images shape: {images.shape}")
                        print(f"✓ Dataset structure verified!")
                    except Exception as e:
                        print(f"✗ Error accessing sample fields: {e}")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ Dataset file not found: {mat_file}")

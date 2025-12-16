"""
Test script to verify the dataloader fix for GF-7 building dataset.
This demonstrates that labels are correctly converted from 255 to 1.
"""
import sys
import os
sys.path.insert(0, '/home/jebediahc/tum/seminar-rs/UniMatch-V2')
os.chdir('/home/jebediahc/tum/seminar-rs/UniMatch-V2')

from dataset.semi import SemiDataset
import numpy as np

def test_dataloader():
    print("Testing GF-7 Building Dataset Dataloader")
    print("=" * 50)
    
    # Test validation set
    valset = SemiDataset('gf7-building', '/home/jebediahc/tum/seminar-rs/data/gf-7-building-3bands', 'val')
    
    print(f"\nValidation set size: {len(valset)} samples\n")
    
    # Check first 5 samples
    for i in range(min(5, len(valset))):
        img, mask, id = valset[i]
        mask_np = mask.numpy()
        unique_vals = np.unique(mask_np)
        
        print(f"Sample {i+1}:")
        print(f"  Image ID: {id.split('/')[-1]}")
        print(f"  Unique mask values: {unique_vals}")
        print(f"  Building pixels (class 1): {(mask_np == 1).sum():,}")
        print(f"  Background pixels (class 0): {(mask_np == 0).sum():,}")
        
        # Verify no 255 values (which would be treated as ignore_index)
        if 255 in unique_vals:
            print(f"  ⚠️  WARNING: Found 255 values (ignore_index) - {(mask_np == 255).sum():,} pixels")
        else:
            print(f"  ✓ No 255 values found (correct)")
        print()
    
    print("=" * 50)
    print("Summary:")
    print("- Labels should only contain 0 (background) and 1 (building)")
    print("- The value 255 (ignore_index) should NOT appear in labels")
    print("- Previously, building pixels were labeled as 255, causing them to be ignored during training")
    print("- After the fix, building pixels are correctly labeled as 1")

if __name__ == '__main__':
    test_dataloader()

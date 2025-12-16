#!/usr/bin/env python3
"""
Generate splits files for GF-7 Building 4-bands dataset.
Creates splits similar to UniMatch-V2 format:
- all: all training images with labels
- 1_32, 1_64: labeled/unlabeled splits (1/32 or 1/64 labeled)
- small_1_32, small_1_64: smaller versions (1/100 of data for testing)
- val.txt: validation set
"""

import os
import random
from pathlib import Path
from typing import List, Tuple


def get_image_files(base_dir: str, split: str = "Train") -> List[str]:
    """Get all image files from a specific split."""
    image_dir = Path(base_dir) / split / "image"
    if not image_dir.exists():
        print(f"Warning: {image_dir} does not exist")
        return []
    
    # Get all .tif files and sort them
    image_files = sorted([f.name for f in image_dir.glob("*.tif")])
    return image_files


def create_split_line(image_name: str, split: str = "Train") -> str:
    """Create a line for the split file in UniMatch-V2 format."""
    # Format: image_path label_path
    # Remove .tif extension and use it for both image and label paths
    base_name = image_name.replace(".tif", "")
    return f"{split}/image/{image_name} {split}/label/{image_name}"


def generate_labeled_unlabeled_split(
    all_images: List[str],
    ratio: int,
    split_name: str = "Train"
) -> Tuple[List[str], List[str]]:
    """
    Generate labeled and unlabeled splits based on ratio.
    ratio=32 means 1/32 of data is labeled, rest is unlabeled.
    """
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    shuffled = all_images.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    num_labeled = len(shuffled) // ratio
    
    labeled = shuffled[:num_labeled]
    unlabeled = shuffled[num_labeled:]
    
    return labeled, unlabeled


def main():
    base_dir = "data/gf-7-building-4bands"
    output_dir = Path("UniMatch-V2/splits/gf7-building")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating GF-7 Building Dataset Splits")
    print("=" * 60)
    
    # Get all training and validation images
    train_images = get_image_files(base_dir, "Train")
    val_images = get_image_files(base_dir, "Val")
    test_images = get_image_files(base_dir, "Test")
    
    print(f"\nDataset statistics:")
    print(f"  Training images:   {len(train_images):,}")
    print(f"  Validation images: {len(val_images):,}")
    print(f"  Test images:       {len(test_images):,}")
    
    # 1. Generate val.txt (validation split)
    print("\n" + "=" * 60)
    print("Generating validation split...")
    val_file = output_dir / "val.txt"
    with open(val_file, "w") as f:
        for img in val_images:
            f.write(create_split_line(img, "Val") + "\n")
    print(f"  Created: {val_file}")
    print(f"  Entries: {len(val_images)}")
    
    # 2. Generate test.txt (test split)
    print("\nGenerating test split...")
    test_file = output_dir / "test.txt"
    with open(test_file, "w") as f:
        for img in test_images:
            f.write(create_split_line(img, "Test") + "\n")
    print(f"  Created: {test_file}")
    print(f"  Entries: {len(test_images)}")
    
    # 3. Generate 'all' split (all training data with labels)
    print("\n" + "=" * 60)
    print("Generating 'all' split (fully supervised)...")
    all_dir = output_dir / "all"
    all_dir.mkdir(exist_ok=True)
    
    all_file = all_dir / "labeled.txt"
    with open(all_file, "w") as f:
        for img in train_images:
            f.write(create_split_line(img, "Train") + "\n")
    print(f"  Created: {all_file}")
    print(f"  Entries: {len(train_images)}")
    
    # 4. Generate 1_32 split
    print("\n" + "=" * 60)
    print("Generating '1_32' split (1/32 labeled, 31/32 unlabeled)...")
    labeled_32, unlabeled_32 = generate_labeled_unlabeled_split(train_images, 32, "Train")
    
    split_32_dir = output_dir / "1_32"
    split_32_dir.mkdir(exist_ok=True)
    
    labeled_32_file = split_32_dir / "labeled.txt"
    with open(labeled_32_file, "w") as f:
        for img in labeled_32:
            f.write(create_split_line(img, "Train") + "\n")
    
    unlabeled_32_file = split_32_dir / "unlabeled.txt"
    with open(unlabeled_32_file, "w") as f:
        for img in unlabeled_32:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {labeled_32_file}")
    print(f"    Labeled entries:   {len(labeled_32):,}")
    print(f"  Created: {unlabeled_32_file}")
    print(f"    Unlabeled entries: {len(unlabeled_32):,}")
    
    # 5. Generate 1_64 split
    print("\n" + "=" * 60)
    print("Generating '1_64' split (1/64 labeled, 63/64 unlabeled)...")
    labeled_64, unlabeled_64 = generate_labeled_unlabeled_split(train_images, 64, "Train")
    
    split_64_dir = output_dir / "1_64"
    split_64_dir.mkdir(exist_ok=True)
    
    labeled_64_file = split_64_dir / "labeled.txt"
    with open(labeled_64_file, "w") as f:
        for img in labeled_64:
            f.write(create_split_line(img, "Train") + "\n")
    
    unlabeled_64_file = split_64_dir / "unlabeled.txt"
    with open(unlabeled_64_file, "w") as f:
        for img in unlabeled_64:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {labeled_64_file}")
    print(f"    Labeled entries:   {len(labeled_64):,}")
    print(f"  Created: {unlabeled_64_file}")
    print(f"    Unlabeled entries: {len(unlabeled_64):,}")
    
    # 5-1. Generate 1_16 split
    print("\n" + "=" * 60)
    print("Generating '1_16' split (1/16 labeled, 15/16 unlabeled)...")
    labeled_16, unlabeled_16 = generate_labeled_unlabeled_split(train_images, 16, "Train")
    
    split_16_dir = output_dir / "1_16"
    split_16_dir.mkdir(exist_ok=True)
    
    labeled_16_file = split_16_dir / "labeled.txt"
    with open(labeled_16_file, "w") as f:
        for img in labeled_16:
            f.write(create_split_line(img, "Train") + "\n")
    
    unlabeled_16_file = split_16_dir / "unlabeled.txt"
    with open(unlabeled_16_file, "w") as f:
        for img in unlabeled_16:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {labeled_16_file}")
    print(f"    Labeled entries:   {len(labeled_16):,}")
    print(f"  Created: {unlabeled_16_file}")
    print(f"    Unlabeled entries: {len(unlabeled_16):,}")

    # 5-2. Generate 1_8 split
    print("\n" + "=" * 60)
    print("Generating '1_8' split (1/8 labeled, 7/8 unlabeled)...")
    labeled_8, unlabeled_8 = generate_labeled_unlabeled_split(train_images, 8, "Train")
    
    split_8_dir = output_dir / "1_8"
    split_8_dir.mkdir(exist_ok=True)
    
    labeled_8_file = split_8_dir / "labeled.txt"
    with open(labeled_8_file, "w") as f:
        for img in labeled_8:
            f.write(create_split_line(img, "Train") + "\n")
    
    unlabeled_8_file = split_8_dir / "unlabeled.txt"
    with open(unlabeled_8_file, "w") as f:
        for img in unlabeled_8:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {labeled_8_file}")
    print(f"    Labeled entries:   {len(labeled_8):,}")
    print(f"  Created: {unlabeled_8_file}")
    print(f"    Unlabeled entries: {len(unlabeled_8):,}")

    # 5-3. Generate 1_4 split
    print("\n" + "=" * 60)
    print("Generating '1_4' split (1/4 labeled, 3/4 unlabeled)...")
    labeled_4, unlabeled_4 = generate_labeled_unlabeled_split(train_images, 4, "Train")
    
    split_4_dir = output_dir / "1_4"
    split_4_dir.mkdir(exist_ok=True)
    
    labeled_4_file = split_4_dir / "labeled.txt"
    with open(labeled_4_file, "w") as f:
        for img in labeled_4:
            f.write(create_split_line(img, "Train") + "\n")
    
    unlabeled_4_file = split_4_dir / "unlabeled.txt"
    with open(unlabeled_4_file, "w") as f:
        for img in unlabeled_4:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {labeled_4_file}")
    print(f"    Labeled entries:   {len(labeled_4):,}")
    print(f"  Created: {unlabeled_4_file}")
    print(f"    Unlabeled entries: {len(unlabeled_4):,}")

    # 6. Generate small_1_32 split (1/100 of 1_32 data for testing)
    print("\n" + "=" * 60)
    print("Generating 'small_1_32' split (1/100 of data for testing)...")
    
    small_labeled_32 = labeled_32[:max(1, len(labeled_32) // 100)]
    small_unlabeled_32 = unlabeled_32[:max(1, len(unlabeled_32) // 100)]
    
    small_32_dir = output_dir / "small_1_32"
    small_32_dir.mkdir(exist_ok=True)
    
    small_labeled_32_file = small_32_dir / "labeled.txt"
    with open(small_labeled_32_file, "w") as f:
        for img in small_labeled_32:
            f.write(create_split_line(img, "Train") + "\n")
    
    small_unlabeled_32_file = small_32_dir / "unlabeled.txt"
    with open(small_unlabeled_32_file, "w") as f:
        for img in small_unlabeled_32:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {small_labeled_32_file}")
    print(f"    Labeled entries:   {len(small_labeled_32):,}")
    print(f"  Created: {small_unlabeled_32_file}")
    print(f"    Unlabeled entries: {len(small_unlabeled_32):,}")
    
    # 7. Generate small_1_64 split (1/100 of 1_64 data for testing)
    print("\n" + "=" * 60)
    print("Generating 'small_1_64' split (1/100 of data for testing)...")
    
    small_labeled_64 = labeled_64[:max(1, len(labeled_64) // 100)]
    small_unlabeled_64 = unlabeled_64[:max(1, len(unlabeled_64) // 100)]
    
    small_64_dir = output_dir / "small_1_64"
    small_64_dir.mkdir(exist_ok=True)
    
    small_labeled_64_file = small_64_dir / "labeled.txt"
    with open(small_labeled_64_file, "w") as f:
        for img in small_labeled_64:
            f.write(create_split_line(img, "Train") + "\n")
    
    small_unlabeled_64_file = small_64_dir / "unlabeled.txt"
    with open(small_unlabeled_64_file, "w") as f:
        for img in small_unlabeled_64:
            f.write(create_split_line(img, "Train") + "\n")
    
    print(f"  Created: {small_labeled_64_file}")
    print(f"    Labeled entries:   {len(small_labeled_64):,}")
    print(f"  Created: {small_unlabeled_64_file}")
    print(f"    Unlabeled entries: {len(small_unlabeled_64):,}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAll splits created in: {output_dir}/")
    print("\nSplit structure:")
    print(f"  val.txt                    - {len(val_images):,} validation images")
    print(f"  test.txt                   - {len(test_images):,} test images")
    print(f"  all/labeled.txt            - {len(train_images):,} labeled (fully supervised)")
    print(f"  1_32/labeled.txt           - {len(labeled_32):,} labeled")
    print(f"  1_32/unlabeled.txt         - {len(unlabeled_32):,} unlabeled")
    print(f"  1_64/labeled.txt           - {len(labeled_64):,} labeled")
    print(f"  1_64/unlabeled.txt         - {len(unlabeled_64):,} unlabeled")
    print(f"  small_1_32/labeled.txt     - {len(small_labeled_32):,} labeled (testing)")
    print(f"  small_1_32/unlabeled.txt   - {len(small_unlabeled_32):,} unlabeled (testing)")
    print(f"  small_1_64/labeled.txt     - {len(small_labeled_64):,} labeled (testing)")
    print(f"  small_1_64/unlabeled.txt   - {len(small_unlabeled_64):,} unlabeled (testing)")
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

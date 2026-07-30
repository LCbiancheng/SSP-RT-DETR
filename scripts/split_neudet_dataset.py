#!/usr/bin/env python3
"""
Script to split NEU-DET dataset into train/val/test sets with 8:1:1 ratio
"""

import os
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def split_neudet_dataset(root_dir, output_dir):
    """Split NEU-DET dataset into train/val/test sets with 8:1:1 ratio
    
    Args:
        root_dir: Root directory of NEU-DET dataset
        output_dir: Output directory for split dataset
    """
    # Define paths
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    
    img_dir = root_dir / "IMAGES"
    ann_dir = root_dir / "ANNOTATIONS"
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "annotations").mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    # Shuffle the list for random split
    random.shuffle(img_files)
    
    # Calculate split indices
    total = len(img_files)
    train_size = int(total * 0.8)  # 1440 images
    val_size = int(total * 0.1)     # 180 images
    test_size = total - train_size - val_size  # 180 images
    
    print(f"Total images: {total}")
    print(f"Train set size: {train_size}")
    print(f"Val set size: {val_size}")
    print(f"Test set size: {test_size}")
    
    # Split the files
    train_files = img_files[:train_size]
    val_files = img_files[train_size:train_size+val_size]
    test_files = img_files[train_size+val_size:]
    
    # Function to copy files to split directory
    def copy_files(file_list, split):
        for img_file in file_list:
            # Copy image file
            src_img = img_dir / img_file
            dst_img = output_dir / split / "images" / img_file
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding annotation file
            ann_file = img_file.replace('.jpg', '.xml')
            src_ann = ann_dir / ann_file
            dst_ann = output_dir / split / "annotations" / ann_file
            shutil.copy2(src_ann, dst_ann)
    
    # Copy files to respective directories
    print("Copying training files...")
    copy_files(train_files, 'train')
    
    print("Copying validation files...")
    copy_files(val_files, 'val')
    
    print("Copying test files...")
    copy_files(test_files, 'test')
    
    print("Dataset splitting completed successfully!")
    
    # Verify the split
    for split in splits:
        split_img_dir = output_dir / split / "images"
        split_ann_dir = output_dir / split / "annotations"
        img_count = len([f for f in os.listdir(split_img_dir) if f.endswith('.jpg')])
        ann_count = len([f for f in os.listdir(split_ann_dir) if f.endswith('.xml')])
        print(f"{split} set: {img_count} images, {ann_count} annotations")

if __name__ == "__main__":
    # Original NEU-DET directory
    original_root = "./NEU-DET"
    # Output directory for split dataset
    output_root = "./NEU-DET-split"
    
    split_neudet_dataset(original_root, output_root)

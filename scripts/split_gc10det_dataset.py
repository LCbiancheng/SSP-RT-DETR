#!/usr/bin/env python3
"""
Split GC10-DET dataset into train/val/test sets with 8:1:1 ratio
GC10-DET structure:
  - 1/ ... 10/ : images organized by defect class
  - lable/     : XML annotations in VOC format
"""

import os
import random
import shutil
from pathlib import Path

random.seed(42)


def collect_matched_pairs(root_dir):
    """Collect image-XML pairs that exist in both class dirs and lable dir."""
    root_dir = Path(root_dir)
    lable_dir = root_dir / "lable"

    xml_set = set()
    for f in os.listdir(lable_dir):
        if f.endswith('.xml'):
            xml_set.add(f.replace('.xml', '.jpg'))

    matched_pairs = []
    for class_id in range(1, 11):
        class_dir = root_dir / str(class_id)
        if not class_dir.is_dir():
            continue
        for f in os.listdir(class_dir):
            if f.endswith('.jpg') and f in xml_set:
                matched_pairs.append((str(class_id), f))

    return matched_pairs


def split_gc10det_dataset(root_dir, output_dir):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    matched_pairs = collect_matched_pairs(root_dir)
    total = len(matched_pairs)
    print(f"Total matched image-XML pairs: {total}")

    random.shuffle(matched_pairs)

    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size

    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_pairs = matched_pairs[:train_size]
    val_pairs = matched_pairs[train_size:train_size + val_size]
    test_pairs = matched_pairs[train_size + val_size:]

    splits = ['train', 'val', 'test']
    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "annotations").mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs, split):
        lable_dir = root_dir / "lable"
        for class_id, img_file in pairs:
            src_img = root_dir / class_id / img_file
            dst_img = output_dir / split / "images" / img_file
            shutil.copy2(src_img, dst_img)

            ann_file = img_file.replace('.jpg', '.xml')
            src_ann = lable_dir / ann_file
            dst_ann = output_dir / split / "annotations" / ann_file
            shutil.copy2(src_ann, dst_ann)

    print("Copying training files...")
    copy_pairs(train_pairs, 'train')
    print("Copying validation files...")
    copy_pairs(val_pairs, 'val')
    print("Copying test files...")
    copy_pairs(test_pairs, 'test')

    print("Dataset splitting completed!")

    for split in splits:
        split_img_dir = output_dir / split / "images"
        split_ann_dir = output_dir / split / "annotations"
        img_count = len([f for f in os.listdir(split_img_dir) if f.endswith('.jpg')])
        ann_count = len([f for f in os.listdir(split_ann_dir) if f.endswith('.xml')])
        print(f"  {split}: {img_count} images, {ann_count} annotations")


if __name__ == "__main__":
    original_root = "./GC10-DET"
    output_root = "./GC10-DET-split"
    split_gc10det_dataset(original_root, output_root)

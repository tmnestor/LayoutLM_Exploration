#!/usr/bin/env python3
"""
Create validation/test split from existing preprocessed data.
This solves the OCR inconsistency issue by using the same preprocessing as training data.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict

from yaml_config_manager import load_config


def create_validation_split(
    processed_data_dir: str,
    validation_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
    config_path: str = None
) -> Dict[str, int]:
    """
    Create validation/test split from existing preprocessed data.
    
    Args:
        processed_data_dir: Directory containing preprocessed annotation files
        validation_dir: Output directory for validation data
        test_ratio: Fraction of data to use for validation/testing
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with statistics about the split
    """
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Setup paths
    processed_path = Path(processed_data_dir)
    validation_path = Path(validation_dir)
    
    # Clear existing validation directory
    if validation_path.exists():
        shutil.rmtree(validation_path)
        print(f"🗑️  Cleared existing validation directory: {validation_path}")
    
    # Create validation directories
    val_images_dir = validation_path / "validation_images"
    val_gt_dir = validation_path / "ground_truth"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all preprocessed annotation files
    annotation_files = list(processed_path.glob("*_annotation.json"))
    
    if not annotation_files:
        raise ValueError(f"No annotation files found in {processed_path}")
    
    # Randomly sample files for validation
    num_validation = max(1, int(len(annotation_files) * test_ratio))
    validation_files = random.sample(annotation_files, num_validation)
    
    print(f"📊 Total preprocessed files: {len(annotation_files)}")
    print(f"📊 Validation split: {num_validation} files ({test_ratio*100:.1f}%)")
    
    # Load label mapping once at the beginning
    if config_path is None:
        # Default config path relative to processed data directory
        default_config_path = processed_path.parent.parent / "config" / "config.yaml"
    else:
        default_config_path = Path(config_path)
    
    config = load_config(str(default_config_path))
    label_mapping = config.get_label_mapping()
    print(f"📋 Loaded label mapping: {label_mapping}")
    
    # Copy selected files and create ground truth CSVs
    copied_count = 0
    total_tokens = 0
    
    for ann_file in validation_files:
        # Load annotation data
        with ann_file.open('r') as f:
            annotation_data = json.load(f)
        
        # Extract the base name (remove _annotation.json suffix)
        base_name = ann_file.stem.replace('_annotation', '')
        
        # Find corresponding image file
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        source_image = None
        
        # Look for image in the raw images directory
        raw_images_dir = processed_path.parent / "raw" / "images"
        if raw_images_dir.exists():
            for ext in image_extensions:
                image_path = raw_images_dir / f"{base_name}{ext}"
                if image_path.exists():
                    source_image = image_path
                    break
        
        if source_image is None:
            print(f"⚠️  Warning: Could not find image for {base_name}")
            continue
        
        # Copy image to validation directory
        dest_image = val_images_dir / source_image.name
        shutil.copy2(source_image, dest_image)
        
        # Create ground truth CSV from annotation data
        csv_data = []
        words = annotation_data.get('words', [])
        boxes = annotation_data.get('boxes', [])
        labels = annotation_data.get('labels', [])
        
        # Use label mapping loaded earlier
        
        for i, (word, box, label) in enumerate(zip(words, boxes, labels, strict=False)):
            # Format bounding box as string to match prediction format
            bbox_str = f"({box[0]}, {box[1]}, {box[2]}, {box[3]})"
            
            # Convert numeric label to string label using mapping
            if isinstance(label, (int, float)):
                string_label = label_mapping.get(int(label), "O")
            else:
                string_label = label
            
            csv_row = {
                "image_id": base_name,
                "block_ids": 0,  # Default block ID
                "word_ids": i,
                "words": word,
                "bboxes": bbox_str,
                "true_label": string_label,  # Use string label
                "confidence": 1.0,  # Ground truth has 100% confidence
            }
            csv_data.append(csv_row)
        
        # Save ground truth CSV
        import csv
        csv_file = val_gt_dir / f"{base_name}_ground_truth.csv"
        with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "image_id",
                "block_ids", 
                "word_ids",
                "words",
                "bboxes",
                "true_label",
                "confidence",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        copied_count += 1
        total_tokens += len(words)
    
    # Create statistics
    stats = {
        "total_files": len(annotation_files),
        "validation_files": copied_count,
        "total_tokens": total_tokens,
        "avg_tokens_per_file": total_tokens / copied_count if copied_count > 0 else 0,
        "validation_ratio": copied_count / len(annotation_files) if annotation_files else 0
    }
    
    print("\n✅ Validation split created successfully!")
    print("📊 Statistics:")
    print(f"  - Validation files: {stats['validation_files']}")
    print(f"  - Total tokens: {stats['total_tokens']}")
    print(f"  - Average tokens per file: {stats['avg_tokens_per_file']:.1f}")
    print(f"  - Images: {val_images_dir}")
    print(f"  - Ground truth: {val_gt_dir}")
    
    return stats


def main():
    """Main function for creating validation split."""
    parser = argparse.ArgumentParser(
        description="Create validation split from existing preprocessed data"
    )
    
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        required=True,
        help="Directory containing preprocessed annotation files"
    )
    parser.add_argument(
        "--validation_dir",
        type=str,
        required=True,
        help="Output directory for validation data"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation/testing (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file for label mapping (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        stats = create_validation_split(
            processed_data_dir=args.processed_data_dir,
            validation_dir=args.validation_dir,
            test_ratio=args.test_ratio,
            seed=args.seed,
            config_path=args.config
        )
        
        print(f"\n🎯 Ready for evaluation with {stats['validation_files']} validation files!")
        return 0
        
    except Exception as e:
        print(f"❌ Validation split creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
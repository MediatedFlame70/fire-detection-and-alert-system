"""
Create train/val split for the Fire and Smoke dataset.
Splits the dataset and saves the file lists for training.
"""

import argparse
from pathlib import Path
import random
import json


def create_train_val_split(
    image_dir: str,
    annotation_dir: str,
    val_ratio: float = 0.2,
    output_dir: str = ".",
    seed: int = 42,
):
    """
    Create train/validation split from Pascal VOC format dataset.
    
    Args:
        image_dir: Directory containing images
        annotation_dir: Directory containing XML annotations
        val_ratio: Fraction of data to use for validation (default: 0.2)
        output_dir: Directory to save split files
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning dataset...")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotation_dir}")
    
    # Get all image files that have corresponding annotations
    image_files = []
    annotation_files = list(annotation_dir.glob("*.xml"))
    
    print(f"\nFound {len(annotation_files)} annotation files")
    
    for xml_file in annotation_files:
        # Find corresponding image
        image_name = xml_file.stem  # filename without extension
        
        # Try different image extensions
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img_path = image_dir / f"{image_name}{ext}"
            if img_path.exists():
                image_files.append(img_path.name)
                break
    
    print(f"Found {len(image_files)} images with valid annotations")
    
    if len(image_files) == 0:
        print("\n❌ Error: No valid image-annotation pairs found!")
        print("Please check that:")
        print("  1. Image directory contains .jpg/.png files")
        print("  2. Annotation directory contains .xml files")
        print("  3. Filenames match (except extension)")
        return
    
    # Shuffle and split
    random.shuffle(image_files)
    val_count = int(len(image_files) * val_ratio)
    train_count = len(image_files) - val_count
    
    train_files = image_files[val_count:]
    val_files = image_files[:val_count]
    
    print(f"\nSplit summary:")
    print(f"  Training samples:   {len(train_files)} ({100*(1-val_ratio):.0f}%)")
    print(f"  Validation samples: {len(val_files)} ({100*val_ratio:.0f}%)")
    
    # Save splits
    train_file = output_dir / "train_split.txt"
    val_file = output_dir / "val_split.txt"
    
    with open(train_file, "w") as f:
        f.write("\n".join(train_files))
    
    with open(val_file, "w") as f:
        f.write("\n".join(val_files))
    
    # Also save as JSON for easier loading
    split_info = {
        "train": train_files,
        "val": val_files,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "val_ratio": val_ratio,
        "seed": seed,
    }
    
    json_file = output_dir / "data_split.json"
    with open(json_file, "w") as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✓ Split files saved:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {json_file}")
    
    # Print sample files
    print(f"\nSample training files (first 5):")
    for fname in train_files[:5]:
        print(f"  - {fname}")
    
    print(f"\nSample validation files (first 5):")
    for fname in val_files[:5]:
        print(f"  - {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val split for Fire/Smoke dataset")
    
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--annotation-dir", type=str, required=True,
                        help="Directory containing XML annotations")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for split files (default: current dir)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    create_train_val_split(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        seed=args.seed,
    )

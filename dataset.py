"""
Dataset loader for Fire and Smoke Detection Dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class FireSmokeDataset(Dataset):
    """
    Dataset loader for Fire and Smoke detection.
    
    Expected structure:
    dataset_root/
        ├── images/
        │   ├── train/
        │   └── val/
        └── annotations/
            ├── train.json  (COCO format or custom)
            └── val.json
    
    Annotations format (per image):
    {
        "filename": "image_001.jpg",
        "class": "fire" | "smoke" | "neutral",
        "bbox": [x, y, w, h]  # normalized or absolute coords
    }
    """
    
    CLASS_MAPPING = {
        "fire": 0,
        "smoke": 1,
        "neutral": 2,
        "none": 2,  # Some datasets use "none" instead of "neutral"
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 448,
        annotation_file: Optional[str] = None,
        transforms: Optional[T.Compose] = None,
    ):
        """
        Args:
            root_dir: Root directory containing images/ and annotations/
            split: "train" or "val"
            img_size: Target image size (default 448x448)
            annotation_file: Custom annotation file path (optional)
            transforms: Custom transforms (optional, default will be used if None)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        
        # Image directory
        self.img_dir = self.root_dir / "images" / split
        if not self.img_dir.exists():
            # Fallback: check if images are directly in root
            self.img_dir = self.root_dir / split
            if not self.img_dir.exists():
                raise ValueError(f"Image directory not found at {self.img_dir}")
        
        # Load annotations
        if annotation_file:
            anno_path = Path(annotation_file)
        else:
            anno_path = self.root_dir / "annotations" / f"{split}.json"
        
        self.annotations = self._load_annotations(anno_path)
        
        # Transforms
        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms
        
        print(f"Loaded {len(self.annotations)} samples for {split} split")
    
    def _load_annotations(self, anno_path: Path) -> list:
        """Load annotations from JSON file."""
        if not anno_path.exists():
            raise ValueError(f"Annotation file not found: {anno_path}")
        
        with open(anno_path, "r") as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # Direct list of annotations
            return data
        elif "annotations" in data:
            # COCO-style format
            return data["annotations"]
        elif "images" in data:
            # Another common format
            return data["images"]
        else:
            raise ValueError(f"Unknown annotation format in {anno_path}")
    
    def _default_transforms(self) -> T.Compose:
        """Default image preprocessing pipeline."""
        if self.split == "train":
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def _normalize_bbox(self, bbox: list, orig_w: int, orig_h: int) -> torch.Tensor:
        """
        Normalize bounding box to [0, 1] range.
        
        Args:
            bbox: [x, y, w, h] in absolute coordinates
            orig_w, orig_h: Original image dimensions
        
        Returns:
            Normalized bbox tensor [x, y, w, h] in [0, 1]
        """
        x, y, w, h = bbox
        
        # If already normalized (all values < 2), assume it's normalized
        if all(v <= 2.0 for v in bbox):
            return torch.tensor(bbox, dtype=torch.float32)
        
        # Normalize to [0, 1]
        x_norm = x / orig_w
        y_norm = y / orig_h
        w_norm = w / orig_w
        h_norm = h / orig_h
        
        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        
        return torch.tensor([x_norm, y_norm, w_norm, h_norm], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "image": Tensor [3, 448, 448],
                "class_label": Tensor (int),
                "bbox": Tensor [4] normalized [x, y, w, h],
                "filename": str
            }
        """
        anno = self.annotations[idx]
        
        # Load image
        if "filename" in anno:
            img_name = anno["filename"]
        elif "file_name" in anno:
            img_name = anno["file_name"]
        elif "image" in anno:
            img_name = anno["image"]
        else:
            raise KeyError(f"Cannot find image filename in annotation: {anno.keys()}")
        
        img_path = self.img_dir / img_name
        if not img_path.exists():
            # Try without subdirectory
            img_path = self.img_dir.parent / img_name
        
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        
        # Get class label
        if "class" in anno:
            class_str = anno["class"].lower()
        elif "category" in anno:
            class_str = anno["category"].lower()
        elif "label" in anno:
            class_str = anno["label"].lower()
        else:
            class_str = "neutral"  # Default
        
        class_label = self.CLASS_MAPPING.get(class_str, 2)  # Default to neutral
        
        # Get bounding box
        if "bbox" in anno:
            bbox = anno["bbox"]
        elif "bounding_box" in anno:
            bbox = anno["bounding_box"]
        else:
            # Default: entire image
            bbox = [0.0, 0.0, 1.0, 1.0]
        
        # Normalize bbox
        bbox_norm = self._normalize_bbox(bbox, orig_w, orig_h)
        
        # Apply transforms
        image_tensor = self.transforms(image)
        
        return {
            "image": image_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "bbox": bbox_norm,
            "filename": img_name,
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 2,
    img_size: int = 448,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        data_root: Root directory of the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        img_size: Image size (default 448)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = FireSmokeDataset(
        root_dir=data_root,
        split="train",
        img_size=img_size,
    )
    
    val_dataset = FireSmokeDataset(
        root_dir=data_root,
        split="val",
        img_size=img_size,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick test of dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <path_to_dataset>")
        print("Example: python dataset.py ./fire_smoke_data")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    try:
        train_loader, val_loader = create_dataloaders(data_root, batch_size=4)
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test first batch
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Class labels: {batch['class_label']}")
        print(f"  BBox shape: {batch['bbox'].shape}")
        print(f"  BBox sample: {batch['bbox'][0]}")
        
        print("\n✓ Dataset loading test passed!")
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()

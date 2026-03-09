"""
Dataset loader for Fire and Smoke Detection with Pascal VOC XML annotations.
Handles the Datacluster dataset format.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class FireSmokePascalVOCDataset(Dataset):
    """
    Dataset loader for Fire and Smoke detection with Pascal VOC XML annotations.
    
    Expected structure:
    dataset_root/
        ├── images/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── annotations/
            ├── image1.xml
            ├── image2.xml
            └── ...
    """
    
    CLASS_MAPPING = {
        "fire": 0,
        "smoke": 1,
        "neutral": 2,
        "none": 2,
    }
    
    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        image_list: Optional[List[str]] = None,
        img_size: int = 448,
        transforms: Optional[T.Compose] = None,
        is_train: bool = True,
    ):
        """
        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing XML annotations
            image_list: List of image filenames to use (for train/val split)
            img_size: Target image size (default 448x448)
            transforms: Custom transforms (optional)
            is_train: Whether this is training set (affects augmentation)
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.img_size = img_size
        self.is_train = is_train
        
        # Get list of images
        if image_list is not None:
            self.image_files = image_list
        else:
            # Get all images that have corresponding XML annotations
            self.image_files = []
            for xml_file in self.annotation_dir.glob("*.xml"):
                # Parse XML to get image filename
                tree = ET.parse(xml_file)
                root = tree.getroot()
                filename = root.find("filename").text
                
                # Check if image exists
                img_path = self.image_dir / filename
                if img_path.exists():
                    self.image_files.append(filename)
        
        # Transforms
        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms
        
        print(f"Loaded {len(self.image_files)} samples")
    
    def _default_transforms(self) -> T.Compose:
        """Default image preprocessing pipeline."""
        if self.is_train:
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
    
    def _parse_xml_annotation(self, xml_path: Path) -> Dict:
        """
        Parse Pascal VOC XML annotation file.
        
        Returns:
            {
                "filename": str,
                "width": int,
                "height": int,
                "objects": [
                    {
                        "name": str,
                        "bbox": [xmin, ymin, xmax, ymax]
                    },
                    ...
                ]
            }
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image info
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # Get objects
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            bndbox = obj.find("bndbox")
            
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            objects.append({
                "name": name,
                "bbox": [xmin, ymin, xmax, ymax]
            })
        
        return {
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects,
        }
    
    def _get_primary_object(self, objects: List[Dict]) -> Dict:
        """
        Get the primary object for single-object detection.
        For multi-object images, prioritize fire > smoke > others.
        """
        if not objects:
            return {"name": "neutral", "bbox": [0, 0, 1, 1]}
        
        # Priority: fire > smoke > neutral
        priority = {"fire": 0, "smoke": 1}
        
        # Sort by priority
        sorted_objs = sorted(objects, key=lambda x: priority.get(x["name"], 999))
        return sorted_objs[0]
    
    def _normalize_bbox(self, bbox: List[float], orig_w: int, orig_h: int) -> torch.Tensor:
        """
        Convert Pascal VOC bbox [xmin, ymin, xmax, ymax] to normalized [x_center, y_center, w, h].
        
        Args:
            bbox: [xmin, ymin, xmax, ymax] in absolute coordinates
            orig_w, orig_h: Original image dimensions
        
        Returns:
            Normalized bbox tensor [x_center, y_center, w, h] in [0, 1]
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Compute center and dimensions
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        
        # Normalize to [0, 1]
        x_center_norm = x_center / orig_w
        y_center_norm = y_center / orig_h
        width_norm = width / orig_w
        height_norm = height / orig_h
        
        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        width_norm = max(0.0, min(1.0, width_norm))
        height_norm = max(0.0, min(1.0, height_norm))
        
        return torch.tensor([x_center_norm, y_center_norm, width_norm, height_norm], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
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
        filename = self.image_files[idx]
        
        # Load image
        img_path = self.image_dir / filename
        image = Image.open(img_path).convert("RGB")
        
        # Parse annotation
        xml_filename = Path(filename).stem + ".xml"
        xml_path = self.annotation_dir / xml_filename
        
        if xml_path.exists():
            annotation = self._parse_xml_annotation(xml_path)
            primary_obj = self._get_primary_object(annotation["objects"])
            
            # Get class label
            class_name = primary_obj["name"]
            class_label = self.CLASS_MAPPING.get(class_name, 2)  # Default to neutral
            
            # Get normalized bbox
            bbox_norm = self._normalize_bbox(
                primary_obj["bbox"],
                annotation["width"],
                annotation["height"]
            )
        else:
            # No annotation - assume neutral/background
            class_label = 2  # neutral
            bbox_norm = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        
        # Apply transforms
        image_tensor = self.transforms(image)
        
        return {
            "image": image_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "bbox": bbox_norm,
            "filename": filename,
        }


def create_dataloaders_voc(
    image_dir: str,
    annotation_dir: str,
    train_list: List[str],
    val_list: List[str],
    batch_size: int = 8,
    num_workers: int = 2,
    img_size: int = 448,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Pascal VOC format.
    
    Args:
        image_dir: Directory containing images
        annotation_dir: Directory containing XML annotations
        train_list: List of training image filenames
        val_list: List of validation image filenames
        batch_size: Batch size for training
        num_workers: Number of worker processes
        img_size: Image size (default 448)
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = FireSmokePascalVOCDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_list=train_list,
        img_size=img_size,
        is_train=True,
    )
    
    val_dataset = FireSmokePascalVOCDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        image_list=val_list,
        img_size=img_size,
        is_train=False,
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
    
    if len(sys.argv) < 3:
        print("Usage: python dataset_voc.py <image_dir> <annotation_dir>")
        print("Example: python dataset_voc.py ./images ./annotations")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    annotation_dir = sys.argv[2]
    
    try:
        dataset = FireSmokePascalVOCDataset(
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            is_train=True,
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test first sample
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Class label: {sample['class_label']}")
        print(f"  BBox: {sample['bbox']}")
        print(f"  Filename: {sample['filename']}")
        
        print("\n✓ Pascal VOC dataset loading test passed!")
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()

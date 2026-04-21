"""
Dataset loader for the fire/smoke detection dataset.

The workspace dataset is stored in YOLO format:

dataset_root/
    train/
        images/
        labels/
    val/
        images/
        labels/

Each label file contains one or more rows in the form:

class_id x_center y_center width height

All coordinates are normalized to the range [0, 1].
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class FireSmokeDataset(Dataset):
    """Dataset loader that supports YOLO text labels and legacy JSON annotations."""

    CLASS_MAPPING = {
        "fire": 0,
        "smoke": 1,
        "neutral": 2,
        "none": 2,
    }

    CLASS_ID_MAPPING = {
        0: "fire",
        1: "smoke",
        2: "neutral",
    }

    CLASS_PRIORITY = {
        "fire": 0,
        "smoke": 1,
        "neutral": 2,
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 448,
        annotation_file: Optional[str] = None,
        transforms: Optional[T.Compose] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size

        self.img_dir = self._resolve_image_dir()
        self.label_dir = self._resolve_label_dir()

        if transforms is None:
            self.transforms = self._default_transforms()
        else:
            self.transforms = transforms

        if annotation_file:
            self.samples = self._load_json_samples(Path(annotation_file))
        elif self.label_dir is not None:
            self.samples = self._load_yolo_samples(self.label_dir)
        else:
            json_path = self.root_dir / "annotations" / f"{split}.json"
            self.samples = self._load_json_samples(json_path)

        print(f"Loaded {len(self.samples)} samples for {split} split")

    def _resolve_image_dir(self) -> Path:
        candidates = [
            self.root_dir / self.split / "images",
            self.root_dir / "images" / self.split,
            self.root_dir / self.split,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise ValueError(f"Image directory not found for split '{self.split}' under {self.root_dir}")

    def _resolve_label_dir(self) -> Optional[Path]:
        candidates = [
            self.root_dir / self.split / "labels",
            self.root_dir / "labels" / self.split,
        ]

        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*.txt")):
                return candidate

        return None

    def _load_json_samples(self, anno_path: Path) -> List[Dict]:
        if not anno_path.exists():
            raise ValueError(f"Annotation file not found: {anno_path}")

        with open(anno_path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if isinstance(data, list):
            records = data
        elif "annotations" in data:
            records = data["annotations"]
        elif "images" in data:
            records = data["images"]
        else:
            raise ValueError(f"Unknown annotation format in {anno_path}")

        samples: List[Dict] = []
        for record in records:
            filename = record.get("filename") or record.get("file_name") or record.get("image")
            if not filename:
                continue

            image_path = self.img_dir / filename
            if not image_path.exists():
                image_path = self.img_dir.parent / filename
            if not image_path.exists():
                continue

            samples.append(
                {
                    "source": "json",
                    "filename": filename,
                    "image_path": image_path,
                    "record": record,
                }
            )

        return samples

    def _load_yolo_samples(self, label_dir: Path) -> List[Dict]:
        samples: List[Dict] = []
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = self._find_image_for_label(label_path.stem)
            if image_path is None:
                continue

            objects = self._parse_yolo_label_file(label_path)
            samples.append(
                {
                    "source": "yolo",
                    "filename": image_path.name,
                    "image_path": image_path,
                    "label_path": label_path,
                    "objects": objects,
                }
            )

        if not samples:
            raise ValueError(f"No image/label pairs found in {self.img_dir} and {label_dir}")

        return samples

    def _find_image_for_label(self, label_stem: str) -> Optional[Path]:
        extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

        for extension in extensions:
            image_path = self.img_dir / f"{label_stem}{extension}"
            if image_path.exists():
                return image_path

        return None

    def _parse_yolo_label_file(self, label_path: Path) -> List[Dict]:
        objects: List[Dict] = []

        with open(label_path, "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(float(parts[0]))
                    bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                except ValueError:
                    continue

                class_name = self.CLASS_ID_MAPPING.get(class_id, f"class_{class_id}")
                objects.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox": bbox,
                    }
                )

        return objects

    def _select_primary_object(self, objects: List[Dict]) -> Dict:
        if not objects:
            return {"class_id": 2, "class_name": "neutral", "bbox": [0.5, 0.5, 1.0, 1.0]}

        return sorted(objects, key=lambda item: self.CLASS_PRIORITY.get(item["class_name"], 999))[0]
    
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
    
    def _normalize_bbox(self, bbox: List[float], orig_w: int, orig_h: int) -> torch.Tensor:
        if all(0.0 <= value <= 1.0 for value in bbox):
            return torch.tensor(bbox, dtype=torch.float32)

        x, y, w, h = bbox
        x_norm = max(0.0, min(1.0, x / orig_w))
        y_norm = max(0.0, min(1.0, y / orig_h))
        w_norm = max(0.0, min(1.0, w / orig_w))
        h_norm = max(0.0, min(1.0, h / orig_h))

        return torch.tensor([x_norm, y_norm, w_norm, h_norm], dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
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
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        orig_w, orig_h = image.size

        if sample["source"] == "yolo":
            primary_object = self._select_primary_object(sample["objects"])
            class_label = primary_object["class_id"]
            bbox_norm = torch.tensor(primary_object["bbox"], dtype=torch.float32)
        else:
            record = sample["record"]

            if "class" in record:
                class_str = str(record["class"]).lower()
                class_label = self.CLASS_MAPPING.get(class_str, 2)
            elif "category" in record:
                class_str = str(record["category"]).lower()
                class_label = self.CLASS_MAPPING.get(class_str, 2)
            elif "label" in record:
                class_str = str(record["label"]).lower()
                class_label = self.CLASS_MAPPING.get(class_str, 2)
            else:
                class_label = 2

            if "bbox" in record:
                bbox = record["bbox"]
            elif "bounding_box" in record:
                bbox = record["bounding_box"]
            else:
                bbox = [0.5, 0.5, 1.0, 1.0]

            bbox_norm = self._normalize_bbox(bbox, orig_w, orig_h)

        image_tensor = self.transforms(image)

        return {
            "image": image_tensor,
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "bbox": bbox_norm,
            "filename": sample["filename"],
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

"""
Training script for CN2VF-Net on Fire and Smoke Detection Dataset.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cn2vf_net import CN2VFNet
from dataset import create_dataloaders


class CN2VFLoss(nn.Module):
    """
    Combined loss for CN2VF-Net:
    - Classification loss (CrossEntropy)
    - Bounding box regression loss (Smooth L1 / IoU)
    """
    
    def __init__(self, cls_weight=1.0, bbox_weight=5.0, use_iou=False):
        super().__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.use_iou = use_iou
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.bbox_loss_fn = nn.SmoothL1Loss()
    
    def compute_iou_loss(self, pred_bbox, target_bbox):
        """
        Compute IoU-based loss for bounding boxes.
        pred_bbox, target_bbox: [B, 4] in format [x, y, w, h]
        """
        # Convert to [x1, y1, x2, y2]
        pred_x1 = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
        pred_y1 = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
        pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2] / 2
        pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3] / 2
        
        target_x1 = target_bbox[:, 0] - target_bbox[:, 2] / 2
        target_y1 = target_bbox[:, 1] - target_bbox[:, 3] / 2
        target_x2 = target_bbox[:, 0] + target_bbox[:, 2] / 2
        target_y2 = target_bbox[:, 1] + target_bbox[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return 1 - iou.mean()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with "cls_logits" [B, 3] and "bbox" [B, 4]
            targets: dict with "class_label" [B] and "bbox" [B, 4]
        
        Returns:
            total_loss, loss_dict
        """
        cls_logits = predictions["cls_logits"]
        pred_bbox = predictions["bbox"]
        
        target_labels = targets["class_label"]
        target_bbox = targets["bbox"]
        
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_logits, target_labels)
        
        # Bounding box loss
        if self.use_iou:
            bbox_loss = self.compute_iou_loss(pred_bbox, target_bbox)
        else:
            bbox_loss = self.bbox_loss_fn(pred_bbox, target_bbox)
        
        # Total loss
        total_loss = self.cls_weight * cls_loss + self.bbox_weight * bbox_loss
        
        loss_dict = {
            "total": total_loss.item(),
            "cls": cls_loss.item(),
            "bbox": bbox_loss.item(),
        }
        
        return total_loss, loss_dict


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        targets = {
            "class_label": batch["class_label"].to(device),
            "bbox": batch["bbox"].to(device),
        }
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss_dict["total"]
        running_cls_loss += loss_dict["cls"]
        running_bbox_loss += loss_dict["bbox"]
        
        # Classification accuracy
        _, predicted = torch.max(predictions["cls_logits"], 1)
        total += targets["class_label"].size(0)
        correct += (predicted == targets["class_label"]).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "cls": f"{loss_dict['cls']:.4f}",
            "bbox": f"{loss_dict['bbox']:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
        
        # TensorBoard logging
        global_step = epoch * len(dataloader) + batch_idx
        if writer and batch_idx % 10 == 0:
            writer.add_scalar("Train/Loss", loss_dict["total"], global_step)
            writer.add_scalar("Train/ClsLoss", loss_dict["cls"], global_step)
            writer.add_scalar("Train/BBoxLoss", loss_dict["bbox"], global_step)
            writer.add_scalar("Train/Accuracy", 100 * correct / total, global_step)
    
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_bbox_loss = running_bbox_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_bbox_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Validation")
    
    for batch in pbar:
        images = batch["image"].to(device)
        targets = {
            "class_label": batch["class_label"].to(device),
            "bbox": batch["bbox"].to(device),
        }
        
        # Forward pass
        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)
        
        # Metrics
        running_loss += loss_dict["total"]
        running_cls_loss += loss_dict["cls"]
        running_bbox_loss += loss_dict["bbox"]
        
        # Classification accuracy
        _, predicted = torch.max(predictions["cls_logits"], 1)
        total += targets["class_label"].size(0)
        correct += (predicted == targets["class_label"]).sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
    
    avg_loss = running_loss / len(dataloader)
    avg_cls_loss = running_cls_loss / len(dataloader)
    avg_bbox_loss = running_bbox_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # TensorBoard logging
    if writer:
        writer.add_scalar("Val/Loss", avg_loss, epoch)
        writer.add_scalar("Val/ClsLoss", avg_cls_loss, epoch)
        writer.add_scalar("Val/BBoxLoss", avg_bbox_loss, epoch)
        writer.add_scalar("Val/Accuracy", accuracy, epoch)
    
    return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy


def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, save_path):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cn2vf_net_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save training config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=output_dir / "tensorboard") if args.use_tensorboard else None
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_root}")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nInitializing CN2VF-Net...")
    model = CN2VFNet(num_classes=3)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = CN2VFLoss(
        cls_weight=args.cls_weight,
        bbox_weight=args.bbox_weight,
        use_iou=args.use_iou,
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_cls_loss, train_bbox_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_cls_loss, val_bbox_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (cls: {train_cls_loss:.4f}, bbox: {train_bbox_loss:.4f})")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} (cls: {val_cls_loss:.4f}, bbox: {val_bbox_loss:.4f})")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save checkpoints
        if epoch % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                checkpoint_dir / "best_model.pth"
            )
            print(f"  ★ New best validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs, val_loss, val_acc,
        checkpoint_dir / "final_model.pth"
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CN2VF-Net for Fire/Smoke Detection")
    
    # Dataset
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--img-size", type=int, default=448,
                        help="Input image size (default: 448)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    
    # Loss weights
    parser.add_argument("--cls-weight", type=float, default=1.0,
                        help="Classification loss weight (default: 1.0)")
    parser.add_argument("--bbox-weight", type=float, default=5.0,
                        help="Bounding box loss weight (default: 5.0)")
    parser.add_argument("--use-iou", action="store_true",
                        help="Use IoU loss instead of Smooth L1 for bbox")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of data loading workers (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    main(args)

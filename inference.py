"""
Inference script for CN2VF-Net on single images or batch of images.
"""

import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from cn2vf_net import CN2VFNet


class FireSmokePredictor:
    """Predictor for fire and smoke detection."""
    
    CLASS_NAMES = ["Fire", "Smoke", "Neutral"]
    CLASS_COLORS = [(255, 0, 0), (169, 169, 169), (0, 255, 0)]  # Red, Gray, Green
    
    def __init__(self, checkpoint_path, device="cuda"):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: "cuda" or "cpu"
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = CN2VFNet(num_classes=3)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Val Accuracy: {checkpoint.get('val_acc', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transforms = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print("✓ Model loaded successfully")
    
    @torch.no_grad()
    def predict(self, image_path, confidence_threshold=0.5):
        """
        Predict fire/smoke detection on a single image.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence to consider detection valid
        
        Returns:
            {
                "class": str,
                "confidence": float,
                "bbox": [x, y, w, h],  # normalized coords
                "is_detection": bool  # True if fire or smoke detected with high confidence
            }
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Inference
        output = self.model(image_tensor)
        
        # Parse results
        class_probs = torch.softmax(output["cls_logits"], dim=1)
        predicted_class_idx = torch.argmax(class_probs, dim=1).item()
        confidence = class_probs[0, predicted_class_idx].item()
        bbox_norm = output["bbox"][0].cpu().numpy()
        
        class_name = self.CLASS_NAMES[predicted_class_idx]
        is_detection = (predicted_class_idx in [0, 1]) and (confidence >= confidence_threshold)
        
        return {
            "class": class_name,
            "class_idx": predicted_class_idx,
            "confidence": confidence,
            "bbox": bbox_norm.tolist(),
            "is_detection": is_detection,
            "image_size": (orig_w, orig_h),
        }
    
    def visualize(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        Visualize detection results on image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            PIL Image with visualization
        """
        # Get prediction
        result = self.predict(image_path, confidence_threshold)
        
        # Load original image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        orig_w, orig_h = result["image_size"]
        class_idx = result["class_idx"]
        color = self.CLASS_COLORS[class_idx]
        
        # Draw bounding box if detection
        if result["is_detection"]:
            x, y, w, h = result["bbox"]
            
            # Convert normalized coords to absolute
            x1 = int((x - w/2) * orig_w)
            y1 = int((y - h/2) * orig_h)
            x2 = int((x + w/2) * orig_w)
            y2 = int((y + h/2) * orig_h)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # Draw label
            label = f"{result['class']}: {result['confidence']*100:.1f}%"
            
            # Try to use a font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)
        else:
            # Just show class label at top
            label = f"{result['class']}: {result['confidence']*100:.1f}%"
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((10, 10), label, fill=color, font=font)
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
            print(f"✓ Visualization saved to: {output_path}")
        
        return image


def main():
    parser = argparse.ArgumentParser(description="CN2VF-Net Inference for Fire/Smoke Detection")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image or directory of images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save visualizations (optional)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization (only print results)")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = FireSmokePredictor(args.checkpoint, device=args.device)
    
    # Get image paths
    image_path = Path(args.image)
    if image_path.is_dir():
        image_paths = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
        print(f"Found {len(image_paths)} images in directory")
    else:
        image_paths = [image_path]
    
    # Create output directory if needed
    if args.output_dir and not args.no_viz:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    print("\n" + "="*70)
    detections_count = 0
    
    for img_path in image_paths:
        print(f"\nProcessing: {img_path.name}")
        
        # Get prediction
        result = predictor.predict(str(img_path), args.confidence)
        
        # Print results
        print(f"  Class:      {result['class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  BBox:       {result['bbox']}")
        
        if result["is_detection"]:
            print(f"  ⚠️  ALERT: {result['class']} detected!")
            detections_count += 1
        
        # Visualize if requested
        if not args.no_viz:
            output_path = None
            if args.output_dir:
                output_path = Path(args.output_dir) / f"pred_{img_path.name}"
            
            predictor.visualize(str(img_path), output_path, args.confidence)
    
    print("\n" + "="*70)
    print(f"Processed {len(image_paths)} images")
    print(f"Detections (Fire/Smoke): {detections_count}")
    print("="*70)


if __name__ == "__main__":
    main()

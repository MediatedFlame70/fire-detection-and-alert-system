# Quick Start Training Script for Your Fire/Smoke Dataset

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "CN2VF-Net Training - Fire and Smoke Detection" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$IMAGE_DIR = "Datacluster Fire and Smoke Sample/Datacluster Fire and Smoke Sample"
$ANNOTATION_DIR = "Annotations/Annotations"
$TRAIN_SPLIT = "train_split.txt"
$VAL_SPLIT = "val_split.txt"
$PYTHON = "D:/VIT/Projects/fire detection and alert systen/.venv/Scripts/python.exe"

# Check if split files exist
if (-not (Test-Path $TRAIN_SPLIT)) {
    Write-Host "❌ Error: $TRAIN_SPLIT not found!" -ForegroundColor Red
    Write-Host "Run prepare_data.py first to create the split files." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Dataset Configuration:" -ForegroundColor Green
Write-Host "  Images:      $IMAGE_DIR" -ForegroundColor White
Write-Host "  Annotations: $ANNOTATION_DIR" -ForegroundColor White
Write-Host "  Train split: $TRAIN_SPLIT" -ForegroundColor White
Write-Host "  Val split:   $VAL_SPLIT" -ForegroundColor White
Write-Host ""

# Training parameters
$EPOCHS = 50
$BATCH_SIZE = 4
$LEARNING_RATE = 1e-4

Write-Host "Training Parameters:" -ForegroundColor Yellow
Write-Host "  Epochs:      $EPOCHS" -ForegroundColor White
Write-Host "  Batch size:  $BATCH_SIZE" -ForegroundColor White
Write-Host "  Learning rate: $LEARNING_RATE" -ForegroundColor White
Write-Host ""

Write-Host "Starting training in 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

# Start training
& $PYTHON train_voc.py `
  --image-dir $IMAGE_DIR `
  --annotation-dir $ANNOTATION_DIR `
  --train-split $TRAIN_SPLIT `
  --val-split $VAL_SPLIT `
  --epochs $EPOCHS `
  --batch-size $BATCH_SIZE `
  --lr $LEARNING_RATE `
  --cls-weight 1.0 `
  --bbox-weight 5.0 `
  --use-tensorboard `
  --save-interval 10 `
  --num-workers 0 `
  --output-dir ./output

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "Training completed!" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan

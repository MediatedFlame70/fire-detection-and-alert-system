# Quick test script for wildfire image
# Save your wildfire image as test_images\wildfire.jpg first

Write-Host "Testing CN2VF-Net on wildfire image..." -ForegroundColor Cyan

& "D:\VIT\Projects\fire detection and alert systen\.venv\Scripts\python.exe" inference.py `
    --checkpoint "output\cn2vf_net_20260311_125010\checkpoints\best_model.pth" `
    --image "test_images\wildfire.jpg" `
    --output-dir "test_images" `
    --confidence 0.3 `
    --device cpu

Write-Host "`nCheck test_images\pred_wildfire.jpg for the visualization!" -ForegroundColor Green

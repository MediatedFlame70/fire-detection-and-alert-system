"""
Download Kaggle fire and smoke dataset
"""
import kagglehub

# Download latest version
path = kagglehub.dataset_download("azimjaan21/fire-and-smoke-dataset-object-detection-yolo")

print("Path to dataset files:", path)

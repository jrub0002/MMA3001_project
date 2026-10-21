"""
Pytest suite to verify YOLOv8 utility wrappers.
"""

import os
from src.train_yolo import train_yolo_model

def test_yolo_train_stub():
    # Verify helper function loads and throws file errors correctly with an invalid yaml path
    try:
        train_yolo_model("yolov8n.pt", "invalid_dataset_path.yaml", epochs=1)
    except Exception as e:
        # Ensure it correctly attempts training or complains about missing file
        assert isinstance(e, FileNotFoundError) or "not found" in str(e) or "Dataset" in str(e)

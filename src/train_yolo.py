"""
YOLOv8 Training and Evaluation Wrapper for MMA3001.
"""

import os
from typing import Dict, Any
from ultralytics import YOLO

def train_yolo_model(
    model_variant: str, 
    data_yaml_path: str, 
    epochs: int = 20, 
    imgsz: int = 640,
    batch: int = 16,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Trains a YOLOv8 variant with a strict training budget and reproducibility seed.

    Args:
        model_variant: Variant name ('yolov8n.pt' or 'yolov8s.pt').
        data_yaml_path: Absolute path to data.yaml dataset config.
        epochs: Total number of training epochs.
        imgsz: Image resolution.
        batch: Batch size.
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Training metadata and path to the trained weights.
    """
    # Load model variant
    model = YOLO(model_variant)
    
    # Run Training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        project="yolo_training",
        name=model_variant.split('.')[0],
        exist_ok=True
    )
    
    weights_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    
    return {
        "model_name": model_variant,
        "weights_path": weights_path,
        "save_dir": model.trainer.save_dir
    }

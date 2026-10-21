"""
Shared Evaluation Framework for MMA3001 Project.
Provides unified, fair evaluation of object-detection models on accuracy,
latency, throughput, and hardware resource utilization.
"""

import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Array [x1, y1, x2, y2] representing coordinates.
        box2: Array [x1, y1, x2, y2] representing coordinates.

    Returns:
        float: Intersection over Union ratio.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = float(box1_area + box2_area - intersection_area)

    if union_area == 0.0:
        return 0.0
    return intersection_area / union_area

def time_inference(model: Any, dummy_input: torch.Tensor, num_runs: int = 100, warmup_runs: int = 10) -> Tuple[float, float]:
    """
    Measures the exact inference latency on a GPU runtime with device synchronization.

    Args:
        model: The detection model to evaluate.
        dummy_input: Input tensor matching the model's required input size.
        num_runs: Number of timed runs to calculate mean and std.
        warmup_runs: Number of initial untimed runs to warm up the GPU compilation cache.

    Returns:
        Tuple[float, float]: (Mean Latency in ms, Frame-Rate (FPS)).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = dummy_input.to(device)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_input)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    latencies: List[float] = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0) # Convert to ms
        
    mean_latency = float(np.mean(latencies))
    fps = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    return mean_latency, fps

def resource_stats() -> Dict[str, Any]:
    """
    Retrieves system resource parameters and peak GPU memory footprint.

    Returns:
        Dict[str, Any]: Logged system memory and metadata.
    """
    stats = {
        "device_name": "CPU",
        "peak_gpu_allocated_mb": 0.0,
        "peak_gpu_reserved_mb": 0.0
    }
    if torch.cuda.is_available():
        stats["device_name"] = torch.cuda.get_device_name(0)
        stats["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        stats["peak_gpu_reserved_mb"] = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
    return stats

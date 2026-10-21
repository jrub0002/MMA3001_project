"""
Orchestrates training and evaluation of YOLOv8 Nano and Small variants on the 200-image subset.
Saves results to results/yolov8_evaluation.json
"""

import os
import json
import torch
from src.train_yolo import train_yolo_model
from src.eval import time_inference, resource_stats
from ultralytics import YOLO

def main():
    # Use data_subset.yaml to load the 200-image subset
    data_yaml_path = "/content/Project_MMA3001_33854920/pork_yolo/data_subset.yaml"
    results_dir = "/content/Project_MMA3001_33854920/results"
    os.makedirs(results_dir, exist_ok=True)

    # Strict training parameters
    epochs = 20
    imgsz = 640
    batch = 16
    seed = 42

    variants = ["yolov8n.pt", "yolov8s.pt"]
    evaluation_results = {}

    # Ensure we use CUDA if available for benchmarking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating using device: {device}")

    for variant in variants:
        print(f"\n=== Starting Training on 200-image Subset: {variant} ===")
        # Train the model
        train_info = train_yolo_model(
            model_variant=variant,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed
        )

        print(f"Training complete for {variant}. Weights saved at {train_info['weights_path']}")

        # Load trained model weights for benchmarking
        model = YOLO(train_info['weights_path'])
        model.to(device)

        # Create dummy input mimicking raw input size
        dummy_input = torch.zeros((1, 3, imgsz, imgsz)).to(device)

        # Compute Latency & Throughput (FPS)
        print(f"Benchmarking latency and resource usage for {variant}...")
        mean_latency, fps = time_inference(model, dummy_input, num_runs=100, warmup_runs=10)
        resource_usage = resource_stats()

        evaluation_results[variant] = {
            "weights_path": train_info['weights_path'],
            "mean_latency_ms": mean_latency,
            "throughput_fps": fps,
            "peak_gpu_allocated_mb": resource_usage["peak_gpu_allocated_mb"],
            "peak_gpu_reserved_mb": resource_usage["peak_gpu_reserved_mb"],
            "device_name": resource_usage["device_name"]
        }

    # Save metrics
    output_path = os.path.join(results_dir, "yolov8_evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"\nSuccessfully saved evaluation metrics to {output_path}!")

if __name__ == "__main__":
    main()

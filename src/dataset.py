"""
PyTorch Dataset class for preparing packaging dataset inputs into torchvision Faster R-CNN target formats.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

class PorkPackagingDataset(Dataset):
    """
    Custom Dataset to parse image-label sequences for Faster R-CNN,
    converting YOLO format annotations to absolute xyxy boxes.
    """
    def __init__(self, image_paths: List[str], img_size: int = 640):
        """
        Args:
            image_paths: List of absolute image paths to include.
            img_size: Absolute image dimension to scale inputs (default: 640).
        """
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load image via OpenCV
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        # Convert BGR to RGB and scale to 0-1 float tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # Resolve label path
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    # Explicitly slice only the first 4 coordinate components to avoid unpacking errors
                    x_c, y_c, w, h = map(float, parts[1:5])

                    # Scale normalized coordinates to raw pixels
                    x1 = (x_c - w / 2.0) * self.img_size
                    y1 = (y_c - h / 2.0) * self.img_size
                    x2 = (x_c + w / 2.0) * self.img_size
                    y2 = (y_c + h / 2.0) * self.img_size

                    # Sanity check: Ensure box has positive dimensions
                    if (x2 - x1) > 1e-3 and (y2 - y1) > 1e-3:
                        boxes.append([x1, y1, x2, y2])
                        # Shift class indexing by +1 (0 is reserved for background)
                        labels.append(class_id + 1)

        # Construct target dictionary
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        else:
            # Handle empty targets safely
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)

        iscrowd_tensor = torch.zeros((len(labels_tensor),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor
        }

        return img_tensor, target

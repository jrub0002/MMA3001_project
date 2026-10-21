"""
Robust subset-loading and integrity validation module.
"""

import os
import json
import hashlib
from typing import List

def load_train_subset(manifest_path: str = "/content/Project_MMA3001_33854920/results/train_subset_manifest.txt") -> List[str]:
    """
    Loads the 200-image training subset manifest and runs extensive validation checks.
    Raises exceptions if the integrity, file structure, or leakage checks fail.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path}")
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        image_paths = [line.strip() for line in f if line.strip()]
        
    # 1. Assert size requirement
    assert len(image_paths) == 200, f"Manifest error: Expected exactly 200 images, found {len(image_paths)}"
    
    # 2. Assert uniqueness
    assert len(set(image_paths)) == 200, "Manifest error: Image paths are not unique!"
    
    for img_path in image_paths:
        # 3. Assert files exist
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Subset image does not exist: {img_path}")
        
        # 4. Verify corresponding label files exist
        lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lbl_path):
            raise FileNotFoundError(f"Subset corresponding label does not exist: {lbl_path}")
            
        # 5. Check split leakage (compare base names to verify no overlaps)
        img_name = os.path.basename(img_path)
        base_name = img_name.split(".rf.")[0] if ".rf." in img_name else os.path.splitext(img_name)[0]
        
        for split in ["valid", "test"]:
            val_test_dir = f"/content/Project_MMA3001_33854920/pork_yolo/{split}/images"
            if os.path.exists(val_test_dir):
                for f_item in os.listdir(val_test_dir):
                    item_base = f_item.split(".rf.")[0] if ".rf." in f_item else os.path.splitext(f_item)[0]
                    if base_name == item_base:
                        raise ValueError(f"Data Leakage Detected! Subset image '{img_name}' overlaps with {split} image '{f_item}'.")

    return image_paths

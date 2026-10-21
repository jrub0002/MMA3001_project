"""
Pytest Suite for Shared Evaluation Functions
"""

import numpy as np
from src.eval import calculate_iou

def test_calculate_iou_perfect_match():
    box = np.array([0, 0, 10, 10])
    assert calculate_iou(box, box) == 1.0

def test_calculate_iou_no_overlap():
    box1 = np.array([0, 0, 5, 5])
    box2 = np.array([10, 10, 15, 15])
    assert calculate_iou(box1, box2) == 0.0

def test_calculate_iou_partial_overlap():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 0, 15, 10])
    # intersection is width 5, height 10 = 50
    # union is 100 + 100 - 50 = 150
    # iou = 50 / 150 = 0.3333...
    assert abs(calculate_iou(box1, box2) - 0.33333) < 1e-4

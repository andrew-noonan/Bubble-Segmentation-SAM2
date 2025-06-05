# src/utils.py
import cv2
import numpy as np
import os

def load_image_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def overlay_masks(image_rgb, masks, color=(0, 255, 0), alpha=0.4):
    overlay = image_rgb.copy()
    for mask in masks:
        overlay[mask > 0] = (
            alpha * np.array(color) + (1 - alpha) * overlay[mask > 0]
        ).astype("uint8")
    return overlay

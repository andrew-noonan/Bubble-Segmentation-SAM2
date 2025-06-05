# src/segmentor.py
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.config import CKPT_PATH, CFG_PATH

def load_predictor():
    return SAM2ImagePredictor(build_sam2(CFG_PATH, CKPT_PATH))

def run_sam_on_image(predictor, image_rgb):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=None
        )
    return masks

# run_pipeline.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TEST_IMAGE_PATH
from src.utils import load_image_rgb, overlay_masks
from src.segmentor import load_predictor, run_sam_on_image
import matplotlib.pyplot as plt

# Load image
img = load_image_rgb(TEST_IMAGE_PATH)

# Load SAM2 predictor
predictor = load_predictor()

# Run segmentation
masks = run_sam_on_image(predictor, img)

# Visualize
overlay = overlay_masks(img, masks)
plt.imshow(overlay)
plt.axis("off")
plt.title("SAM2 Output")
plt.show()

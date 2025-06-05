# run_pipeline.py
from src.config import TEST_IMAGE_PATH
from src.utils import load_image_rgb, overlay_masks
from src.segmentor import load_predictor, run_sam_on_image
import matplotlib.pyplot as plt

fileNumber = 1
img_path = TEST_IMAGE_PATH
img = load_image_rgb(img_path)
predictor = load_predictor()
masks = run_sam_on_image(predictor, img)
overlay = overlay_masks(img, masks)

plt.imshow(overlay)
plt.axis("off")
plt.title("SAM2 Output")
plt.show()

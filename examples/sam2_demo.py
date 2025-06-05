"""
# SAM2 Image Segmentation Demo

This script demonstrates how to use the SAM2 model for image segmentation.
"""

# Install required packages
!pip install -q transformers torch torchvision numpy opencv-python matplotlib Pillow ipywidgets tqdm

# Clone the repository (if not already cloned)
import os
if not os.path.exists('Bubble-Segmentation-SAM2'):
    !git clone https://github.com/yourusername/Bubble-Segmentation-SAM2.git
    %cd Bubble-Segmentation-SAM2
else:
    %cd Bubble-Segmentation-SAM2

# Add src to Python path
import sys
sys.path.append(os.path.abspath('src'))

# Now import the helper
from sam_utils import SAM2Helper
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Initialize SAM2 Helper
sam_helper = SAM2Helper()

# Example image path - replace with your image path
image_path = "path_to_your_image.jpg"

# Load and display the image
image = sam_helper.load_image(image_path)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

# Example: Generate mask using a point prompt
# Format: [[x, y, label]] where label is 1 for foreground, 0 for background
points = torch.tensor([[[500, 500, 1]]])  # Example point at (500, 500)

# Generate mask
masks = sam_helper.generate_mask(image, points=points)

# Visualize the mask
sam_helper.visualize_mask(image, masks[0])

# Example: Generate mask using a bounding box
# Format: [[x1, y1, x2, y2]]
boxes = torch.tensor([[[100, 100, 400, 400]]])  # Example box

# Generate mask
masks = sam_helper.generate_mask(image, boxes=boxes)

# Visualize the mask
sam_helper.visualize_mask(image, masks[0]) 
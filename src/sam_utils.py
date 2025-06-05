import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor
import cv2

class SAM2Helper:
    def __init__(self, device=None):
        """Initialize SAM2 helper with model and processor."""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
    def load_image(self, image_path):
        """Load and preprocess an image from path."""
        image = Image.open(image_path)
        return image
    
    def process_image(self, image):
        """Process image for SAM2 model."""
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        return inputs, image
    
    def generate_mask(self, image, points=None, boxes=None):
        """Generate mask using SAM2 model with points or boxes."""
        inputs, original_image = self.process_image(image)
        
        if points is not None:
            inputs["input_points"] = points
        if boxes is not None:
            inputs["input_boxes"] = boxes
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        return masks
    
    def visualize_mask(self, image, mask, alpha=0.5):
        """Visualize mask overlay on image."""
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        image = np.array(image)
        mask = mask.cpu().numpy()
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = [255, 0, 0]  # Red mask
        
        # Blend image and mask
        output = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(output)
        plt.axis('off')
        plt.show()
        
        return output 
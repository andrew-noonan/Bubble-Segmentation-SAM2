import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def overlay_grid_opencv(filepath, n):
    """
    Load an image and overlay an n×n grid of red circles using OpenCV
    
    Args:
        filepath (str): Windows file path to the image
        n (int): Number of points per side (creates n×n grid)
    """
    # Load the image
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not load image from {filepath}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate grid spacing
    x_spacing = width / (n + 1)  # +1 to avoid points on edges
    y_spacing = height / (n + 1)
    
    # Draw red circles at grid points
    radius = max(3, min(width, height) // (n * 10))  # Adaptive radius
    color = (0, 0, 255)  # Red in BGR format
    thickness = -1  # Filled circle
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x = int(i * x_spacing)
            y = int(j * y_spacing)
            cv2.circle(img, (x, y), radius, color, thickness)
    
    return img

def overlay_grid_pil(filepath, n):
    """
    Load an image and overlay an n×n grid of red circles using PIL
    
    Args:
        filepath (str): Windows file path to the image
        n (int): Number of points per side (creates n×n grid)
    """
    # Load the image
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    
    # Get image dimensions
    width, height = img.size
    
    # Calculate grid spacing
    x_spacing = width / (n + 1)  # +1 to avoid points on edges
    y_spacing = height / (n + 1)
    
    # Calculate circle radius
    radius = max(3, min(width, height) // (n * 10))  # Adaptive radius
    
    # Draw red circles at grid points
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            x = i * x_spacing
            y = j * y_spacing
            
            # PIL draws circles using bounding box
            left = x - radius
            top = y - radius
            right = x + radius
            bottom = y + radius
            
            draw.ellipse([left, top, right, bottom], fill='red')
    
    return img

def display_with_matplotlib(img_array, title="Image with Grid Overlay"):
    """Display image using matplotlib"""
    plt.figure(figsize=(10, 8))
    if len(img_array.shape) == 3:
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else:
        plt.imshow(img_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example Windows file path - modify as needed
    filepath = r"C:\Users\anoon\Downloads\Frame_0921.png"
    n = 50  # 5x5 grid
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        print("Please update the filepath variable with a valid image path")
    else:
        try:
            # Method 1: Using OpenCV
            result_cv = overlay_grid_opencv(filepath, n)
            display_with_matplotlib(result_cv, f"{n}×{n} Grid Overlay")
            
            # Method 2: Using PIL (alternative)
            # result_pil = overlay_grid_pil(filepath, n)
            # result_pil.show()  # Opens in default image viewer
            
            # Save the result
            output_path = filepath.replace('.', f'_grid_{n}x{n}.')
            cv2.imwrite(output_path, result_cv)
            print(f"Grid overlay saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
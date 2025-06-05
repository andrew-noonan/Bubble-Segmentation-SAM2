# SAM2 Image Segmentation for Google Colab

This repository provides a simple interface to use the Segment Anything Model 2 (SAM2) from Meta AI in Google Colab. It's designed to be easy to use and integrate into your Jupyter notebooks.

## Features

- Easy-to-use helper functions for SAM2 model
- Support for both point and box prompts
- Automatic GPU detection and utilization
- Simple visualization tools
- Google Colab ready

## Quick Start in Google Colab

1. Open a new Google Colab notebook
2. Install the required packages:
```python
!pip install -q transformers torch torchvision numpy opencv-python matplotlib Pillow ipywidgets tqdm
```

3. Clone this repository:
```python
!git clone https://github.com/yourusername/Bubble-Segmentation-SAM2.git
%cd Bubble-Segmentation-SAM2
```

4. Import the helper class:
```python
import sys
sys.path.append('src')
from sam_utils import SAM2Helper
```

5. Initialize the helper:
```python
sam_helper = SAM2Helper()  # Automatically uses GPU if available
```

## Usage Examples

### Loading and Processing Images

```python
# Load an image
image = sam_helper.load_image("path_to_your_image.jpg")

# Process image with point prompts
points = torch.tensor([[[x, y, 1]]])  # 1 for foreground, 0 for background
masks = sam_helper.generate_mask(image, points=points)

# Process image with box prompts
boxes = torch.tensor([[[x1, y1, x2, y2]]])
masks = sam_helper.generate_mask(image, boxes=boxes)

# Visualize results
sam_helper.visualize_mask(image, masks[0])
```

## Example Notebook

Check out the `examples/sam2_demo.ipynb` notebook for a complete demonstration of the library's features.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta AI for the SAM2 model
- Hugging Face for the transformers library


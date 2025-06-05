# Bubble Segmentation in Viscous Fluids using SAM2 and Classical Edge Refinement

This repository contains a hybrid segmentation pipeline for detecting microbubbles (5–100+ px diameter) in oil-based flows using Meta's Segment Anything Model (SAM2) and Canny edge detection with focus filtering.

## Features
- Automatic bubble region segmentation with SAM2
- Sharpness-based filtering using Laplacian or Sobel
- Canny edge refinement for boundary accuracy
- Batch processing for large frame datasets

## Getting Started
Clone this repo and install dependencies:

## Setup
Create and activate the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt


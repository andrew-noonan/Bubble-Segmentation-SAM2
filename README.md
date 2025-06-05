# Bubble Segmentation in Viscous Fluids using SAM2 and Classical Edge Refinement

This repository contains a hybrid segmentation pipeline for detecting microbubbles (5–100+ px diameter) in flows using Meta's Segment Anything Model (SAM2) and Canny edge detection with focus filtering.

## Features
- Automatic bubble region segmentation with SAM2
- Sharpness-based filtering using Laplacian or Sobel
- Canny edge refinement for boundary accuracy
- Batch processing for large frame datasets

---

# Installation Instructions

## For Windows Users: Using WSL (Windows Subsystem for Linux)

If you're on Windows, it is recommended to use Ubuntu via WSL for full compatibility with SAM2, PyTorch, and CUDA.

### Step-by-Step Setup (First-Time Only)

1. **Install WSL and Ubuntu**
   ```powershell
   wsl --install
   ```
   - Reboot your machine when prompted.
   - Launch “Ubuntu” from the Start Menu and complete the setup.

2. **Install Python & Tools inside Ubuntu**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3 python3-venv python3-pip git build-essential -y
   ```

3. **Clone This Repository**
   ```bash
   cd ~
   git clone https://github.com/yourusername/bubble-segmentation.git
   cd bubble-segmentation
   ```

4. **Create and Activate Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

5. **Install Required Python Packages**
   ```bash
   pip install -r requirements.txt
   ```

6. **Clone and Install Meta's SAM2 Repository**
   ```bash
   cd ~
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   ```

7. **Return to Your Project Directory**
   ```bash
   cd ~/bubble-segmentation
   ```

8. **(Optional) Open the Project in VS Code**
   If you have the [Remote - WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl):
   ```bash
   code .
   ```

---

## For Native Linux Users (Ubuntu 20.04+, Debian, etc.)

If you're already using Linux, skip WSL setup and start here:

1. **Install Python and Tools**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install python3 python3-venv python3-pip git build-essential -y
   ```

2. **Clone This Repository**
   ```bash
   cd ~
   git clone https://github.com/yourusername/bubble-segmentation.git
   cd bubble-segmentation
   ```

3. **Create and Activate Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Install Required Python Packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Clone and Install Meta's SAM2 Repository**
   ```bash
   cd ~
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   cd ~/bubble-segmentation
   ```

---

## Download SAM2 Model Weights

1. **Manually download SAM2 weights and config** ([linked here](https://github.com/facebookresearch/sam2#download-checkpoints)):

Example (Tiny model):
- sam2.1_hiera_tiny.pt
- sam2.1_hiera_t.yaml

2. **Place them in the following directory:**
```
bubble-segmentation/
└── sam_model/
    ├── sam2.1_hiera_tiny.pt
    └── sam2.1_hiera_t.yaml
```

---

## Running the Pipeline

Once installed, you can run the bubble segmentation pipeline on a test frame:
```bash
python run_pipeline.py
```

You can modify the image path, model config, and other options in `src/config.py`.

---

## Notes

- This project is designed to run entirely inside Linux (native or WSL) for compatibility with PyTorch and SAM2.
- GPU acceleration with CUDA is supported in WSL2 if NVIDIA drivers are properly installed.


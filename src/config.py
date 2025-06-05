# src/config.py
import os
import torch

# Assume project root as current working directory
CFG_PATH = "../sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
CKPT_PATH = "../sam2/checkpoints/sam2.1_hiera_tiny.pt"

TEST_IMAGE_PATH = os.path.join("data", "test_frames", "Frame_0001.png")

# Uncomment if you have NVIDIA GPU + Drivers installed
#DEVICE = "cuda"
#DTYPE = torch.bfloat16

DEVICE = "cpu"
DTYPE = torch.float32

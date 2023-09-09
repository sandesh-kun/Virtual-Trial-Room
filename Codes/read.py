import os
import cv2
import json
import numpy as np

BASE_DIR = "C:/Project/fashion/"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
SEGMENTATION_DIR = os.path.join(BASE_DIR, "segm")
DENSEPOSE_DIR = os.path.join(BASE_DIR, "densepose")

# Load only first 5000 images due to memory constraints
image_files = sorted(os.listdir(IMAGE_DIR))[:5000]
segm_files = sorted(os.listdir(SEGMENTATION_DIR))[:5000]
densepose_files = sorted(os.listdir(DENSEPOSE_DIR))[:5000]

for idx in range(5000):
    # Base file name
    file_base = image_files[idx].replace('.jpg', '')

    # Load and display image shape
    img = cv2.imread(os.path.join(IMAGE_DIR, file_base + '.jpg'))
    if img is not None:
        print(f"Loaded image {file_base}.jpg with shape: {img.shape}")
    else:
        print(f"Could not load image {file_base}.jpg")
    
    # Load and display segmentation shape
    segm = cv2.imread(os.path.join(SEGMENTATION_DIR, file_base + '_segm.png'))
    if segm is not None:
        print(f"Loaded segmentation {file_base}_segm.png with shape: {segm.shape}")
    else:
        print(f"Could not load segmentation {file_base}_segm.png")
    
    # Load and display densepose shape
    densepose = cv2.imread(os.path.join(DENSEPOSE_DIR, file_base + '_densepose.png'))
    if densepose is not None:
        print(f"Loaded Densepose {file_base}_densepose.png with shape: {densepose.shape}")
    else:
        print(f"Could not load Densepose {file_base}_densepose.png")

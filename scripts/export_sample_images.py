"""
One-time script to export Fashion-MNIST training images to data/sample_images/
so that index i in features.npy corresponds to data/sample_images/{i}.png.
Run from project root: python scripts/export_sample_images.py
"""
import os
import numpy as np
from PIL import Image
from torchvision import datasets

# Paths relative to project root (parent of scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "features", "features.npy")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "sample_images")


def main():
    n_samples = np.load(FEATURES_PATH).shape[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    # Load Fashion-MNIST train (no transform so we get raw 28x28 PIL)
    dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    for i in range(n_samples):
        img, _ = dataset[i]
        # img is PIL Image (L mode)
        path = os.path.join(OUT_DIR, f"{i}.png")
        img.save(path)
    print(f"Saved {n_samples} images to {OUT_DIR}")


if __name__ == "__main__":
    main()

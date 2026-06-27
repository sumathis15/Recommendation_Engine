"""
Download Fashion-MNIST (60k train + 10k test) into data/FashionMNIST/.
Torchvision stores the official raw idx files; the app reads train images by catalog index.

Run from project root: python scripts/download_fashion_mnist.py
"""
import os

from torchvision import datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading Fashion-MNIST training set (60,000 images)...")
    train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    print("Downloading Fashion-MNIST test set (10,000 images)...")
    test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)
    print(f"Train: {len(train)} images")
    print(f"Test:  {len(test)} images")
    print(f"Raw files: {os.path.join(DATA_DIR, 'FashionMNIST', 'raw')}")
    print("Done.")


if __name__ == "__main__":
    main()

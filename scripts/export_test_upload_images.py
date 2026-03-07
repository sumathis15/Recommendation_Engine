"""
Export a few Fashion-MNIST test images to data/test_upload/ for use as
upload samples in the Streamlit app. Run from project root:
  python scripts/export_test_upload_images.py
"""
import os
from torchvision import datasets

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "test_upload")

# Fashion-MNIST class names
CLASS_NAMES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)
    # Save one image per class (first occurrence)
    seen = set()
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label not in seen:
            seen.add(label)
            name = f"test_{CLASS_NAMES[label]}.png"
            path = os.path.join(OUT_DIR, name)
            img.save(path)
        if len(seen) == 10:
            break
    # Also save a few extra varied examples
    for idx in [0, 100, 500, 1000, 2000]:
        img, label = dataset[idx]
        path = os.path.join(OUT_DIR, f"test_sample_{idx}.png")
        img.save(path)
    print(f"Test images saved to {OUT_DIR}")
    print("Upload any of these in the Streamlit app to try recommendations.")


if __name__ == "__main__":
    main()

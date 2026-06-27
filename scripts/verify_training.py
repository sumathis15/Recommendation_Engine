"""Quick check that training artifacts are complete and valid."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fashion_model.pth")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "features", "features.npy")
LABEL_PATH = os.path.join(PROJECT_ROOT, "features", "labels.npy")

checks_passed = 0
checks_total = 0


def check(name: str, ok: bool, detail: str = ""):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if ok else "FAIL"
    if ok:
        checks_passed += 1
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


print("=" * 50)
print("Training verification")
print("=" * 50)

check("Model file exists", os.path.isfile(MODEL_PATH), MODEL_PATH)
check("Features file exists", os.path.isfile(FEATURE_PATH), FEATURE_PATH)
check("Labels file exists", os.path.isfile(LABEL_PATH), LABEL_PATH)

features = np.load(FEATURE_PATH)
labels = np.load(LABEL_PATH)
check("Features shape (60000, 512)", features.shape == (60_000, 512), str(features.shape))
check("Labels shape (60000,)", labels.shape == (60_000,), str(labels.shape))
check("Labels in range 0-9", labels.min() >= 0 and labels.max() <= 9)

norms = np.linalg.norm(features, axis=1)
check("Features L2-normalized", np.allclose(norms, 1.0, atol=1e-5),
      f"norm mean={norms.mean():.4f}")

device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
try:
    model.load_state_dict(state, strict=True)
    check("Model loads (strict)", True, "all keys matched")
except RuntimeError as e:
    check("Model loads (strict)", False, str(e))

model.eval()
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
test_ds = datasets.FashionMNIST(
    os.path.join(PROJECT_ROOT, "data"), train=False, download=True, transform=tf
)
loader = DataLoader(test_ds, batch_size=64, shuffle=False)
correct = total = 0
with torch.no_grad():
    for images, y in loader:
        correct += (model(images).argmax(1) == y).sum().item()
        total += y.size(0)
acc = 100 * correct / total
check("Test accuracy >= 90%", acc >= 90.0, f"{acc:.2f}%")

raw_train = os.path.join(PROJECT_ROOT, "data", "FashionMNIST", "raw", "train-images-idx3-ubyte")
check("Fashion-MNIST train raw present", os.path.isfile(raw_train),
      "run: python scripts/download_fashion_mnist.py")

print("=" * 50)
if checks_passed == checks_total:
    print(f"ALL CHECKS PASSED ({checks_passed}/{checks_total})")
    print("Training finished fully. You are good to go.")
    sys.exit(0)
else:
    print(f"SOME CHECKS FAILED ({checks_passed}/{checks_total})")
    sys.exit(1)

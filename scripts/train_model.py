"""
Train Fashion-MNIST ResNet18 model and extract recommendation features.
Run from project root: python scripts/train_model.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fashion_model.pth")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "features", "features.npy")
LABEL_PATH = os.path.join(PROJECT_ROOT, "features", "labels.npy")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

num_classes = 10
epochs_head = 8
epochs_finetune = 4
batch_size = 64
device = torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss, 100 * correct / total


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "features"), exist_ok=True)

    for path in (MODEL_PATH, FEATURE_PATH, LABEL_PATH):
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed stale file: {path}", flush=True)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, train=False, download=True, transform=eval_transform
    )
    feature_dataset = torchvision.datasets.FashionMNIST(
        root=DATA_PATH, train=True, download=False, transform=eval_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    feature_loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training images: {len(train_dataset)}", flush=True)
    print(f"Catalog images: {len(feature_dataset)}", flush=True)

    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print("Phase 1: training classifier head...", flush=True)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    for epoch in range(epochs_head):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Head epoch {epoch + 1}/{epochs_head} - loss: {loss:.2f}, train acc: {train_acc:.2f}%", flush=True)

    print("Phase 2: fine-tuning layer4 + head...", flush=True)
    for param in model.layer4.parameters():
        param.requires_grad = True
    optimizer = optim.Adam([
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.layer4.parameters(), "lr": 1e-4},
    ])
    for epoch in range(epochs_finetune):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Finetune epoch {epoch + 1}/{epochs_finetune} - loss: {loss:.2f}, train acc: {train_acc:.2f}%", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    print("Extracting features...")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    features = []
    labels_all = []
    with torch.no_grad():
        for images, labels in feature_loader:
            images = images.to(device)
            output = feature_extractor(images).view(images.size(0), -1)
            features.append(output.cpu().numpy())
            labels_all.append(labels.numpy())

    features = np.vstack(features).astype(np.float32)
    labels_all = np.hstack(labels_all)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)

    np.save(FEATURE_PATH, features)
    np.save(LABEL_PATH, labels_all)
    print(f"Features saved: {features.shape}")
    print("Done.")


if __name__ == "__main__":
    main()

"""
Load ResNet18 from models/fashion_model.pth and use it as a 512-dim embedding extractor.
Uses CPU only; final classification layer is replaced with Identity().
"""
import os
import torch
import torch.nn as nn
from torchvision import models


# ResNet18 backbone outputs 512-dim features before the final fc layer.
EMBEDDING_DIM = 512


def load_embedding_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load ResNet18 from a saved state_dict, replace the final fc with Identity(),
    and return the model in eval mode for 512-dim embeddings.

    Args:
        model_path: Path to fashion_model.pth (e.g. models/fashion_model.pth).
        device: Device to load the model on (use CPU for this demo).

    Returns:
        Model with .fc replaced by Identity(), ready for inference.
    """
    # Build ResNet18 (no pretrained weights; we load from checkpoint)
    model = models.resnet18(weights=None)
    # Replace final classification layer with Identity to get 512-dim embeddings
    model.fc = nn.Identity()
    # Load saved state_dict; fc keys will not match, so load with strict=False
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

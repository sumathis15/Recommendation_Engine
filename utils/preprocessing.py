"""
Image preprocessing for the recommendation model: 224x224, grayscale->3 channels,
ImageNet normalization.
"""
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# ImageNet mean and std (RGB) for normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Single transform for inference: resize, grayscale->3ch, ToTensor, normalize
PREPROCESS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for the ResNet18 embedding model.

    - Resize to 224x224
    - Convert grayscale to 3 channels
    - Normalize with ImageNet mean/std

    Args:
        image: PIL Image (any size; can be grayscale or RGB).

    Returns:
        Tensor of shape (1, 3, 224, 224) on CPU.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    # Ensure RGB for Grayscale transform (PIL L mode -> 3 channels)
    if image.mode != "RGB" and image.mode != "L":
        image = image.convert("RGB")
    tensor = PREPROCESS_TRANSFORM(image)
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)

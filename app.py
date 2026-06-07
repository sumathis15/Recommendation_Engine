"""
Streamlit UI for content-based image recommendation.
Uses a trained ResNet18 embedding model and precomputed features; recommends top-5
similar images from data/sample_images/.
"""
import os
import streamlit as st
import torch
import numpy as np
from PIL import Image

from utils import load_embedding_model, preprocess_image, ContentRecommender

# Project root (directory containing app.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fashion_model.pth")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "features", "features.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "features", "labels.npy")
SAMPLE_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "sample_images")

# Fashion-MNIST class names for user-friendly captions
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


@st.cache_resource
def get_model():
    """Load and cache the embedding model (CPU only)."""
    device = torch.device("cpu")
    return load_embedding_model(MODEL_PATH, device)


@st.cache_data
def get_features_and_recommender():
    """Load precomputed features/labels and build recommender; cache result."""
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    return ContentRecommender(features, labels), features.shape[0]


def get_embedding(model, tensor_batch, device):
    """Get 512-dim embedding for a single image batch (1, 3, 224, 224)."""
    with torch.no_grad():
        out = model(tensor_batch.to(device))
    return out.cpu().numpy().squeeze()


def main():
    st.set_page_config(page_title="Fashion Image Recommender", layout="wide")
    st.title("Content-based Image Recommendation")
    st.markdown("Upload an image to get the top-5 similar items from the catalog.")

    # Load model and recommender (cached)
    try:
        model = get_model()
        recommender, n_items = get_features_and_recommender()
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}. Ensure models/fashion_model.pth and features/*.npy exist.")
        return
    device = torch.device("cpu")

    # Image upload
    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to see recommendations.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    # Preprocess -> embed -> recommend
    tensor = preprocess_image(image)
    embedding = get_embedding(model, tensor, device)
    indices = recommender.recommend(embedding, top_k=5)

    st.subheader("Top 5 recommended images")
    # Display recommended images from data/sample_images/{idx}.png without classification labels
    cols = st.columns(5)
    for i, idx in enumerate(indices):
        img_path = os.path.join(SAMPLE_IMAGES_DIR, f"{idx}.png")
        with cols[i]:
            if os.path.isfile(img_path):
                rec_img = Image.open(img_path)
                st.image(rec_img, use_container_width=True)
            else:
                st.warning(f"Image not found: {img_path}. Run the export script to populate data/sample_images/.")


if __name__ == "__main__":
    main()

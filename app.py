"""
Streamlit UI for content-based image recommendation.
Uses a trained ResNet18 embedding model and precomputed features; recommends top-5
similar images from the Fashion-MNIST training catalog (60k).
"""
import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import datasets

from utils import load_classifier_model, load_embedding_model, preprocess_image, ContentRecommender

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fashion_model.pth")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "features", "features.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "features", "labels.npy")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


@st.cache_resource
def get_models():
    """Load and cache classifier + embedding models (CPU only)."""
    device = torch.device("cpu")
    classifier = load_classifier_model(MODEL_PATH, device)
    embedder = load_embedding_model(MODEL_PATH, device)
    return classifier, embedder, device


@st.cache_data
def get_features_and_recommender():
    """Load precomputed features/labels and build recommender; cache result."""
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    return ContentRecommender(features, labels), features.shape[0]


@st.cache_resource
def get_catalog_dataset():
    """
    Fashion-MNIST training set (60k). Catalog index i == dataset[i].
    Downloads to data/FashionMNIST/ on first use if missing.
    """
    return datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)


def get_embedding(model, tensor_batch, device):
    """Get 512-dim embedding for a single image batch (1, 3, 224, 224)."""
    with torch.no_grad():
        out = model(tensor_batch.to(device))
    embedding = out.cpu().numpy().squeeze()
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def predict_label(classifier, tensor_batch, device):
    """Predict Fashion-MNIST class for an uploaded image."""
    with torch.no_grad():
        logits = classifier(tensor_batch.to(device))
    return int(logits.argmax(dim=1).item())


def main():
    st.set_page_config(page_title="Fashion Image Recommender", layout="wide")
    st.title("Content-based Image Recommendation")
    st.markdown(
        "Upload an image to get the top-5 **visually similar** items from the 60k Fashion-MNIST "
        "training catalog. Matching uses neural-network embeddings and cosine similarity."
    )

    with st.sidebar:
        st.subheader("How matching works")
        st.markdown(
            "1. Your image → 512-number embedding (ResNet18)\n"
            "2. Compare to all 60,000 catalog embeddings (`features.npy`)\n"
            "3. Pick the 5 highest cosine similarity scores\n"
            "4. Display images from Fashion-MNIST train set by catalog index"
        )
        use_category_boost = st.checkbox(
            "Boost same category",
            value=False,
            help="Optional: nudge scores toward the detected class (T-shirt, Dress, etc.). "
            "Off = pure visual similarity only.",
        )
        boost = 0.15 if use_category_boost else 0.0

    try:
        classifier, embedder, device = get_models()
        recommender, n_items = get_features_and_recommender()
        catalog = get_catalog_dataset()
    except FileNotFoundError as e:
        st.error(
            f"Missing required file: {e}. Ensure models/fashion_model.pth and features/*.npy exist."
        )
        return

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to see recommendations. Try files in `data/test_upload/`.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    tensor = preprocess_image(image)
    predicted_label = predict_label(classifier, tensor, device)
    st.caption(f"Detected category: **{CLASS_NAMES[predicted_label]}**")
    embedding = get_embedding(embedder, tensor, device)
    label_for_ranking = predicted_label if use_category_boost else None
    results = recommender.recommend_with_scores(
        embedding, top_k=5, query_label=label_for_ranking, same_class_boost=boost
    )

    st.subheader("Top 5 recommended images")
    cols = st.columns(5)
    for col, (idx, raw_score, rank_score) in zip(cols, results):
        with col:
            if idx < len(catalog):
                rec_img, _ = catalog[idx]
                label = int(recommender.labels[idx])
                st.image(rec_img, use_container_width=True)
                st.caption(
                    f"{CLASS_NAMES[label]} · catalog #{idx} · "
                    f"visual match {raw_score:.3f}"
                )
            else:
                st.warning(f"Catalog index {idx} out of range.")


if __name__ == "__main__":
    main()

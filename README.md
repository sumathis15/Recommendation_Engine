# Fashion Image Recommendation Demo

Content-based image recommendation using a trained ResNet18 model and precomputed embeddings.

## About the notebook (`FinalProject_1.instructions.ipynb`)

The model and features used by this app were produced by the Jupyter notebook in this repo. Here’s what the notebook does:

1. **Data** – Loads Fashion-MNIST (train/test), resizes to 224×224, converts grayscale to 3 channels, and uses a **subset of 8,000 training images** for speed.
2. **Model** – Builds a **pretrained ResNet18**, freezes the backbone, and replaces the final layer with a **Linear(512, 10)** head for the 10 fashion classes. It trains only that head for 2 epochs (or loads existing `fashion_model.pth` if present).
3. **Evaluation** – Reports test accuracy (e.g. ~81%).
4. **Feature extraction** – Uses the backbone (all layers except the final classifier) to compute **512‑dim embeddings** for each of the 8,000 training images, then saves them to `features/features.npy` and `features/labels.npy`.
5. **Recommendation** – Defines `recommend(index, top_k=5)`: takes the embedding at `index`, computes **cosine similarity** to all stored embeddings, and returns the **top‑5** most similar indices (excluding the query itself).

The **Streamlit app** does not retrain or change any of this. It loads the same saved model and feature files, replaces the final layer with `Identity()` to get 512‑dim embeddings for **uploaded images**, and uses the same cosine-similarity logic to show the top‑5 recommendations with human‑readable class names (T-shirt, Dress, Bag, etc.).

## Prerequisites

- Python 3.8+
- Existing project assets (unchanged):
  - `models/fashion_model.pth`
  - `features/features.npy`
  - `features/labels.npy`
  - `data/FashionMNIST/raw/` (Fashion-MNIST data)

## Setup

1. Create and activate a virtual environment (optional if you already use one):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Populate sample images for the UI (one-time). From the project root:

   ```bash
   python scripts/export_sample_images.py
   ```

   This writes images to `data/sample_images/` so that recommendation indices match filenames (`0.png`, `1.png`, …).

4. Optional: export test images for uploads:

   ```bash
   python scripts/export_test_upload_images.py
   ```

   Test images are saved to `data/test_upload/`.

## Run the app

From the project root:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (e.g. http://localhost:8501). Upload an image to see the top-5 recommended images from the catalog (with class names like T-shirt, Dress, Bag).

## Project layout (new files only)

- `app.py` – Streamlit entry point
- `utils/model.py` – ResNet18 embedding model loader
- `utils/preprocessing.py` – Image preprocessing (224×224, ImageNet norm)
- `utils/recommender.py` – Cosine-similarity recommender
- `scripts/export_sample_images.py` – Exports training images to `data/sample_images/`
- `scripts/export_test_upload_images.py` – Exports test images to `data/test_upload/`
- `data/sample_images/` – Sample images used for display (populated by the script above)
- `data/test_upload/` – Optional test images for upload (populated by the script above)

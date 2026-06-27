# Fashion Image Recommendation Demo

Content-based image recommendation using a trained ResNet18 model and precomputed embeddings on Fashion-MNIST.

**Full learning guide:** [Image Recommendation Guide.md](Image%20Recommendation%20Guide.md)

## What this project does

1. **Train** ResNet18 on 60k Fashion-MNIST images (transfer learning, two phases).
2. **Extract** 512-dim embeddings → `features/features.npy` + `features/labels.npy`.
3. **Recommend** via cosine similarity (notebook + Streamlit app).
4. **Upload** an image in the app → top-5 visually similar items from the 60k catalog.

Current model: **~92.8%** test accuracy. Catalog: **60,000** training images.

## Prerequisites

- Python 3.8+
- **In this repo (git):**
  - `models/fashion_model.pth` — trained model
  - `features/features.npy`, `features/labels.npy` — 60k catalog embeddings (the “search index”)
  - `data/test_upload/` — 10 small PNGs to try uploads in the UI
- **On your machine (not in git):**
  - `data/FashionMNIST/` — full dataset (**required** for the app to show recommendation images)

## Data: what you need locally (and why)

The app finds closest matches using **`features.npy`** (in git), but it **displays** those matches by loading the matching **training** image from Fashion-MNIST on disk.

```
Upload your image
    → embedding compared to features.npy (60k vectors, in git)
    → top 5 catalog indices e.g. [882, 4271, ...]
    → app loads Fashion-MNIST TRAIN images at those indices to show you the pictures
```

| Data | Count | In git? | Required for app? | Purpose |
|------|-------|---------|-------------------|---------|
| **Train images** | 60,000 | No — download | **Yes** | Recommendation catalog display; row `i` in `features.npy` = train image `i` |
| **Test images** | 10,000 | No — download | No (for app UI) | Accuracy / `verify_training.py` only; not in the recommendation catalog |
| **`features.npy`** | 60k rows | Yes | Yes | Precomputed embeddings used for similarity search |
| **`test_upload/`** | 10 PNGs | Yes | Optional | Convenience files to drag into the uploader |

**You do not push the 70k images to git** — they are large and downloadable. Only embeddings + model are stored in the repo.

## Setup

1. Virtual environment (optional):

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Fashion-MNIST locally** (one-time, ~70k images — **not pushed to git**):

   ```bash
   python scripts/download_fashion_mnist.py
   ```

   This downloads the official dataset into `data/FashionMNIST/raw/`:
   - **60,000 train images** — needed so the app can **show** the top-5 matches
   - **10,000 test images** — used for accuracy checks, not for recommendations

   **Alternative:** skip this step and the app will auto-download train data on first run (`download=True`). Running the script upfront is clearer and fetches both splits.

   **Windows (PowerShell), from project root:**

   ```powershell
   .\venv\Scripts\python scripts\download_fashion_mnist.py
   ```

   After download, you should have files like:
   `data/FashionMNIST/raw/train-images-idx3-ubyte` and `t10k-images-idx3-ubyte`.

## Run the app

```bash
streamlit run app.py
```

Upload an image (try `data/test_upload/test_sneaker.png`) to see top-5 matches from the catalog.

## Retrain (optional)

**Notebook:** `FinalProject_1.ipynb` — set `FORCE_RETRAIN = True` to retrain.

**Script:** `python scripts/train_model.py`

**Verify:** `python scripts/verify_training.py`

## Project layout

| Path | Purpose |
|------|---------|
| `FinalProject_1.ipynb` | Train, evaluate, visualize |
| `app.py` | Streamlit demo |
| `utils/` | Model loading, preprocessing, recommender |
| `models/fashion_model.pth` | Trained weights (in repo) |
| `features/*.npy` | Catalog embeddings + labels (in repo) |
| `data/FashionMNIST/` | Downloaded dataset (local only) |
| `data/test_upload/` | Demo upload images (in repo) |
| `scripts/download_fashion_mnist.py` | Download 70k images |
| `scripts/train_model.py` | Full training pipeline |
| `scripts/verify_training.py` | Sanity-check artifacts |

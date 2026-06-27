# Data folders

## `FashionMNIST/` — download locally, **do not commit**

Full official dataset: **60,000 train + 10,000 test** images.

```bash
python scripts/download_fashion_mnist.py
```

Stored as raw idx files in `data/FashionMNIST/raw/` (not PNGs — torchvision reads them directly).

### Train vs test — which does the app use?

| Split | Count | Used for recommendations? |
|-------|-------|---------------------------|
| **Train** | 60,000 | **Yes** — search uses `features.npy`; display loads `train[i]` for catalog index `i` |
| **Test** | 10,000 | **No** — only for measuring accuracy (`verify_training.py`, notebook eval) |

**Catalog index `i`** in `features.npy` = **training image `i`** (indices 0–59999).

Without the train download, the app can still **rank** matches (from `features.npy`) but cannot **show** the images until Fashion-MNIST train data is present.

---

## `test_upload/` — in git (optional demo uploads)

Ten small PNGs from the **test** set (`test_sneaker.png`, etc.).

- For you to **drag into the Streamlit uploader** when testing
- **Not** the recommendation catalog
- **Not** used automatically by the app

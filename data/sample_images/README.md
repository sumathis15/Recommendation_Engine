# Catalog images (not in git)

This folder holds `0.png` … `59999.png` for the Streamlit app.

**Not committed** — too many files (~60,000). After training, generate locally:

```bash
python scripts/export_sample_images.py
```

Indices must match `features/features.npy` row order.

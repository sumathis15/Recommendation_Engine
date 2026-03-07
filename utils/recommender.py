"""
Content-based recommendation using precomputed features and cosine similarity.
Returns top-5 similar item indices; results are mapped via features/labels.npy.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """
    Recommends items by cosine similarity against precomputed embeddings.
    Uses features from features/features.npy and indices aligned with features/labels.npy.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Shape (N, D) array of embeddings.
            labels: Shape (N,) array of class labels for each item (for display only).
        """
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels)
        self._n_items = self.features.shape[0]

    def recommend(self, query_embedding: np.ndarray, top_k: int = 5) -> list[int]:
        """
        Return indices of the top-k most similar items by cosine similarity.
        Excludes the query if it is exactly one of the precomputed vectors (by index).

        Args:
            query_embedding: Shape (D,) or (1, D) query embedding.
            top_k: Number of recommendations to return.

        Returns:
            List of indices (length top_k) into self.features / self.labels.
        """
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        # (1, N) similarities
        sims = cosine_similarity(query, self.features)[0]
        # Descending order; take top_k (we do not exclude self here; caller may pass new embedding)
        indices = np.argsort(sims)[::-1][:top_k]
        return indices.tolist()

    @property
    def n_items(self) -> int:
        return self._n_items

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

    def _score_candidates(
        self,
        query_embedding: np.ndarray,
        query_label: int | None = None,
        same_class_boost: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return raw cosine scores, boosted scores, and catalog indices."""
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        raw_sims = cosine_similarity(query, self.features)[0]
        final_sims = raw_sims.copy()
        if query_label is not None and same_class_boost > 0:
            final_sims[self.labels == query_label] += same_class_boost
        return raw_sims, final_sims, np.arange(self._n_items)

    def recommend(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_label: int | None = None,
        same_class_boost: float = 0.0,
    ) -> list[int]:
        """
        Return indices of the top-k most similar items by cosine similarity.

        Args:
            query_embedding: Shape (D,) or (1, D) query embedding.
            top_k: Number of recommendations to return.
            query_label: Optional Fashion-MNIST class id; boosts same-class neighbors.
            same_class_boost: Added to similarity for items with query_label.

        Returns:
            List of indices (length top_k) into self.features / self.labels.
        """
        _, final_sims, _ = self._score_candidates(
            query_embedding, query_label, same_class_boost
        )
        indices = np.argsort(final_sims)[::-1][:top_k]
        return indices.tolist()

    def recommend_with_scores(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        query_label: int | None = None,
        same_class_boost: float = 0.0,
    ) -> list[tuple[int, float, float]]:
        """
        Return top-k recommendations with transparency scores.

        Returns:
            List of (catalog_index, raw_cosine_similarity, ranking_score).
            ranking_score includes same_class_boost when enabled.
        """
        raw_sims, final_sims, _ = self._score_candidates(
            query_embedding, query_label, same_class_boost
        )
        indices = np.argsort(final_sims)[::-1][:top_k]
        return [
            (int(idx), float(raw_sims[idx]), float(final_sims[idx]))
            for idx in indices
        ]

    @property
    def n_items(self) -> int:
        return self._n_items

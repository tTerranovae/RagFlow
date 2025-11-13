"""Text embedding using sentence transformers."""

from __future__ import annotations

import os
from typing import Any

from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates embeddings for text using sentence transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize the embedder.

        Args:
            model_name: Name of the sentence transformer model to use.
                       Defaults to 'sentence-transformers/all-MiniLM-L6-v2'
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as a list of floats
        """
        if self.model is None:
            self._load_model()

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self._load_model()

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension
        """
        if self.model is None:
            self._load_model()

        # Get dimension by encoding a dummy text
        dummy_embedding = self.model.encode("dummy", convert_to_numpy=True)
        return len(dummy_embedding)


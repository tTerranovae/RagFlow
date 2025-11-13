"""Retriever for finding relevant document chunks."""

from __future__ import annotations

from typing import Any

from src.embedding import Embedder
from src.vectorstore import ChromaVectorStore


class Retriever:
    """Retrieves relevant document chunks using cosine similarity."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedder: Embedder,
        top_k: int = 3,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            vector_store: ChromaDB vector store instance
            embedder: Embedder instance for query vectorization
            top_k: Number of top chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text

        Returns:
            List of relevant chunk dictionaries with 'text', 'metadata', 'distance', 'id'
        """
        # Embed the query
        query_embedding = self.embedder.embed_text(query)

        # Search in vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        return results

    def retrieve_with_scores(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks with similarity scores.

        Args:
            query: Query text

        Returns:
            List of chunks with similarity scores (lower distance = more similar)
        """
        results = self.retrieve(query)

        # Convert distance to similarity score (1 - distance for cosine similarity)
        for result in results:
            result["similarity"] = 1.0 - result.get("distance", 0.0)

        return results


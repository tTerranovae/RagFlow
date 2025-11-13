"""ChromaDB vector store implementation."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """ChromaDB-based vector store for storing and retrieving embeddings."""

    def __init__(
        self,
        db_path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize the ChromaDB vector store.

        Args:
            db_path: Path to store ChromaDB data. Defaults to ./data/chroma_db
            collection_name: Name of the collection. Defaults to 'rag_documents'
        """
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.collection_name = (
            collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        )

        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs. If not provided, will be auto-generated
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if ids is None:
            # Generate IDs based on index
            ids = [f"doc_{i}" for i in range(len(texts))]

        if len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of texts")

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of dictionaries containing 'text', 'metadata', 'distance', and 'id'
        """
        # Build where clause if filter provided
        where = filter_metadata if filter_metadata else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        # Format results
        formatted_results: list[dict[str, Any]] = []
        if results["documents"] and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "id": results["ids"][0][i],
                    }
                )

        return formatted_results

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        return self.collection.count()


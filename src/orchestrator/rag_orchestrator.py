"""End-to-end RAG orchestrator combining all components."""

from __future__ import annotations

import os
from typing import Any

from src.chunking import DocumentChunker
from src.embedding import Embedder
from src.generator import LMStudioClient
from src.retriever import Retriever
from src.vectorstore import ChromaVectorStore


class RAGOrchestrator:
    """Orchestrates the complete RAG pipeline."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        embedding_model: str | None = None,
        db_path: str | None = None,
        collection_name: str | None = None,
        lm_studio_url: str | None = None,
    ) -> None:
        """
        Initialize the RAG orchestrator.

        Args:
            chunk_size: Size of text chunks (default: 500)
            chunk_overlap: Overlap between chunks (default: 50)
            top_k: Number of chunks to retrieve (default: 3)
            embedding_model: Sentence transformer model name
            db_path: Path to ChromaDB storage
            collection_name: ChromaDB collection name
            lm_studio_url: LM Studio API URL
        """
        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=chunk_size
            or int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=chunk_overlap
            or int(os.getenv("CHUNK_OVERLAP", "50")),
        )

        self.embedder = Embedder(model_name=embedding_model)

        self.vector_store = ChromaVectorStore(
            db_path=db_path,
            collection_name=collection_name,
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=top_k or int(os.getenv("TOP_K_CHUNKS", "3")),
        )

        self.generator = LMStudioClient(base_url=lm_studio_url)

    def index_documents(
        self,
        file_paths: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> int:
        """
        Index documents into the vector store.

        Args:
            file_paths: List of file paths to index
            texts: List of text strings to index (alternative to file_paths)

        Returns:
            Number of chunks indexed
        """
        all_chunks: list[dict[str, Any]] = []

        # Process files
        if file_paths:
            for file_path in file_paths:
                chunks = self.chunker.chunk_file(file_path)
                all_chunks.extend(chunks)

        # Process texts
        if texts:
            for i, text in enumerate(texts):
                chunks = self.chunker.chunk_text(text)
                # Add source identifier
                for chunk in chunks:
                    chunk["metadata"]["source_text_index"] = i
                all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(chunk_texts)

        # Prepare metadata and IDs
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        ids = [
            f"chunk_{i}_{metadatas[i].get('chunk_index', i)}"
            for i in range(len(all_chunks))
        ]

        # Add to vector store
        self.vector_store.add_documents(
            texts=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(all_chunks)

    def query(
        self,
        query: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> dict[str, Any]:
        """
        Query the RAG system.

        Args:
            query: User query
            system_prompt: Optional system prompt for the generator
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'answer', 'context_chunks', and 'metadata'
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_with_scores(query)

        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the knowledge base.",
                "context_chunks": [],
                "metadata": {"retrieved_count": 0},
            }

        # Extract context texts
        context_texts = [chunk["text"] for chunk in retrieved_chunks]

        # Generate response
        answer = self.generator.generate_with_context(
            query=query,
            context_chunks=context_texts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": answer,
            "context_chunks": retrieved_chunks,
            "metadata": {
                "retrieved_count": len(retrieved_chunks),
                "query": query,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the indexed documents.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_chunks": self.vector_store.get_collection_count(),
            "embedding_model": self.embedder.model_name,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap,
            "top_k": self.retriever.top_k,
        }


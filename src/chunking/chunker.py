"""Document chunking with overlap support."""

from __future__ import annotations

import re
from typing import Any


class DocumentChunker:
    """Splits documents into chunks with configurable size and overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (default: paragraph breaks)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """
        Split text into chunks with metadata.

        Args:
            text: Input text to chunk

        Returns:
            List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if not text.strip():
            return []

        chunks: list[dict[str, Any]] = []
        current_chunk = ""
        chunk_index = 0

        # Try splitting by separators first
        parts = self._split_by_separators(text)

        for part in parts:
            # If adding this part would exceed chunk size
            if len(current_chunk) + len(part) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "metadata": {
                            "chunk_index": chunk_index,
                            "chunk_size": len(current_chunk),
                        },
                    }
                )
                chunk_index += 1

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + part
            else:
                current_chunk += part

            # If a single part exceeds chunk size, split it further
            while len(current_chunk) > self.chunk_size:
                # Extract chunk of exact size
                chunk_text = current_chunk[: self.chunk_size]
                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "metadata": {
                            "chunk_index": chunk_index,
                            "chunk_size": len(chunk_text),
                        },
                    }
                )
                chunk_index += 1

                # Keep overlap for next chunk
                current_chunk = self._get_overlap_text(chunk_text) + current_chunk[
                    self.chunk_size :
                ]

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "metadata": {
                        "chunk_index": chunk_index,
                        "chunk_size": len(current_chunk),
                    },
                }
            )

        return chunks

    def _split_by_separators(self, text: str) -> list[str]:
        """Split text by separators, trying each separator in order."""
        parts = [text]

        for separator in self.separators:
            if not separator:
                # Last separator: split by character
                new_parts: list[str] = []
                for part in parts:
                    new_parts.extend(list(part))
                return new_parts

            new_parts: list[str] = []
            for part in parts:
                if separator in part:
                    split_parts = part.split(separator)
                    # Add separator back except for last part
                    for i, split_part in enumerate(split_parts[:-1]):
                        new_parts.append(split_part + separator)
                    new_parts.append(split_parts[-1])
                else:
                    new_parts.append(part)
            parts = new_parts

        return parts

    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from the end of a chunk."""
        if self.chunk_overlap == 0:
            return ""

        # Try to overlap at word boundaries
        if len(text) <= self.chunk_overlap:
            return text

        overlap_start = len(text) - self.chunk_overlap
        # Find last space before overlap point for word boundary
        last_space = text.rfind(" ", 0, overlap_start)
        if last_space > overlap_start - self.chunk_overlap // 2:
            return text[last_space + 1 :]

        return text[overlap_start:]

    def chunk_file(self, file_path: str) -> list[dict[str, Any]]:
        """
        Read and chunk a file.

        Args:
            file_path: Path to the file to chunk

        Returns:
            List of chunk dictionaries
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, "r", encoding="latin-1") as f:
                text = f.read()

        chunks = self.chunk_text(text)
        # Add file path to metadata
        for chunk in chunks:
            chunk["metadata"]["source_file"] = file_path

        return chunks


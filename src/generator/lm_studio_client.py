"""LM Studio API client for text generation."""

from __future__ import annotations

import os
from typing import Any

import requests


class LMStudioClient:
    """Client for interacting with LM Studio API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """
        Initialize the LM Studio client.

        Args:
            base_url: Base URL for LM Studio API. Defaults to http://localhost:1234
            api_key: Optional API key for authentication
        """
        self.base_url = base_url or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234")
        self.api_key = api_key or os.getenv("LM_STUDIO_API_KEY", "")
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using LM Studio.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Model name (optional, uses default if not specified)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API

        Returns:
            Generated text response
        """
        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if model:
            payload["model"] = model

        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"LM Studio API request failed: {e}") from e

    def generate_with_context(
        self,
        query: str,
        context_chunks: list[str],
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate response with retrieved context chunks.

        Args:
            query: User query
            context_chunks: List of retrieved context chunks
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # Build context from chunks
        context = "\n\n".join(
            [f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)]
        )

        # Construct prompt
        user_prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        return self.generate(messages=messages, **kwargs)

    def check_health(self) -> bool:
        """
        Check if LM Studio is running and accessible.

        Returns:
            True if LM Studio is accessible, False otherwise
        """
        try:
            # Try to list models endpoint
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


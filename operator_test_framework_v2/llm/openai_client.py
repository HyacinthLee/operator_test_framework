"""
OpenAI API client implementation.
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import override

from .client import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        **kwargs,
    ):
        """Initialize OpenAI client."""
        super().__init__(model=model, **kwargs)
        self.api_key = api_key
    
    @override
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        ...

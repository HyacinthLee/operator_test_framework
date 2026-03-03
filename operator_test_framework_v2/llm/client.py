"""
Base LLM client interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from LLM.
    
    Attributes:
        content: Response content
        model: Model used
        usage: Token usage statistics
        finish_reason: Why generation stopped
        metadata: Additional metadata
    """
    content: str
    model: str
    usage: Dict[str, int] = None
    finish_reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.metadata is None:
            self.metadata = {}


class LLMClient(ABC):
    """Abstract base class for LLM clients.
    
    Provides unified interface for different LLM providers.
    
    Example:
        >>> client = OpenAIClient(api_key="...")
        >>> response = client.complete(
        ...     "Generate test cases for softmax",
        ...     temperature=0.7
        ... )
        >>> print(response.content)
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 60.0,
    ):
        """Initialize LLM client.
        
        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            timeout: Request timeout
        """
        ...
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            LLM response
        """
        ...
    
    def complete_structured(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured completion.
        
        Args:
            prompt: User prompt
            output_schema: Expected output schema
            system_prompt: System prompt
            
        Returns:
            Structured output
        """
        ...
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Chat completion with message history.
        
        Args:
            messages: List of messages (role, content)
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            LLM response
        """
        ...
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        **kwargs,
    ):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name
            **kwargs: Additional arguments
        """
        ...
    
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


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        **kwargs,
    ):
        """Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model name
            **kwargs: Additional arguments
        """
        ...
    
    @override
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        ...

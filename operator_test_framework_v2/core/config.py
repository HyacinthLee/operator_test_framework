"""
Framework configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for LLM integration.
    
    Attributes:
        provider: LLM provider ('openai', 'anthropic', 'local', etc.)
        model: Model name
        api_key: API key
        api_base: API base URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens per request
        timeout: Request timeout in seconds
        retry_count: Number of retries on failure
    """
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 60.0
    retry_count: int = 3


@dataclass
class GenerationConfig:
    """Configuration for test generation.
    
    Attributes:
        num_random_cases: Number of random test cases
        num_boundary_cases: Number of boundary cases
        num_symbolic_cases: Number of symbolic cases
        max_retries: Maximum generation retries
        validation_enabled: Whether to validate generated cases
        deduplication_enabled: Whether to deduplicate cases
    """
    num_random_cases: int = 10
    num_boundary_cases: int = 10
    num_symbolic_cases: int = 5
    max_retries: int = 3
    validation_enabled: bool = True
    deduplication_enabled: bool = True
    seed: Optional[int] = None


@dataclass
class ExecutionConfig:
    """Configuration for test execution.
    
    Attributes:
        max_workers: Maximum parallel workers
        timeout_seconds: Timeout per test case
        continue_on_error: Continue after errors
        capture_traceback: Capture full tracebacks
        device: Device to run on
        memory_limit_mb: Memory limit per test
    """
    max_workers: int = 1
    timeout_seconds: float = 60.0
    continue_on_error: bool = True
    capture_traceback: bool = True
    device: str = "cpu"
    memory_limit_mb: Optional[int] = None


@dataclass
class ReportingConfig:
    """Configuration for report generation.
    
    Attributes:
        formats: Report formats to generate
        output_dir: Output directory
        include_visualizations: Include charts/plots
        include_code_snippets: Include code examples
        template_dir: Custom template directory
    """
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    output_dir: str = "./reports"
    include_visualizations: bool = True
    include_code_snippets: bool = True
    template_dir: Optional[str] = None


@dataclass
class FrameworkConfig:
    """Main framework configuration.
    
    Attributes:
        llm: LLM configuration
        generation: Generation configuration
        execution: Execution configuration
        reporting: Reporting configuration
        cache_dir: Cache directory
        log_level: Logging level
        enable_iterative_repair: Enable repair loop
        max_repair_iterations: Maximum repair iterations
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    cache_dir: str = "./.cache"
    log_level: str = "INFO"
    enable_iterative_repair: bool = True
    max_repair_iterations: int = 5
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrameworkConfig:
        """Create config from dictionary."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        ...
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> FrameworkConfig:
        """Load config from file (JSON/YAML)."""
        ...
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to file."""
        ...

"""
Base class for test generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Result of test case generation.
    
    Attributes:
        test_cases: Generated test cases
        success_count: Number successfully generated
        failure_count: Number that failed validation
        metadata: Generation metadata
    """
    test_cases: List["TestCase"]
    success_count: int
    failure_count: int
    metadata: Dict[str, Any]


class TestGenerator(ABC):
    """Abstract base class for test generators.
    
    All test generators inherit from this class and implement
    the generate method to produce test cases.
    
    Example:
        >>> class MyGenerator(TestGenerator):
        ...     def generate(self, spec, constraint, count):
        ...         # Generate test cases
        ...         return test_cases
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name."""
        ...
    
    @abstractmethod
    def generate(
        self,
        operator_spec: "OperatorSpec",
        constraint: "TensorConstraint",
        count: int,
        **kwargs,
    ) -> GenerationResult:
        """Generate test cases.
        
        Args:
            operator_spec: Operator specification
            constraint: Tensor constraint to satisfy
            count: Number of cases to generate
            **kwargs: Additional parameters
            
        Returns:
            Generation result
        """
        ...
    
    def can_generate(
        self,
        operator_spec: "OperatorSpec",
        constraint: "TensorConstraint",
    ) -> bool:
        """Check if this generator can handle the constraint.
        
        Args:
            operator_spec: Operator specification
            constraint: Tensor constraint
            
        Returns:
            Whether this generator can generate
        """
        return True
    
    def validate_generated(
        self,
        test_cases: List["TestCase"],
        operator_spec: "OperatorSpec",
    ) -> tuple[List["TestCase"], List[tuple["TestCase", str]]]:
        """Validate generated test cases.
        
        Args:
            test_cases: Generated test cases
            operator_spec: Operator specification
            
        Returns:
            Tuple of (valid_cases, invalid_cases_with_errors)
        """
        ...

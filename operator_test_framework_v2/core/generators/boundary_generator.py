"""
Boundary value test case generator.

Generates test cases for boundary values, edge cases, and
corner cases that are likely to trigger bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Callable
from typing_extensions import override
from enum import Enum, auto

from .base import TestGenerator, GenerationResult


class BoundaryStrategy(Enum):
    """Strategies for boundary value selection."""
    MINIMAL = auto()        # Minimum values only
    MAXIMAL = auto()        # Maximum values only
    EXTREMES = auto()       # Both min and max
    OFF_BY_ONE = auto()     # Boundary ± 1
    SPECIAL_VALUES = auto() # Domain-specific special values
    ALL = auto()            # All boundary strategies


@dataclass
class BoundaryValue:
    """A boundary value with metadata.
    
    Attributes:
        value: The boundary value
        description: Description of what boundary this tests
        category: Category (min, max, special, etc.)
        risk_score: Likelihood of triggering bugs (0-1)
    """
    value: Any
    description: str
    category: str
    risk_score: float = 0.5


class BoundaryGenerator(TestGenerator):
    """Boundary value test case generator.
    
    Generates test cases targeting boundary values and edge cases:
    - Dimension boundaries (0, 1, small, large values)
    - Value boundaries (min, max, zero, inf, nan)
    - Shape boundaries (empty tensors, single elements)
    - Boundary combinations
    
    Based on boundary value analysis (BVA) principles.
    
    Example:
        >>> generator = BoundaryGenerator(strategy=BoundaryStrategy.EXTREMES)
        >>> result = generator.generate(spec, constraint, count=20)
        >>> # Includes cases with shape [0], [1], [max_value], etc.
    """
    
    def __init__(
        self,
        strategy: BoundaryStrategy = BoundaryStrategy.ALL,
        combine_boundaries: bool = True,
        max_combinations: int = 100,
    ):
        """Initialize boundary generator.
        
        Args:
            strategy: Boundary value selection strategy
            combine_boundaries: Whether to combine multiple boundaries
            max_combinations: Maximum boundary combinations
        """
        ...
    
    @property
    @override
    def name(self) -> str:
        return "boundary"
    
    @override
    def generate(
        self,
        operator_spec: "OperatorSpec",
        constraint: "TensorConstraint",
        count: int,
        **kwargs,
    ) -> GenerationResult:
        """Generate boundary value test cases.
        
        Args:
            operator_spec: Operator specification
            constraint: Tensor constraint
            count: Number of cases to generate
            **kwargs: Additional parameters
            
        Returns:
            Generation result with boundary test cases
        """
        ...
    
    def compute_boundary_values(
        self,
        constraint: "TensorConstraint",
    ) -> List[BoundaryValue]:
        """Compute boundary values for a constraint.
        
        Args:
            constraint: Tensor constraint
            
        Returns:
            List of boundary values
        """
        ...
    
    def compute_shape_boundaries(
        self,
        shape_constraint: "ShapeConstraint",
    ) -> List[BoundaryValue]:
        """Compute boundary values for shape.
        
        Args:
            shape_constraint: Shape constraint
            
        Returns:
            Shape boundary values (0, 1, typical values, max values)
        """
        ...
    
    def compute_value_boundaries(
        self,
        value_constraint: Optional["ValueConstraint"],
        dtype: str,
    ) -> List[BoundaryValue]:
        """Compute boundary values for tensor values.
        
        Args:
            value_constraint: Value constraint
            dtype: Data type
            
        Returns:
            Value boundary values (min, max, zero, nan, inf, etc.)
        """
        ...
    
    def combine_boundaries(
        self,
        boundaries: List[List[BoundaryValue]],
        max_combinations: int,
    ) -> List[List[BoundaryValue]]:
        """Combine boundary values from multiple dimensions.
        
        Args:
            boundaries: Boundary values per dimension
            max_combinations: Maximum combinations
            
        Returns:
            Combined boundary sets
        """
        ...
    
    def get_special_values(self, dtype: str) -> List[BoundaryValue]:
        """Get special values for a data type.
        
        Args:
            dtype: Data type
            
        Returns:
            Special values (nan, inf, -inf, 0, etc.)
        """
        ...
    
    def get_dtype_limits(self, dtype: str) -> tuple[Any, Any]:
        """Get min and max values for a data type.
        
        Args:
            dtype: Data type
            
        Returns:
            Tuple of (min_value, max_value)
        """
        ...


class ShapeBoundaryAnalyzer:
    """Analyzer for shape-related boundaries."""
    
    def analyze_broadcast_boundaries(
        self,
        shapes: List[List[int]],
    ) -> List[List[BoundaryValue]]:
        """Analyze boundaries for broadcasting compatibility.
        
        Args:
            shapes: Input shapes
            
        Returns:
            Boundary values per shape
        """
        ...
    
    def analyze_reduction_boundaries(
        self,
        input_shape: List[int],
        dim: Optional[Union[int, List[int]]] = None,
    ) -> List[BoundaryValue]:
        """Analyze boundaries for reduction operations.
        
        Args:
            input_shape: Input tensor shape
            dim: Dimension(s) to reduce
            
        Returns:
            Boundary values for reduction
        """
        ...
    
    def analyze_matmul_boundaries(
        self,
        shape_a: List[int],
        shape_b: List[int],
    ) -> List[List[BoundaryValue]]:
        """Analyze boundaries for matrix multiplication.
        
        Args:
            shape_a: Shape of first matrix
            shape_b: Shape of second matrix
            
        Returns:
            Boundary values for both matrices
        """
        ...


class ValueBoundaryAnalyzer:
    """Analyzer for value-related boundaries."""
    
    COMMON_BOUNDARIES = {
        "float32": [
            (float("-inf"), "negative infinity"),
            (-1e38, "large negative"),
            (-1.0, "negative one"),
            (-1e-7, "small negative"),
            (-0.0, "negative zero"),
            (0.0, "zero"),
            (1e-7, "small positive"),
            (1.0, "one"),
            (1e38, "large positive"),
            (float("inf"), "positive infinity"),
            (float("nan"), "not a number"),
        ],
        "int32": [
            (-2**31, "min int32"),
            (-1, "negative one"),
            (0, "zero"),
            (1, "one"),
            (2**31 - 1, "max int32"),
        ],
    }
    
    def get_boundary_values(
        self,
        dtype: str,
        include_special: bool = True,
    ) -> List[BoundaryValue]:
        """Get boundary values for a dtype.
        
        Args:
            dtype: Data type
            include_special: Whether to include special values (nan, inf)
            
        Returns:
            Boundary values
        """
        ...
    
    def get_near_zero_values(self, dtype: str) -> List[BoundaryValue]:
        """Get values near zero for a dtype.
        
        Args:
            dtype: Data type
            
        Returns:
            Values near zero
        """
        ...
    
    def get_extreme_values(self, dtype: str) -> List[BoundaryValue]:
        """Get extreme (min/max) values for a dtype.
        
        Args:
            dtype: Data type
            
        Returns:
            Extreme values
        """
        ...

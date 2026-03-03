"""
Tensor constraint models.

Defines constraints on tensor properties including shape, dtype,
device placement, and value ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable, Set
from enum import Enum, auto
import re


class ConstraintType(Enum):
    """Types of tensor constraints."""
    EXACT = auto()      # Exact match required
    RANGE = auto()      # Within a range
    LIST = auto()       # One of a list
    PATTERN = auto()    # Matches a pattern
    FORMULA = auto()    # Derived from formula
    ANY = auto()        # No constraint


@dataclass
class ShapeConstraint:
    """Constraint on tensor shape.
    
    Supports:
        - Exact shapes: [3, 224, 224]
        - Dynamic shapes: ["B", "C", "H", "W"] or [-1, 3, 224, 224]
        - Ranges: [(1, 32), (3, 3), (224, 224), (224, 224)]
        - Rank constraints: rank=4
        - Broadcast compatibility
    
    Attributes:
        shape: Expected shape (None = any, -1 = dynamic)
        min_rank: Minimum tensor rank
        max_rank: Maximum tensor rank
        broadcastable_with: Other tensors this must broadcast with
        constraint_type: Type of constraint
    
    Example:
        >>> # Image batch: any batch size, 3 channels, 224x224
        >>> constraint = ShapeConstraint(
        ...     shape=[-1, 3, 224, 224],
        ...     min_rank=4,
        ...     max_rank=4
        ... )
        >>> 
        >>> # Dynamic shape with symbolic names
        >>> constraint = ShapeConstraint(
        ...     shape=["batch", "seq_len", "hidden_dim"],
        ...     constraint_type=ConstraintType.FORMULA
        ... )
    """
    shape: Optional[List[Union[int, str]]] = None
    min_rank: Optional[int] = None
    max_rank: Optional[int] = None
    broadcastable_with: Optional[List[str]] = None
    constraint_type: ConstraintType = ConstraintType.EXACT
    
    def validate(self, tensor_shape: List[int]) -> tuple[bool, Optional[str]]:
        """Validate a tensor shape against this constraint.
        
        Args:
            tensor_shape: Shape to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def generate_valid_shape(self, **context) -> List[int]:
        """Generate a valid shape satisfying this constraint.
        
        Args:
            **context: Context for shape generation (e.g., batch_size=4)
            
        Returns:
            Valid shape
        """
        ...
    
    def is_compatible_with(self, other: ShapeConstraint) -> bool:
        """Check if this constraint is compatible with another.
        
        Args:
            other: Another shape constraint
            
        Returns:
            True if compatible
        """
        ...


@dataclass
class DtypeConstraint:
    """Constraint on tensor data type.
    
    Attributes:
        dtypes: Allowed dtypes (empty = any)
        promote_from: Dtypes that can be promoted to allowed dtypes
        constraint_type: Type of constraint
    
    Example:
        >>> # Allow float32 and float64
        >>> constraint = DtypeConstraint(
        ...     dtypes=["float32", "float64"]
        ... )
        >>> 
        >>> # Allow any float type
        >>> constraint = DtypeConstraint(
        ...     dtypes=["float16", "float32", "float64", "bfloat16"]
        ... )
    """
    dtypes: List[str] = None
    promote_from: Optional[List[str]] = None
    constraint_type: ConstraintType = ConstraintType.LIST
    
    def __post_init__(self):
        if self.dtypes is None:
            self.dtypes = []
    
    def validate(self, dtype: str) -> tuple[bool, Optional[str]]:
        """Validate a dtype against this constraint.
        
        Args:
            dtype: Data type string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def get_common_dtype(self, other_dtypes: List[str]) -> Optional[str]:
        """Get common dtype among allowed dtypes.
        
        Args:
            other_dtypes: Other dtypes to find commonality with
            
        Returns:
            Common dtype or None
        """
        ...


@dataclass
class DeviceConstraint:
    """Constraint on tensor device placement.
    
    Attributes:
        devices: Allowed devices ('cpu', 'cuda', 'cuda:0', etc.)
        memory_requirement: Minimum memory required in bytes
        constraint_type: Type of constraint
    
    Example:
        >>> # CUDA only
        >>> constraint = DeviceConstraint(devices=["cuda"])
        >>> 
        >>> # Any device with 1GB memory
        >>> constraint = DeviceConstraint(
        ...     devices=["cpu", "cuda"],
        ...     memory_requirement=1024**3
        ... )
    """
    devices: List[str] = None
    memory_requirement: Optional[int] = None
    constraint_type: ConstraintType = ConstraintType.LIST
    
    def __post_init__(self):
        if self.devices is None:
            self.devices = ["cpu", "cuda"]
    
    def validate(self, device: str) -> tuple[bool, Optional[str]]:
        """Validate a device against this constraint.
        
        Args:
            device: Device string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@dataclass
class ValueConstraint:
    """Constraint on tensor values.
    
    Attributes:
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        allow_nan: Whether NaN values are allowed
        allow_inf: Whether Inf values are allowed
        allowed_values: Specific allowed values
        forbidden_values: Specific forbidden values
        constraint_type: Type of constraint
    
    Example:
        >>> # Probability values [0, 1]
        >>> constraint = ValueConstraint(
        ...     min_value=0.0,
        ...     max_value=1.0
        ... )
        >>> 
        >>> # Non-negative, no inf/nan
        >>> constraint = ValueConstraint(
        ...     min_value=0.0,
        ...     allow_nan=False,
        ...     allow_inf=False
        ... )
    """
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allow_nan: bool = True
    allow_inf: bool = True
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None
    constraint_type: ConstraintType = ConstraintType.RANGE
    
    def validate(self, tensor: Any) -> tuple[bool, Optional[str]]:
        """Validate tensor values against this constraint.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def generate_valid_value(self, shape: List[int], dtype: str) -> Any:
        """Generate valid tensor values satisfying this constraint.
        
        Args:
            shape: Tensor shape
            dtype: Tensor dtype
            
        Returns:
            Tensor with valid values
        """
        ...


@dataclass
class TensorConstraint:
    """Complete tensor constraint specification.
    
    Combines shape, dtype, device, and value constraints into
    a single comprehensive constraint object.
    
    Attributes:
        name: Tensor name/identifier
        shape: Shape constraint
        dtype: Dtype constraint
        device: Device constraint
        values: Value constraint
        requires_grad: Whether gradient computation is required
        description: Human-readable description
    
    Example:
        >>> constraint = TensorConstraint(
        ...     name="input",
        ...     shape=ShapeConstraint(shape=[-1, 3, 224, 224]),
        ...     dtype=DtypeConstraint(dtypes=["float32", "float64"]),
        ...     device=DeviceConstraint(devices=["cpu", "cuda"]),
        ...     values=ValueConstraint(min_value=-1e6, max_value=1e6)
        ... )
    """
    name: str
    shape: Optional[ShapeConstraint] = None
    dtype: Optional[DtypeConstraint] = None
    device: Optional[DeviceConstraint] = None
    values: Optional[ValueConstraint] = None
    requires_grad: bool = False
    description: str = ""
    
    def validate(self, tensor: Any) -> tuple[bool, List[str]]:
        """Validate a tensor against all constraints.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...
    
    def generate(self, **context) -> Any:
        """Generate a tensor satisfying all constraints.
        
        Args:
            **context: Context for generation (e.g., batch_size=4)
            
        Returns:
            Generated tensor
        """
        ...
    
    def intersect(self, other: TensorConstraint) -> TensorConstraint:
        """Compute intersection of two constraints.
        
        Args:
            other: Another tensor constraint
            
        Returns:
            Constraint that satisfies both
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TensorConstraint:
        """Create from dictionary representation."""
        ...

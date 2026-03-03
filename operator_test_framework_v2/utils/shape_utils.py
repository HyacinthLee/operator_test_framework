"""
Shape utility functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def broadcast_shapes(*shapes: List[int]) -> List[int]:
    """Compute broadcasted shape from input shapes.
    
    Args:
        *shapes: Input shapes
        
    Returns:
        Broadcasted shape
    """
    ...


def infer_output_shape(
    operator_name: str,
    input_shapes: Dict[str, List[int]],
    attributes: Dict[str, Any],
) -> List[int]:
    """Infer output shape for an operator.
    
    Args:
        operator_name: Operator name
        input_shapes: Input tensor shapes
        attributes: Operator attributes
        
    Returns:
        Inferred output shape
    """
    ...


def check_shape_compatibility(
    shape1: List[int],
    shape2: List[int],
    broadcast: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Check if two shapes are compatible.
    
    Args:
        shape1: First shape
        shape2: Second shape
        broadcast: Whether to allow broadcasting
        
    Returns:
        Tuple of (compatible, reason_if_not)
    """
    ...


def compute_strides(shape: List[int]) -> List[int]:
    """Compute strides for a shape.
    
    Args:
        shape: Tensor shape
        
    Returns:
        Strides
    """
    ...


def is_contiguous(shape: List[int], strides: List[int]) -> bool:
    """Check if tensor is contiguous.
    
    Args:
        shape: Tensor shape
        strides: Tensor strides
        
    Returns:
        Whether contiguous
    """
    ...

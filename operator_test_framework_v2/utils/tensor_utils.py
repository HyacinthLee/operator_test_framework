"""
Tensor utility functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def generate_random_tensor(
    shape: List[int],
    dtype: str = "float32",
    device: str = "cpu",
    value_constraint: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Any:
    """Generate a random tensor with specified constraints.
    
    Args:
        shape: Tensor shape
        dtype: Data type
        device: Device
        value_constraint: Optional value constraints
        seed: Random seed
        
    Returns:
        Generated tensor
    """
    ...


def compare_tensors(
    actual: Any,
    expected: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare two tensors for approximate equality.
    
    Args:
        actual: Actual tensor
        expected: Expected tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        equal_nan: Whether to treat NaN as equal
        
    Returns:
        Tuple of (is_equal, details)
    """
    ...


def compute_gradient_numerical(
    func: callable,
    inputs: Dict[str, Any],
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Compute numerical gradients using finite differences.
    
    Args:
        func: Function to differentiate
        inputs: Input tensors
        epsilon: Finite difference epsilon
        
    Returns:
        Numerical gradients
    """
    ...


def check_numerical_stability(
    func: callable,
    inputs: Dict[str, Any],
    perturbation_scale: float = 1e-5,
) -> Tuple[bool, Dict[str, Any]]:
    """Check numerical stability of function.
    
    Args:
        func: Function to check
        inputs: Input tensors
        perturbation_scale: Perturbation magnitude
        
    Returns:
        Tuple of (is_stable, details)
    """
    ...

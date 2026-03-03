"""
Numerical stability validator.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


class NumericalValidator:
    """Validator for numerical stability.
    
    Checks:
    - NaN/Inf handling
    - Numerical precision
    - Stability under perturbation
    """
    
    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ):
        """Initialize validator.
        
        Args:
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        self.rtol = rtol
        self.atol = atol
    
    def validate(
        self,
        func: callable,
        inputs: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate numerical stability.
        
        Args:
            func: Function to validate
            inputs: Input tensors
            
        Returns:
            Tuple of (is_stable, details)
        """
        ...

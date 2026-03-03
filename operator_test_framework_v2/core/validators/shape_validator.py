"""
Shape validation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class ShapeValidator:
    """Validator for tensor shapes.
    
    Validates that output shapes match expected shapes.
    """
    
    def validate(
        self,
        actual_shape: List[int],
        expected_shape: List[int],
    ) -> Tuple[bool, Optional[str]]:
        """Validate shape matches expected.
        
        Args:
            actual_shape: Actual output shape
            expected_shape: Expected shape
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def validate_broadcast(
        self,
        shapes: List[List[int]],
    ) -> Tuple[bool, Optional[str]]:
        """Validate shapes can broadcast.
        
        Args:
            shapes: Input shapes
            
        Returns:
            Tuple of (can_broadcast, error_message)
        """
        ...

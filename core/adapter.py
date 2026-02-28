"""
Core adapter interface for operator testing.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
import torch


class OperatorTestAdapter(ABC):
    """
    Abstract base class for operator test adapters.
    
    Each operator type should implement this interface to enable
    unified testing across different operators.
    """
    
    def __init__(self, name: str, operator_type: str):
        """
        Args:
            name: Name of the operator
            operator_type: Category (e.g., 'attention', 'normalization', 'activation')
        """
        self.name = name
        self.operator_type = operator_type
        self.test_history = []
    
    @abstractmethod
    def reference_implementation(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation using PyTorch native operations.
        This serves as the ground truth for correctness verification.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def test_implementation(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        The custom operator implementation to be tested.
        
        Args:
            *inputs: Input tensors
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def generate_test_cases(self) -> List[Tuple[torch.Tensor, ...]]:
        """
        Generate a list of test input cases.
        Should cover:
        - Standard cases
        - Boundary conditions
        - Edge cases
        
        Returns:
            List of input tensor tuples
        """
        pass
    
    @abstractmethod
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor, ...]) -> bool:
        """
        Verify operator-specific mathematical properties.
        
        Args:
            output: Output from test implementation
            inputs: Input tensors
            
        Returns:
            True if all properties hold
        """
        pass
    
    def run_full_test_suite(self, verbose: bool = True) -> dict:
        """
        Run the complete test suite for this operator.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with test results
        """
        results = {
            "operator": self.name,
            "type": self.operator_type,
            "tests": []
        }
        
        for i, test_input in enumerate(self.generate_test_cases()):
            test_result = {"case_id": i, "passed": True, "errors": []}
            
            try:
                # 1. Mathematical correctness
                ref_out = self.reference_implementation(*test_input)
                test_out = self.test_implementation(*test_input)
                
                if not torch.allclose(ref_out, test_out, rtol=1e-4, atol=1e-6):
                    test_result["passed"] = False
                    test_result["errors"].append("Mathematical correctness failed")
                
                # 2. Property verification
                if not self.verify_properties(test_out, test_input):
                    test_result["passed"] = False
                    test_result["errors"].append("Property verification failed")
                
                # 3. Gradient check (if inputs require grad)
                if any(inp.requires_grad for inp in test_input if isinstance(inp, torch.Tensor)):
                    from .gradient_check import verify_gradient
                    try:
                        verify_gradient(
                            lambda *x: self.test_implementation(*x),
                            test_input[0] if len(test_input) == 1 else test_input
                        )
                    except AssertionError as e:
                        test_result["passed"] = False
                        test_result["errors"].append(f"Gradient check failed: {e}")
                
            except Exception as e:
                test_result["passed"] = False
                test_result["errors"].append(f"Exception: {str(e)}")
            
            results["tests"].append(test_result)
            
            if verbose:
                status = "✓" if test_result["passed"] else "✗"
                print(f"Test case {i}: {status}")
                if not test_result["passed"]:
                    for error in test_result["errors"]:
                        print(f"  - {error}")
        
        results["total"] = len(results["tests"])
        results["passed"] = sum(1 for t in results["tests"] if t["passed"])
        results["failed"] = results["total"] - results["passed"]
        
        return results

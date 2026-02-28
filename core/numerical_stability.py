"""
Numerical stability testing for operators.
"""

import torch
import math
from typing import List, Dict, Callable, Optional


class NumericalStabilityTester:
    """
    Test numerical stability of operators under various extreme conditions.
    """
    
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Dict]:
        """Generate standard numerical stability test cases."""
        cases = [
            {
                "name": "large_values",
                "description": "Very large values (near FP32 max)",
                "generator": lambda shape: torch.full(shape, 1e10),
                "check_finite": True
            },
            {
                "name": "small_values", 
                "description": "Very small values (near FP32 min)",
                "generator": lambda shape: torch.full(shape, 1e-10),
                "check_finite": True
            },
            {
                "name": "mixed_magnitudes",
                "description": "Mixed large and small values",
                "generator": lambda shape: self._generate_mixed(shape),
                "check_finite": True
            },
            {
                "name": "near_zero_variance",
                "description": "Values with very small variance",
                "generator": lambda shape: torch.ones(shape) + torch.randn(shape) * 1e-7,
                "check_finite": True
            },
            {
                "name": "nan_input",
                "description": "Input containing NaN",
                "generator": lambda shape: self._insert_special(shape, float('nan')),
                "check_nan_propagation": True
            },
            {
                "name": "inf_input",
                "description": "Input containing Inf",
                "generator": lambda shape: self._insert_special(shape, float('inf')),
                "check_inf_propagation": True
            },
            {
                "name": "zeros",
                "description": "All zeros",
                "generator": lambda shape: torch.zeros(shape),
                "check_finite": True
            },
            {
                "name": "random_normal",
                "description": "Standard normal distribution",
                "generator": lambda shape: torch.randn(shape),
                "check_finite": True
            }
        ]
        return cases
    
    def _generate_mixed(self, shape: tuple) -> torch.Tensor:
        """Generate tensor with mixed magnitudes."""
        result = torch.zeros(shape)
        # Fill different regions with different magnitudes
        result[..., ::4] = 1e8
        result[..., 1::4] = 1.0
        result[..., 2::4] = 1e-8
        result[..., 3::4] = torch.randn(shape[:-1] + (shape[-1] // 4,)) * 0.1
        return result
    
    def _insert_special(self, shape: tuple, value: float) -> torch.Tensor:
        """Insert special value (NaN or Inf) into tensor."""
        result = torch.randn(shape)
        # Insert at random positions
        flat_idx = torch.randint(0, result.numel(), (max(1, result.numel() // 100),))
        result.view(-1)[flat_idx] = value
        return result
    
    def test_operator(
        self,
        operator: Callable,
        input_shape: tuple,
        expected_behavior: Optional[Dict] = None
    ) -> Dict:
        """
        Test operator numerical stability.
        
        Args:
            operator: Operator function to test
            input_shape: Shape of input tensor
            expected_behavior: Expected behavior for each test case
            
        Returns:
            Dictionary with test results
        """
        results = {
            "operator": operator.__name__ if hasattr(operator, '__name__') else str(operator),
            "input_shape": input_shape,
            "test_cases": []
        }
        
        for case in self.test_cases:
            case_result = {"name": case["name"], "passed": True, "errors": []}
            
            try:
                # Generate test input
                test_input = case["generator"](input_shape).to(self.dtype)
                
                # Run operator
                output = operator(test_input)
                
                # Check conditions
                if case.get("check_finite"):
                    if torch.isnan(output).any():
                        case_result["passed"] = False
                        case_result["errors"].append("Output contains NaN")
                    if torch.isinf(output).any():
                        case_result["passed"] = False
                        case_result["errors"].append("Output contains Inf")
                
                if case.get("check_nan_propagation"):
                    # NaN should propagate or be handled explicitly
                    nan_in_input = torch.isnan(test_input).any()
                    nan_in_output = torch.isnan(output).any()
                    # Either no NaN in output (handled) or NaN propagates correctly
                    if nan_in_input and not nan_in_output:
                        case_result["notes"] = "NaN was handled (not propagated)"
                
                if case.get("check_inf_propagation"):
                    inf_in_input = torch.isinf(test_input).any()
                    inf_in_output = torch.isinf(output).any()
                    if inf_in_input and not inf_in_output:
                        case_result["notes"] = "Inf was handled (not propagated)"
                
                # Check output range for specific cases
                if case["name"] == "zeros":
                    # Many operators should preserve zeros or handle them gracefully
                    pass  # Operator-specific checks can be added
                
            except Exception as e:
                case_result["passed"] = False
                case_result["errors"].append(f"Exception: {str(e)}")
            
            results["test_cases"].append(case_result)
        
        results["total"] = len(results["test_cases"])
        results["passed"] = sum(1 for t in results["test_cases"] if t["passed"])
        results["failed"] = results["total"] - results["passed"]
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print test results."""
        print(f"\nNumerical Stability Test Results for {results['operator']}")
        print(f"Input shape: {results['input_shape']}")
        print(f"Passed: {results['passed']}/{results['total']}")
        print("-" * 60)
        
        for case in results["test_cases"]:
            status = "✓" if case["passed"] else "✗"
            print(f"{status} {case['name']}")
            if not case["passed"]:
                for error in case["errors"]:
                    print(f"    Error: {error}")
            if "notes" in case:
                print(f"    Note: {case['notes']}")

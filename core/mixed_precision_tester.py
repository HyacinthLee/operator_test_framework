"""
Mixed precision testing for operators.
Tests FP16/BF16 numerical stability and equivalence to FP32.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable
import numpy as np


class MixedPrecisionTester:
    """
    Test operator behavior under mixed precision (FP16/BF16).
    
    Key checks:
    - Numerical equivalence between FP32 and low precision
    - Gradient scaling behavior
    - Loss of significance detection
    """
    
    def __init__(self, rtol: float = 1e-3, atol: float = 1e-5):
        """
        Args:
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
        """
        self.rtol = rtol
        self.atol = atol
        self.tested_dtypes = [torch.float16, torch.bfloat16]
    
    def test_numerical_equivalence(
        self,
        operator: Callable,
        inputs_fp32: Tuple[torch.Tensor, ...],
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, any]:
        """
        Test if low precision operator matches FP32 reference.
        
        Args:
            operator: Operator function to test
            inputs_fp32: FP32 input tensors
            dtype: Target low precision dtype (FP16 or BF16)
            
        Returns:
            Test results dictionary
        """
        # Convert inputs to low precision
        inputs_low = tuple(
            inp.to(dtype) if isinstance(inp, torch.Tensor) else inp
            for inp in inputs_fp32
        )
        
        # Run operator in both precisions
        with torch.no_grad():
            output_fp32 = operator(*inputs_fp32)
            output_low = operator(*inputs_low)
        
        # Convert low precision output to FP32 for comparison
        output_low_as_fp32 = output_low.float()
        
        # Compute differences
        abs_diff = (output_fp32 - output_low_as_fp32).abs()
        rel_diff = abs_diff / (output_fp32.abs() + 1e-8)
        
        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        mean_rel_error = rel_diff.mean().item()
        
        # Check if within tolerance
        passed = torch.allclose(
            output_fp32, output_low_as_fp32,
            rtol=self.rtol, atol=self.atol
        )
        
        return {
            "dtype": str(dtype),
            "passed": passed,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "mean_abs_error": mean_abs_error,
            "mean_rel_error": mean_rel_error,
            "output_fp32_shape": output_fp32.shape,
            "output_low_shape": output_low.shape,
        }
    
    def test_gradient_equivalence(
        self,
        operator: Callable,
        input_shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, any]:
        """
        Test if gradients match between FP32 and low precision.
        
        This is critical for training stability.
        """
        # Create test input
        x_fp32 = torch.randn(input_shape, requires_grad=True)
        x_low = x_fp32.detach().to(dtype).requires_grad_(True)
        
        # Forward + backward in FP32
        out_fp32 = operator(x_fp32)
        loss_fp32 = out_fp32.sum()
        loss_fp32.backward()
        grad_fp32 = x_fp32.grad.clone()
        
        # Forward + backward in low precision
        out_low = operator(x_low)
        loss_low = out_low.sum()
        loss_low.backward()
        grad_low = x_low.grad.float()  # Convert to FP32 for comparison
        
        # Compare gradients
        abs_diff = (grad_fp32 - grad_low).abs()
        rel_diff = abs_diff / (grad_fp32.abs() + 1e-8)
        
        passed = torch.allclose(grad_fp32, grad_low, rtol=self.rtol, atol=self.atol)
        
        return {
            "dtype": str(dtype),
            "passed": passed,
            "max_grad_error": abs_diff.max().item(),
            "mean_grad_error": abs_diff.mean().item(),
            "grad_fp32_norm": grad_fp32.norm().item(),
            "grad_low_norm": grad_low.norm().item(),
        }
    
    def test_loss_scaling_behavior(
        self,
        operator: Callable,
        input_shape: Tuple[int, ...],
        loss_scaler: torch.cuda.amp.GradScaler = None
    ) -> Dict[str, any]:
        """
        Test operator behavior with loss scaling (for FP16 training).
        
        Checks if gradients can be properly scaled to prevent underflow.
        """
        if not torch.cuda.is_available():
            return {
                "skipped": True,
                "reason": "CUDA not available",
            }
        
        if loss_scaler is None:
            loss_scaler = torch.cuda.amp.GradScaler()
        
        x = torch.randn(input_shape, requires_grad=True).cuda().half()
        
        # Forward with autocast
        with torch.cuda.amp.autocast():
            output = operator(x)
            loss = output.sum()
        
        # Backward with scaling
        scaled_loss = loss_scaler.scale(loss)
        scaled_loss.backward()
        
        # Check gradient stats
        grad = x.grad
        grad_min = grad.min().item()
        grad_max = grad.max().item()
        grad_mean = grad.mean().item()
        
        # Check for underflow (gradients too small)
        underflow_ratio = (grad.abs() < 1e-7).float().mean().item()
        
        return {
            "loss_scale": loss_scaler.get_scale(),
            "grad_min": grad_min,
            "grad_max": grad_max,
            "grad_mean": grad_mean,
            "underflow_ratio": underflow_ratio,
            "has_underflow": underflow_ratio > 0.5,
        }
    
    def run_full_precision_test_suite(
        self,
        operator: Callable,
        input_shape: Tuple[int, ...],
        verbose: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Run complete mixed precision test suite.
        """
        results = {
            "operator": operator.__name__ if hasattr(operator, '__name__') else str(operator),
            "input_shape": input_shape,
            "numerical_tests": [],
            "gradient_tests": [],
        }
        
        # Generate FP32 test input
        x_fp32 = torch.randn(input_shape)
        
        for dtype in self.tested_dtypes:
            if verbose:
                print(f"\nTesting {dtype}...")
            
            # Numerical equivalence
            num_result = self.test_numerical_equivalence(
                operator, (x_fp32,), dtype
            )
            results["numerical_tests"].append(num_result)
            
            if verbose:
                status = "✓" if num_result["passed"] else "✗"
                print(f"  Numerical: {status} (max rel error: {num_result['max_rel_error']:.2e})")
            
            # Gradient equivalence
            grad_result = self.test_gradient_equivalence(
                operator, input_shape, dtype
            )
            results["gradient_tests"].append(grad_result)
            
            if verbose:
                status = "✓" if grad_result["passed"] else "✗"
                print(f"  Gradient: {status} (max error: {grad_result['max_grad_error']:.2e})")
        
        # Summary
        all_passed = all(
            t["passed"] for t in results["numerical_tests"] + results["gradient_tests"]
        )
        results["all_passed"] = all_passed
        
        return results


# Example usage
if __name__ == "__main__":
    # Test LayerNorm with mixed precision
    tester = MixedPrecisionTester()
    
    layernorm = torch.nn.LayerNorm(64)
    results = tester.run_full_precision_test_suite(
        layernorm,
        input_shape=(4, 64),
        verbose=True
    )
    
    print(f"\nOverall: {'PASSED' if results['all_passed'] else 'FAILED'}")

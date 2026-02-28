"""
Test utilities for operator testing.

This module provides utility functions for gradient checking, tensor comparison,
benchmarking, and fuzz testing.
"""

import torch
import numpy as np
import time
from typing import Callable, Union, Tuple, Optional, Dict, Any


def gradient_check(
    func: Callable,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    rtol: float = 1e-4,
    atol: float = 1e-6,
    eps: float = 1e-5
) -> bool:
    """
    Check if analytical gradient matches numerical gradient.
    
    Args:
        func: Function to test
        inputs: Input tensor(s) with requires_grad=True
        rtol: Relative tolerance
        atol: Absolute tolerance
        eps: Finite difference epsilon
        
    Returns:
        True if gradients match, False otherwise
    """
    try:
        from .core.gradient_check import verify_gradient
        verify_gradient(func, inputs, rtol=rtol, atol=atol, eps=eps)
        return True
    except AssertionError:
        return False


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare two tensors and return detailed comparison results.
    
    Args:
        actual: Actual tensor
        expected: Expected tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary with comparison results
    """
    if actual.shape != expected.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: {actual.shape} vs {expected.shape}"
        }
    
    if actual.dtype != expected.dtype:
        return {
            "match": False,
            "error": f"Dtype mismatch: {actual.dtype} vs {expected.dtype}"
        }
    
    diff = (actual - expected).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    match = torch.allclose(actual, expected, rtol=rtol, atol=atol)
    
    return {
        "match": match,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rtol": rtol,
        "atol": atol
    }


def benchmark_performance(
    func: Callable,
    *inputs: torch.Tensor,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark operator performance.
    
    Args:
        func: Function to benchmark
        *inputs: Input tensors
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
        device: Device to run on ('cuda' or 'cpu')
        
    Returns:
        Dictionary with performance metrics
    """
    from .core.performance_benchmark import PerformanceBenchmark
    
    benchmark = PerformanceBenchmark(
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        device=device
    )
    
    return benchmark.measure_latency(func, *inputs)


def fuzz_test(
    func: Callable,
    input_shape: Tuple[int, ...],
    num_trials: int = 100,
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Fuzz test an operator with random inputs.
    
    Args:
        func: Function to test
        input_shape: Shape of input tensors
        num_trials: Number of random trials
        dtype: Data type for inputs
        device: Device to run on
        
    Returns:
        Dictionary with fuzz testing results
    """
    results = {
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    for i in range(num_trials):
        # Generate random input
        x = torch.randn(input_shape, dtype=dtype, device=device)
        
        try:
            output = func(x)
            
            # Check for NaN or Inf
            if torch.isnan(output).any():
                results["failed"] += 1
                results["errors"].append(f"Trial {i}: NaN detected in output")
            elif torch.isinf(output).any():
                results["failed"] += 1
                results["errors"].append(f"Trial {i}: Inf detected in output")
            else:
                results["passed"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"Trial {i}: {str(e)}")
    
    results["pass_rate"] = results["passed"] / num_trials
    return results

"""
Core module initialization.
"""

from .adapter import OperatorTestAdapter
from .gradient_check import numerical_jacobian, verify_gradient, check_gradient_numerical_stability
from .numerical_stability import NumericalStabilityTester
from .performance_benchmark import PerformanceBenchmark, profile_with_pytorch_profiler

__all__ = [
    "OperatorTestAdapter",
    "numerical_jacobian",
    "verify_gradient",
    "check_gradient_numerical_stability",
    "NumericalStabilityTester",
    "PerformanceBenchmark",
    "profile_with_pytorch_profiler",
]

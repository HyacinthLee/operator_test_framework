"""
Deep Learning Operator Test Framework

A comprehensive testing framework for validating deep learning operators
with four-layer validation: mathematical, numerical, functional, and performance.
"""

from .adapter import OperatorTestAdapter, TestConfig
from .validators import (
    MathematicalValidator,
    NumericalValidator,
    FunctionalValidator,
    PerformanceValidator,
)
from .test_utils import (
    gradient_check,
    fuzz_test,
    benchmark_performance,
    compare_tensors,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "OperatorTestAdapter",
    "TestConfig",
    # Validators
    "MathematicalValidator",
    "NumericalValidator",
    "FunctionalValidator",
    "PerformanceValidator",
    # Utilities
    "gradient_check",
    "fuzz_test",
    "benchmark_performance",
    "compare_tensors",
]
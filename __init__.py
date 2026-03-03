"""
Operator Test Framework
A unified testing framework for deep learning operators.
"""

__version__ = "0.1.0"
__author__ = "Researcher Agent"

# 使用绝对导入避免相对导入问题
from core.adapter import OperatorTestAdapter
from core.gradient_check import numerical_jacobian, verify_gradient
from core.numerical_stability import NumericalStabilityTester
from core.performance_benchmark import PerformanceBenchmark

__all__ = [
    "OperatorTestAdapter",
    "numerical_jacobian",
    "verify_gradient",
    "NumericalStabilityTester",
    "PerformanceBenchmark",
]

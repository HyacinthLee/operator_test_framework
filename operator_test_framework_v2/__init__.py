"""
Operator Test Framework v2 - ATTest-based Testing Framework

A next-generation testing framework for deep learning operators,
based on the ATTest paper's seven-stage workflow and agent-driven architecture.

Architecture:
    - Seven-stage workflow: Understand → Requirements → Planning → 
      Generation → Execution → Analysis → Reporting
    - Agent-driven: LLM-based autonomous engineering agents
    - Iterative repair: Generation-validation-repair loop
    - Constraint extraction: API-based tensor constraint inference

Example:
    from operator_test_framework_v2 import TestFramework
    
    framework = TestFramework()
    results = framework.test_operator(
        operator_name="torch.nn.functional.softmax",
        implementation=my_softmax_impl
    )
"""

__version__ = "2.0.0"
__author__ = "Researcher Agent"

from .core.framework import TestFramework
from .core.models.operator_spec import OperatorSpec
from .core.models.tensor_constraint import TensorConstraint
from .core.models.test_case import TestCase, TestOracle

__all__ = [
    "TestFramework",
    "OperatorSpec",
    "TensorConstraint", 
    "TestCase",
    "TestOracle",
]

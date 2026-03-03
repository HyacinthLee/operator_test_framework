"""
Data models for operator testing.

This module provides type definitions for:
    - OperatorSpec: Operator specification and metadata
    - TensorConstraint: Tensor shape, dtype, and device constraints
    - TestCase: Individual test case with inputs and expected outputs
    - TestOracle: Pass/fail criteria for test validation
"""

from .operator_spec import OperatorSpec, OperatorAttribute, InputSpec, OutputSpec
from .tensor_constraint import (
    TensorConstraint,
    ShapeConstraint,
    DtypeConstraint,
    DeviceConstraint,
    ValueConstraint,
)
from .test_case import TestCase, TestOracle, TestResult, TestStatus

__all__ = [
    "OperatorSpec",
    "OperatorAttribute",
    "InputSpec",
    "OutputSpec",
    "TensorConstraint",
    "ShapeConstraint",
    "DtypeConstraint",
    "DeviceConstraint",
    "ValueConstraint",
    "TestCase",
    "TestOracle",
    "TestResult",
    "TestStatus",
]

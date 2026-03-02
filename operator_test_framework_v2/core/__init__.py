"""
Core module for operator test framework v2.

This module contains the fundamental components:
    - models: Data models for operators, constraints, and test cases
    - workflow: Seven-stage testing workflow implementation
    - agents: LLM-based autonomous testing agents
    - generators: Test case generation strategies
    - validators: Result validation and oracles
"""

from .framework import TestFramework
from .config import FrameworkConfig

__all__ = ["TestFramework", "FrameworkConfig"]

"""
Seven-stage testing workflow implementation.

The ATTest workflow consists of:
1. Understand: Parse operator and extract constraints
2. Requirements: Generate test requirements
3. Planning: Design test strategy
4. Generation: Generate test cases
5. Execution: Execute tests
6. Analysis: Analyze results
7. Report: Generate reports
"""

from .base import WorkflowStage, WorkflowContext, WorkflowResult
from .understand import UnderstandStage
from .requirements import RequirementsStage
from .planning import PlanningStage
from .generation import GenerationStage
from .execution import ExecutionStage
from .analysis import AnalysisStage
from .report import ReportStage

__all__ = [
    "WorkflowStage",
    "WorkflowContext",
    "WorkflowResult",
    "UnderstandStage",
    "RequirementsStage",
    "PlanningStage",
    "GenerationStage",
    "ExecutionStage",
    "AnalysisStage",
    "ReportStage",
]

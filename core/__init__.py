"""
Core module initialization.
"""

from .adapter import OperatorTestAdapter
from .gradient_check import numerical_jacobian, verify_gradient, check_gradient_numerical_stability
from .numerical_stability import NumericalStabilityTester
from .performance_benchmark import PerformanceBenchmark, profile_with_pytorch_profiler

# 导出七阶段工作流引擎
from .workflow import (
    # Data Models
    OperatorSignature,
    TensorConstraint,
    TestRequirement,
    TestPlan,
    TestCase,
    TestResult,
    FailureAnalysis,
    TestReport,
    WorkflowState,
    TestType,
    GenerationStrategy,
    FailureCategory,
    # Phase Classes
    UnderstandPhase,
    RequirementsPhase,
    PlanningPhase,
    GenerationPhase,
    ExecutionPhase,
    AnalysisPhase,
    ReportPhase,
    # Engine
    WorkflowEngine,
    WorkflowConfig,
    PhaseExecutor,
    IncrementalWorkflowEngine,
    ParallelWorkflowEngine,
    create_default_engine,
    run_full_workflow,
)

__all__ = [
    "OperatorTestAdapter",
    "numerical_jacobian",
    "verify_gradient",
    "check_gradient_numerical_stability",
    "NumericalStabilityTester",
    "PerformanceBenchmark",
    "profile_with_pytorch_profiler",
    # Workflow Engine
    "OperatorSignature",
    "TensorConstraint",
    "TestRequirement",
    "TestPlan",
    "TestCase",
    "TestResult",
    "FailureAnalysis",
    "TestReport",
    "WorkflowState",
    "TestType",
    "GenerationStrategy",
    "FailureCategory",
    "UnderstandPhase",
    "RequirementsPhase",
    "PlanningPhase",
    "GenerationPhase",
    "ExecutionPhase",
    "AnalysisPhase",
    "ReportPhase",
    "WorkflowEngine",
    "WorkflowConfig",
    "PhaseExecutor",
    "IncrementalWorkflowEngine",
    "ParallelWorkflowEngine",
    "create_default_engine",
    "run_full_workflow",
]

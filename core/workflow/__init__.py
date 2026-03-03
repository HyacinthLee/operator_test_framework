"""
七阶段工作流引擎 - ATTest 思想实现

该模块实现基于 ATTest 思想的七阶段测试工作流：
1. 算子理解 (Understand)
2. 需求生成 (Requirements)
3. 测试计划 (Planning)
4. 测试生成 (Generation)
5. 执行测试 (Execution)
6. 结果分析 (Analysis)
7. 报告生成 (Report)
"""

from .data_models import (
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
)

from .understand import UnderstandPhase, SignatureParser, ConstraintExtractor
from .requirements import RequirementsPhase, TestRequirementGenerator
from .planning import PlanningPhase, TestPlanner, StrategySelector
from .generation import GenerationPhase, InputGenerator, ConstraintSatisfier
from .execution import ExecutionPhase, TestExecutor, ResultCollector
from .analysis import AnalysisPhase, FailureAnalyzer, FixSuggester
from .report import ReportPhase, ReportGenerator, CoverageCalculator
from .engine import (
    WorkflowEngine,
    PhaseExecutor,
    WorkflowConfig,
    IncrementalWorkflowEngine,
    ParallelWorkflowEngine,
    create_default_engine,
    run_full_workflow,
)

__all__ = [
    # Data Models
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
    # Phase Classes
    "UnderstandPhase",
    "SignatureParser",
    "ConstraintExtractor",
    "RequirementsPhase",
    "TestRequirementGenerator",
    "PlanningPhase",
    "TestPlanner",
    "StrategySelector",
    "GenerationPhase",
    "InputGenerator",
    "ConstraintSatisfier",
    "ExecutionPhase",
    "TestExecutor",
    "ResultCollector",
    "AnalysisPhase",
    "FailureAnalyzer",
    "FixSuggester",
    "ReportPhase",
    "ReportGenerator",
    "CoverageCalculator",
    # Engine
    "WorkflowEngine",
    "PhaseExecutor",
    "WorkflowConfig",
    "IncrementalWorkflowEngine",
    "ParallelWorkflowEngine",
    "create_default_engine",
    "run_full_workflow",
]

"""
Base classes for workflow stages.

Defines the interface for all seven workflow stages and shared
context/result types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
from enum import Enum, auto


class StageStatus(Enum):
    """Status of a workflow stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class WorkflowContext:
    """Shared context across all workflow stages.
    
    The context maintains state throughout the seven-stage workflow,
    allowing each stage to access outputs from previous stages.
    
    Attributes:
        operator_name: Target operator name
        operator_spec: Parsed operator specification
        test_requirements: Generated test requirements
        test_plan: Designed test plan
        test_suite: Generated test cases
        test_results: Execution results
        analysis_result: Analysis output
        report: Final report
        metadata: Additional shared data
        errors: Collected errors
    """
    operator_name: Optional[str] = None
    operator_spec: Optional["OperatorSpec"] = None
    test_requirements: Optional["TestRequirements"] = None
    test_plan: Optional["TestPlan"] = None
    test_suite: Optional["TestSuite"] = None
    test_results: Optional["TestResults"] = None
    analysis_result: Optional["AnalysisResult"] = None
    report: Optional["TestReport"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def set_error(self, stage: str, error: str) -> None:
        """Record an error from a stage."""
        ...
    
    def get_stage_output(self, stage: str) -> Optional[Any]:
        """Get output from a specific stage."""
        ...


@dataclass
class StageResult:
    """Result of a single workflow stage.
    
    Attributes:
        stage_name: Name of the stage
        status: Execution status
        output: Stage output data
        execution_time_ms: Execution time
        error_message: Error if failed
        timestamp: Completion timestamp
    """
    stage_name: str
    status: StageStatus
    output: Any = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success(self) -> bool:
        """Check if stage completed successfully."""
        return self.status == StageStatus.COMPLETED


@dataclass
class WorkflowResult:
    """Result of the complete workflow.
    
    Attributes:
        context: Final workflow context
        stage_results: Results from each stage
        overall_success: Whether all stages succeeded
        total_execution_time_ms: Total execution time
        timestamp: Completion timestamp
    """
    context: WorkflowContext
    stage_results: List[StageResult] = field(default_factory=list)
    overall_success: bool = False
    total_execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for a specific stage."""
        ...


class WorkflowStage(ABC):
    """Abstract base class for workflow stages.
    
    All seven stages (Understand, Requirements, Planning, Generation,
    Execution, Analysis, Report) inherit from this class.
    
    Example:
        >>> class MyStage(WorkflowStage):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_stage"
        ...     
        ...     def execute(self, context: WorkflowContext) -> StageResult:
        ...         # Stage implementation
        ...         return StageResult(...)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name."""
        ...
    
    @property
    def description(self) -> str:
        """Return the stage description."""
        return f"{self.name} stage"
    
    @abstractmethod
    def execute(self, context: WorkflowContext) -> StageResult:
        """Execute this workflow stage.
        
        Args:
            context: Shared workflow context
            
        Returns:
            Stage execution result
        """
        ...
    
    def can_execute(self, context: WorkflowContext) -> tuple[bool, Optional[str]]:
        """Check if this stage can be executed.
        
        Args:
            context: Current workflow context
            
        Returns:
            Tuple of (can_execute, reason_if_not)
        """
        return True, None
    
    def validate_output(self, output: Any) -> tuple[bool, Optional[str]]:
        """Validate stage output.
        
        Args:
            output: Stage output to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None

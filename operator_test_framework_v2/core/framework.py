"""
Main test framework orchestrating the seven-stage workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

from .config import FrameworkConfig
from .models import OperatorSpec, TestSuite, TestResult, TestReport
from .workflow import (
    WorkflowStage,
    WorkflowContext,
    WorkflowResult,
    UnderstandStage,
    RequirementsStage,
    PlanningStage,
    GenerationStage,
    ExecutionStage,
    AnalysisStage,
    ReportStage,
)


class TestFramework:
    """Main test framework orchestrating the seven-stage workflow.
    
    This is the primary entry point for operator testing. It coordinates
    the seven stages of the ATTest workflow:
    1. Understand: Parse operator API
    2. Requirements: Generate test requirements
    3. Planning: Design test strategy
    4. Generation: Generate test cases
    5. Execution: Execute tests
    6. Analysis: Analyze results
    7. Report: Generate reports
    
    The framework supports both full workflow execution and
    individual stage execution for flexibility.
    
    Example:
        >>> framework = TestFramework(config)
        >>> results = framework.test_operator(
        ...     operator_name="torch.nn.functional.softmax",
        ...     implementation=my_softmax,
        ...     reference_impl=torch.softmax
        ... )
        >>> print(results.report.summary)
    """
    
    def __init__(
        self,
        config: Optional[FrameworkConfig] = None,
        stages: Optional[Dict[str, WorkflowStage]] = None,
        llm_client: Optional["LLMClient"] = None,
    ):
        """Initialize the test framework.
        
        Args:
            config: Framework configuration
            stages: Custom workflow stages (uses defaults if None)
            llm_client: LLM client for agent-based stages
        """
        ...
    
    def test_operator(
        self,
        operator_name: str,
        implementation: Optional[Callable[..., Any]] = None,
        reference_impl: Optional[Callable[..., Any]] = None,
        operator_spec: Optional[OperatorSpec] = None,
        custom_stages: Optional[List[str]] = None,
    ) -> WorkflowResult:
        """Run complete testing workflow for an operator.
        
        Args:
            operator_name: Name of the operator to test
            implementation: Implementation to test
            reference_impl: Reference implementation for comparison
            operator_spec: Pre-built operator spec (optional)
            custom_stages: List of stage names to run (default: all)
            
        Returns:
            Complete workflow result
        """
        ...
    
    def run_stage(
        self,
        stage_name: str,
        context: WorkflowContext,
    ) -> WorkflowContext:
        """Run a single workflow stage.
        
        Args:
            stage_name: Name of stage to run
            context: Current workflow context
            
        Returns:
            Updated context
        """
        ...
    
    def create_context(
        self,
        operator_name: str,
        implementation: Optional[Callable] = None,
        reference_impl: Optional[Callable] = None,
    ) -> WorkflowContext:
        """Create initial workflow context.
        
        Args:
            operator_name: Operator name
            implementation: Implementation to test
            reference_impl: Reference implementation
            
        Returns:
            Initialized workflow context
        """
        ...
    
    def register_stage(
        self,
        name: str,
        stage: WorkflowStage,
    ) -> None:
        """Register a custom workflow stage.
        
        Args:
            name: Stage name
            stage: Stage instance
        """
        ...
    
    def enable_iterative_repair(
        self,
        max_iterations: int = 5,
        success_threshold: float = 0.95,
    ) -> None:
        """Enable iterative repair loop.
        
        Args:
            max_iterations: Maximum repair iterations
            success_threshold: Target success rate
        """
        ...
    
    def get_default_stages(
        self,
        llm_client: Optional["LLMClient"] = None,
    ) -> Dict[str, WorkflowStage]:
        """Get default workflow stage implementations.
        
        Args:
            llm_client: LLM client for agent stages
            
        Returns:
            Dictionary of stage name to stage instance
        """
        ...


@dataclass
class FrameworkState:
    """Persistent state for the framework.
    
    Attributes:
        cached_specs: Cached operator specifications
        historical_patterns: Historical failure patterns
        test_history: Previous test results
        config: Framework configuration
    """
    cached_specs: Dict[str, OperatorSpec] = field(default_factory=dict)
    historical_patterns: Dict[str, List[Dict]] = field(default_factory=dict)
    test_history: List[WorkflowResult] = field(default_factory=list)
    config: FrameworkConfig = field(default_factory=FrameworkConfig)
    
    def cache_spec(self, name: str, spec: OperatorSpec) -> None:
        """Cache an operator specification."""
        ...
    
    def get_cached_spec(self, name: str) -> Optional[OperatorSpec]:
        """Get cached operator specification."""
        ...
    
    def save(self, path: Union[str, Path]) -> None:
        """Save state to file."""
        ...
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> FrameworkState:
        """Load state from file."""
        ...


class BatchTestRunner:
    """Runner for batch testing multiple operators."""
    
    def __init__(self, framework: TestFramework):
        """Initialize batch runner.
        
        Args:
            framework: Test framework instance
        """
        ...
    
    def run_batch(
        self,
        operators: List[Dict[str, Any]],
        parallel: bool = False,
        max_workers: int = 4,
    ) -> List[WorkflowResult]:
        """Run tests for multiple operators.
        
        Args:
            operators: List of operator configurations
            parallel: Whether to run in parallel
            max_workers: Number of parallel workers
            
        Returns:
            List of workflow results
        """
        ...
    
    def generate_comparison_report(
        self,
        results: List[WorkflowResult],
    ) -> TestReport:
        """Generate comparison report for batch results.
        
        Args:
            results: Batch results
            
        Returns:
            Comparison report
        """
        ...

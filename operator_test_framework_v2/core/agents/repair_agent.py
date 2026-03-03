"""
Agent for repairing failed tests and implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing_extensions import override
from enum import Enum, auto

from .base import BaseAgent, AgentContext, AgentAction, AgentActionType


class RepairTarget(Enum):
    """What to repair."""
    TEST_CASE = auto()
    IMPLEMENTATION = auto()
    CONSTRAINT = auto()
    ORACLE = auto()


@dataclass
class RepairPrompt:
    """Prompt templates for test repair."""
    
    SYSTEM_PROMPT: str = """You are an expert in debugging and repairing deep learning code.
Your task is to analyze test failures and propose fixes.

Analyze the failure carefully:
1. Identify the root cause
2. Determine what needs to be fixed
3. Propose a minimal, correct fix
4. Verify the fix resolves the issue

Be precise and provide clear explanations."""

    TEST_REPAIR_PROMPT: str = """Repair the following test case:

Test Case: {test_case}
Failure: {failure}
Error: {error}
Operator Spec: {operator_spec}

The test case failed because: {failure_reason}

Propose a repaired test case that:
1. Satisfies the operator constraints
2. Correctly tests the intended behavior
3. Has correct expected outputs

Output the repaired test case in the same format."""

    IMPLEMENTATION_REPAIR_PROMPT: str = """Analyze and fix the implementation:

Failed Test: {test_case}
Failure: {failure}
Stack Trace: {stack_trace}
Implementation: {implementation}

Identify the bug and propose a fix.

Output:
1. Root cause analysis
2. Proposed fix (code)
3. Explanation of changes"""


class RepairAgent(BaseAgent):
    """Agent for repairing failed tests and implementations.
    
    This agent analyzes test failures and generates repairs for:
    - Invalid test cases (constraint violations, incorrect expectations)
    - Implementation bugs
    - Overly strict oracles
    - Incorrect constraints
    
    It uses failure patterns and root cause analysis to determine
    the appropriate fix.
    
    Example:
        >>> agent = RepairAgent(llm_client)
        >>> context = AgentContext()
        >>> context.remember("test_result", failed_result)
        >>> result = agent.execute(context)
        >>> repair = result["repair_suggestion"]
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        max_repair_attempts: int = 3,
        verify_repairs: bool = True,
    ):
        """Initialize repair agent.
        
        Args:
            llm_client: LLM client
            max_repair_attempts: Maximum repair attempts per failure
            verify_repairs: Whether to verify proposed repairs
        """
        ...
    
    @override
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """Generate repair for failure.
        
        Args:
            context: Context with test_result or failure information
            
        Returns:
            Dictionary with repair suggestion
        """
        ...
    
    def diagnose_failure(
        self,
        test_result: "TestResult",
        operator_spec: "OperatorSpec",
    ) -> Dict[str, Any]:
        """Diagnose the cause of a test failure.
        
        Args:
            test_result: Failed test result
            operator_spec: Operator specification
            
        Returns:
            Diagnosis with failure type and root cause
        """
        ...
    
    def repair_test_case(
        self,
        test_case: "TestCase",
        failure: "OracleResult",
        operator_spec: "OperatorSpec",
    ) -> Optional["TestCase"]:
        """Repair an invalid test case.
        
        Args:
            test_case: Failed test case
            failure: Oracle failure result
            operator_spec: Operator specification
            
        Returns:
            Repaired test case or None
        """
        ...
    
    def repair_implementation(
        self,
        implementation: str,
        test_failure: "TestResult",
        operator_spec: "OperatorSpec",
    ) -> Optional[str]:
        """Propose a fix for implementation bugs.
        
        Args:
            implementation: Implementation code
            test_failure: Test failure information
            operator_spec: Operator specification
            
        Returns:
            Proposed fix or None
        """
        ...
    
    def adjust_oracle(
        self,
        oracle: "TestOracle",
        failure: "OracleResult",
        test_case: "TestCase",
    ) -> Optional["TestOracle"]:
        """Adjust oracle tolerances or criteria.
        
        Args:
            oracle: Current oracle
            failure: Oracle failure
            test_case: Test case
            
        Returns:
            Adjusted oracle or None
        """
        ...
    
    def verify_repair(
        self,
        original: Any,
        repaired: Any,
        repair_type: RepairTarget,
        test_context: Dict[str, Any],
    ) -> bool:
        """Verify that a repair is correct.
        
        Args:
            original: Original (failed) object
            repaired: Proposed repair
            repair_type: Type of repair
            test_context: Testing context
            
        Returns:
            Whether repair is verified
        """
        ...
    
    def suggest_constraint_relaxation(
        self,
        constraint: "TensorConstraint",
        violations: List[str],
    ) -> Optional["TensorConstraint"]:
        """Suggest relaxing a constraint that is too strict.
        
        Args:
            constraint: Current constraint
            violations: Violation descriptions
            
        Returns:
            Relaxed constraint or None
        """
        ...


class IterativeRepairLoop:
    """Iterative repair loop for generation-validation-repair cycle."""
    
    def __init__(
        self,
        repair_agent: RepairAgent,
        max_iterations: int = 5,
        success_threshold: float = 0.95,
    ):
        """Initialize repair loop.
        
        Args:
            repair_agent: Repair agent
            max_iterations: Maximum iterations
            success_threshold: Target success rate
        """
        ...
    
    def run(
        self,
        test_cases: List["TestCase"],
        test_executor: Callable,
        operator_spec: "OperatorSpec",
    ) -> Dict[str, Any]:
        """Run iterative repair loop.
        
        Args:
            test_cases: Initial test cases
            test_executor: Function to execute tests
            operator_spec: Operator specification
            
        Returns:
            Final results with repaired test cases
        """
        ...
    
    def should_continue(
        self,
        iteration: int,
        current_success_rate: float,
        repairs_made: int,
    ) -> bool:
        """Determine if repair loop should continue.
        
        Args:
            iteration: Current iteration
            current_success_rate: Current success rate
            repairs_made: Number of repairs in last iteration
            
        Returns:
            Whether to continue
        """
        ...

"""
Agent for generating test cases using LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import override

from .base import BaseAgent, AgentContext, AgentAction, AgentActionType


@dataclass
class TestGenerationPrompt:
    """Prompt templates for test case generation."""
    
    SYSTEM_PROMPT: str = """You are an expert in test case generation for deep learning.
Your task is to generate concrete test cases that satisfy given requirements.

Generate test cases that:
1. Satisfy the given constraints
2. Cover the specified requirements
3. Include edge cases and boundary values
4. Have verifiable expected outputs
5. Use realistic input values

Output test cases as executable Python code."""

    TEST_GENERATION_PROMPT: str = """Generate test cases for:

Operator: {operator_name}
Requirement: {requirement}
Constraints: {constraints}
Oracle Type: {oracle_type}

Generate {count} test cases.

Output format:
```python
test_cases = [
    {{
        "id": "test_001",
        "name": "...",
        "inputs": {{...}},
        "attributes": {{...}},
        "expected": {{...}},
        "oracles": [...]
    }},
    ...
]
```"""


class TestGeneratorAgent(BaseAgent):
    """Agent for generating test cases.
    
    This agent generates concrete test cases based on requirements:
    - Generates input tensors satisfying constraints
    - Determines expected outputs
    - Creates appropriate test oracles
    - Handles various test categories (functional, boundary, etc.)
    
    It can use both LLM reasoning and programmatic generation
    to create diverse and comprehensive test cases.
    
    Example:
        >>> agent = TestGeneratorAgent(llm_client)
        >>> context = AgentContext(operator_spec=spec)
        >>> context.remember("requirement", req)
        >>> result = agent.execute(context)
        >>> test_cases = result["test_cases"]
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        fallback_generators: Optional[Dict[str, Callable]] = None,
        validation_enabled: bool = True,
    ):
        """Initialize test generator agent.
        
        Args:
            llm_client: LLM client
            fallback_generators: Fallback generators for different types
            validation_enabled: Whether to validate generated cases
        """
        ...
    
    @override
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """Generate test cases.
        
        Args:
            context: Context with operator_spec and requirement
            
        Returns:
            Dictionary with test cases
        """
        ...
    
    def generate_for_requirement(
        self,
        requirement: "TestRequirement",
        operator_spec: "OperatorSpec",
        count: int,
    ) -> List["TestCase"]:
        """Generate test cases for a specific requirement.
        
        Args:
            requirement: Test requirement
            operator_spec: Operator specification
            count: Number of cases to generate
            
        Returns:
            Generated test cases
        """
        ...
    
    def generate_inputs(
        self,
        input_specs: List["InputSpec"],
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate input tensors satisfying constraints.
        
        Args:
            input_specs: Input specifications
            constraints: Generation constraints
            
        Returns:
            Generated inputs
        """
        ...
    
    def generate_oracle(
        self,
        operator_spec: "OperatorSpec",
        oracle_type: str,
        reference_impl: Optional[Callable] = None,
    ) -> "TestOracle":
        """Generate appropriate test oracle.
        
        Args:
            operator_spec: Operator specification
            oracle_type: Type of oracle needed
            reference_impl: Optional reference implementation
            
        Returns:
            Generated test oracle
        """
        ...
    
    def compute_expected_output(
        self,
        inputs: Dict[str, Any],
        attributes: Dict[str, Any],
        reference_impl: Optional[Callable],
    ) -> Any:
        """Compute expected output for test case.
        
        Args:
            inputs: Test inputs
            attributes: Operator attributes
            reference_impl: Reference implementation
            
        Returns:
            Expected output
        """
        ...
    
    def validate_generated_case(
        self,
        test_case: "TestCase",
        operator_spec: "OperatorSpec",
    ) -> tuple[bool, Optional[str]]:
        """Validate a generated test case.
        
        Args:
            test_case: Generated test case
            operator_spec: Operator specification
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...
    
    def repair_invalid_case(
        self,
        test_case: "TestCase",
        error: str,
        operator_spec: "OperatorSpec",
    ) -> Optional["TestCase"]:
        """Attempt to repair an invalid test case.
        
        Args:
            test_case: Invalid test case
            error: Validation error
            operator_spec: Operator specification
            
        Returns:
            Repaired test case or None
        """
        ...


class PropertyBasedTestGenerator:
    """Generator for property-based test cases."""
    
    def generate_properties(
        self,
        operator_spec: "OperatorSpec",
    ) -> List[Callable]:
        """Generate property check functions.
        
        Args:
            operator_spec: Operator specification
            
        Returns:
            List of property check functions
        """
        ...
    
    def generate_hypothesis_strategies(
        self,
        operator_spec: "OperatorSpec",
    ) -> Dict[str, Any]:
        """Generate Hypothesis strategies for property testing.
        
        Args:
            operator_spec: Operator specification
            
        Returns:
            Hypothesis strategies
        """
        ...

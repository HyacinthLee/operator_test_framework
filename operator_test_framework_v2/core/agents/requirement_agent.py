"""
Agent for generating test requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from typing_extensions import override

from .base import BaseAgent, AgentContext, AgentAction, AgentActionType


@dataclass
class RequirementGenerationPrompt:
    """Prompt templates for requirement generation."""
    
    SYSTEM_PROMPT: str = """You are an expert in software testing and deep learning.
Your task is to generate comprehensive test requirements for deep learning operators.

Generate requirements for:
1. Functional correctness
2. Numerical stability
3. Boundary conditions
4. Error handling
5. Performance characteristics
6. Edge cases and corner cases

Be thorough and consider real-world bug patterns."""

    REQUIREMENT_GENERATION_PROMPT: str = """Generate test requirements for the following operator:

Operator: {operator_name}
Specification: {operator_spec}
Constraints: {constraints}

Generate requirements in JSON format:
{{
    "functional": [
        {{
            "id": "FUNC_001",
            "description": "...",
            "priority": "high",
            "constraints": {{...}}
        }}
    ],
    "numerical": [...],
    "boundary": [...],
    "error_handling": [...],
    "performance": [...]
}}"""


class RequirementGeneratorAgent(BaseAgent):
    """Agent for generating test requirements.
    
    This agent analyzes operator specifications and generates
    comprehensive test requirements covering:
    - Functional requirements (correctness)
    - Numerical requirements (stability, precision)
    - Boundary requirements (edge cases)
    - Error handling requirements
    - Performance requirements
    
    It uses domain knowledge and historical bug patterns to
    ensure comprehensive coverage.
    
    Example:
        >>> agent = RequirementGeneratorAgent(llm_client)
        >>> context = AgentContext(operator_spec=spec)
        >>> result = agent.execute(context)
        >>> requirements = result["requirements"]
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        use_historical_patterns: bool = True,
        coverage_targets: Optional[Dict[str, float]] = None,
    ):
        """Initialize requirement generator agent.
        
        Args:
            llm_client: LLM client
            use_historical_patterns: Whether to use historical bug patterns
            coverage_targets: Target coverage metrics
        """
        ...
    
    @override
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """Generate test requirements.
        
        Args:
            context: Context with operator_spec
            
        Returns:
            Dictionary with test requirements
        """
        ...
    
    def generate_functional_requirements(
        self,
        operator_spec: "OperatorSpec",
    ) -> List[Dict[str, Any]]:
        """Generate functional correctness requirements.
        
        Args:
            operator_spec: Operator specification
            
        Returns:
            Functional requirements
        """
        ...
    
    def generate_numerical_requirements(
        self,
        operator_spec: "OperatorSpec",
    ) -> List[Dict[str, Any]]:
        """Generate numerical stability requirements.
        
        Args:
            operator_spec: Operator specification
            
        Returns:
            Numerical requirements
        """
        ...
    
    def generate_boundary_requirements(
        self,
        operator_spec: "OperatorSpec",
    ) -> List[Dict[str, Any]]:
        """Generate boundary value requirements.
        
        Args:
            operator_spec: Operator specification
            
        Returns:
            Boundary requirements
        """
        ...
    
    def apply_historical_patterns(
        self,
        requirements: List[Dict[str, Any]],
        operator_type: str,
    ) -> List[Dict[str, Any]]:
        """Enrich requirements with historical bug patterns.
        
        Args:
            requirements: Generated requirements
            operator_type: Type of operator
            
        Returns:
            Enriched requirements
        """
        ...
    
    def prioritize_requirements(
        self,
        requirements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prioritize requirements based on risk and importance.
        
        Args:
            requirements: Requirements to prioritize
            
        Returns:
            Prioritized requirements
        """
        ...
    
    def validate_requirement_completeness(
        self,
        requirements: List[Dict[str, Any]],
        operator_spec: "OperatorSpec",
    ) -> tuple[bool, List[str]]:
        """Validate that requirements cover the specification.
        
        Args:
            requirements: Generated requirements
            operator_spec: Operator specification
            
        Returns:
            Tuple of (is_complete, missing_coverage)
        """
        ...

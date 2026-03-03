"""
Agent for extracting operator constraints from API definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import override

from .base import BaseAgent, AgentContext, AgentAction, AgentActionType


@dataclass
class ConstraintExtractionPrompt:
    """Prompt templates for constraint extraction."""
    
    SYSTEM_PROMPT: str = """You are an expert in deep learning operator analysis.
Your task is to analyze API definitions and extract precise tensor constraints.

Extract the following information:
1. Input tensor specifications (shape, dtype, device)
2. Output tensor specifications
3. Attribute/parameter constraints
4. Relationships between inputs and outputs
5. Valid ranges and edge cases
6. Broadcasting rules
7. Gradient requirements

Be precise and consider all edge cases."""

    API_ANALYSIS_PROMPT: str = """Analyze the following API:

API Name: {api_name}
Signature: {signature}
Docstring: {docstring}

Extract constraints in the following JSON format:
{{
    "inputs": [
        {{
            "name": "...",
            "shape_constraint": "...",
            "dtype_constraint": "...",
            "optional": false,
            "description": "..."
        }}
    ],
    "outputs": [...],
    "attributes": [...],
    "shape_relationships": "...",
    "special_cases": [...]
}}"""


class ConstraintExtractorAgent(BaseAgent):
    """Agent for extracting operator constraints.
    
    This agent analyzes operator APIs (signatures, docstrings, type hints)
    to extract precise tensor constraints including:
    - Shape constraints and relationships
    - Data type requirements
    - Device compatibility
    - Attribute constraints
    - Edge cases and special conditions
    
    Example:
        >>> agent = ConstraintExtractorAgent(llm_client)
        >>> context = AgentContext()
        >>> context.remember("api_callable", torch.softmax)
        >>> result = agent.execute(context)
        >>> spec = result["operator_spec"]
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        use_type_hints: bool = True,
        use_docstring: bool = True,
        use_source_analysis: bool = False,
    ):
        """Initialize constraint extractor agent.
        
        Args:
            llm_client: LLM client
            use_type_hints: Whether to use type hints
            use_docstring: Whether to analyze docstrings
            use_source_analysis: Whether to analyze source code
        """
        ...
    
    @override
    def execute(self, context: AgentContext) -> Dict[str, Any]:
        """Extract constraints from operator API.
        
        Args:
            context: Context with 'api_callable' in memory
            
        Returns:
            Dictionary with extracted constraints
        """
        ...
    
    def extract_from_signature(
        self,
        api_callable: Callable,
    ) -> Dict[str, Any]:
        """Extract constraints from function signature.
        
        Args:
            api_callable: Operator function/method
            
        Returns:
            Signature-based constraints
        """
        ...
    
    def extract_from_docstring(
        self,
        docstring: str,
        api_name: str,
    ) -> Dict[str, Any]:
        """Extract constraints from docstring using LLM.
        
        Args:
            docstring: Function docstring
            api_name: API name
            
        Returns:
            Docstring-based constraints
        """
        ...
    
    def extract_shape_relationships(
        self,
        api_callable: Callable,
        extracted_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract shape relationships between inputs and outputs.
        
        Args:
            api_callable: Operator function
            extracted_info: Previously extracted information
            
        Returns:
            Shape relationship constraints
        """
        ...
    
    def validate_extracted_constraints(
        self,
        constraints: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """Validate extracted constraints for completeness.
        
        Args:
            constraints: Extracted constraints
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        ...
    
    def build_operator_spec(
        self,
        api_name: str,
        extracted_constraints: Dict[str, Any],
    ) -> "OperatorSpec":
        """Build OperatorSpec from extracted constraints.
        
        Args:
            api_name: Operator name
            extracted_constraints: All extracted constraints
            
        Returns:
            Complete operator specification
        """
        ...

"""
Operator specification models.

Defines the structure for describing operator interfaces, including
inputs, outputs, and attributes with their constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum, auto


class AttributeType(Enum):
    """Types of operator attributes."""
    INT = auto()
    FLOAT = auto()
    BOOL = auto()
    STRING = auto()
    LIST_INT = auto()
    LIST_FLOAT = auto()
    TENSOR = auto()
    OPTIONAL = auto()


@dataclass
class OperatorAttribute:
    """Specification of an operator attribute/parameter.
    
    Attributes:
        name: Attribute name
        type: Attribute data type
        default: Default value if not provided
        required: Whether attribute is mandatory
        constraints: Optional validation constraints
        description: Human-readable description
    
    Example:
        >>> attr = OperatorAttribute(
        ...     name="dim",
        ...     type=AttributeType.INT,
        ...     default=-1,
        ...     required=False,
        ...     description="Dimension to operate on"
        ... )
    """
    name: str
    type: AttributeType
    default: Any = None
    required: bool = False
    constraints: Optional[Dict[str, Any]] = None
    description: str = ""
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate an attribute value.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@dataclass  
class InputSpec:
    """Specification of an operator input.
    
    Attributes:
        name: Input tensor name
        constraints: Tensor constraints (shape, dtype, etc.)
        optional: Whether input is optional
        variadic: Whether input accepts variable number of tensors
        description: Human-readable description
    """
    name: str
    constraints: Optional["TensorConstraint"] = None
    optional: bool = False
    variadic: bool = False
    description: str = ""


@dataclass
class OutputSpec:
    """Specification of an operator output.
    
    Attributes:
        name: Output tensor name
        shape_formula: Shape derivation formula/function
        dtype_formula: Dtype derivation formula/function
        description: Human-readable description
    """
    name: str
    shape_formula: Optional[Union[str, Callable[..., List[int]]]] = None
    dtype_formula: Optional[Union[str, Callable[..., str]]] = None
    description: str = ""
    
    def derive_shape(self, inputs: Dict[str, Any], attrs: Dict[str, Any]) -> List[int]:
        """Derive output shape from inputs and attributes.
        
        Args:
            inputs: Input tensors
            attrs: Operator attributes
            
        Returns:
            Derived output shape
        """
        ...
    
    def derive_dtype(self, inputs: Dict[str, Any], attrs: Dict[str, Any]) -> str:
        """Derive output dtype from inputs and attributes.
        
        Args:
            inputs: Input tensors
            attrs: Operator attributes
            
        Returns:
            Derived output dtype
        """
        ...


@dataclass
class OperatorSpec:
    """Complete specification of a deep learning operator.
    
    This is the central data model for describing operator interfaces
    and is used throughout the testing workflow.
    
    Attributes:
        name: Operator name (e.g., "torch.nn.functional.softmax")
        domain: Operator domain (e.g., "pytorch", "onnx", "tensorflow")
        version: Operator version
        inputs: List of input specifications
        outputs: List of output specifications
        attributes: List of attribute specifications
        description: Operator description
        constraints: Additional global constraints
        metadata: Additional metadata
    
    Example:
        >>> spec = OperatorSpec(
        ...     name="softmax",
        ...     domain="pytorch",
        ...     inputs=[
        ...         InputSpec(name="input", constraints=...)
        ...     ],
        ...     outputs=[
        ...         OutputSpec(name="output", shape_formula="same_as_input")
        ...     ],
        ...     attributes=[
        ...         OperatorAttribute(name="dim", type=AttributeType.INT, default=-1)
        ...     ]
        ... )
    """
    name: str
    domain: str
    version: Optional[str] = None
    inputs: List[InputSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)
    attributes: List[OperatorAttribute] = field(default_factory=list)
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_input(self, name: str) -> Optional[InputSpec]:
        """Get input specification by name.
        
        Args:
            name: Input name
            
        Returns:
            Input specification or None
        """
        ...
    
    def get_output(self, name: str) -> Optional[OutputSpec]:
        """Get output specification by name.
        
        Args:
            name: Output name
            
        Returns:
            Output specification or None
        """
        ...
    
    def get_attribute(self, name: str) -> Optional[OperatorAttribute]:
        """Get attribute specification by name.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute specification or None
        """
        ...
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a set of inputs against this specification.
        
        Args:
            inputs: Input tensors to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary representation."""
        ...
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OperatorSpec:
        """Create specification from dictionary representation."""
        ...
    
    @classmethod
    def from_api(cls, api_callable: Callable[..., Any]) -> OperatorSpec:
        """Extract operator specification from API callable.
        
        This method analyzes the function signature and docstring
        to automatically extract operator specifications.
        
        Args:
            api_callable: The operator function/method
            
        Returns:
            Extracted operator specification
        """
        ...

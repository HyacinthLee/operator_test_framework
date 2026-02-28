"""
Base adapter class for operator testing.

This module provides the abstract base class that all operator test adapters
must inherit from. It defines the interface for operator testing and provides
default implementations for common testing patterns.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable
import torch
import torch.nn as nn
import time


@dataclass
class TestConfig:
    """Configuration for operator testing.
    
    Attributes:
        device: Device to run tests on ('cpu', 'cuda', etc.)
        dtype: Default data type for tensors
        tolerance: Numerical tolerance for comparisons
        max_batch_size: Maximum batch size for testing
        num_iterations: Number of iterations for performance tests
        warmup_iterations: Number of warmup iterations before benchmarking
        seed: Random seed for reproducibility
    """
    device: str = 'cpu'
    dtype: torch.dtype = torch.float32
    tolerance: float = 1e-5
    max_batch_size: int = 128
    num_iterations: int = 100
    warmup_iterations: int = 10
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = 'cpu'


class OperatorTestAdapter(ABC):
    """Abstract base class for operator test adapters.
    
    This class provides a standardized interface for testing deep learning
    operators. Subclasses must implement the forward() and generate_inputs()
    methods, and can override other methods for custom validation.
    
    Example:
        class MyOpAdapter(OperatorTestAdapter):
            def __init__(self):
                super().__init__("MyOp")
            
            def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
                return torch.relu(inputs['x'])
            
            def generate_inputs(self, batch_size: int = 2, **kwargs) -> Dict[str, torch.Tensor]:
                return {'x': torch.randn(batch_size, 64)}
    """
    
    def __init__(self, name: str, config: Optional[TestConfig] = None):
        """Initialize the operator test adapter.
        
        Args:
            name: Name of the operator being tested
            config: Test configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or TestConfig()
        self._set_seed()
        
        # Storage for test results
        self.test_results: Dict[str, Any] = {}
        
    def _set_seed(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute the operator forward pass.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Output tensor from the operator
        """
        pass
    
    @abstractmethod
    def generate_inputs(self, batch_size: int = 2, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate input tensors for testing.
        
        Args:
            batch_size: Batch size for generated inputs
            **kwargs: Additional arguments for input generation
            
        Returns:
            Dictionary of input tensors
        """
        pass
    
    def backward(self, outputs: torch.Tensor, inputs: Dict[str, torch.Tensor],
                 grad_output: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Execute backward pass and return gradients.
        
        Args:
            outputs: Output tensor from forward pass
            inputs: Input tensors (must have requires_grad=True)
            grad_output: Gradient of loss w.r.t. outputs (uses ones if None)
            
        Returns:
            Dictionary of gradients for each input
        """
        if grad_output is None:
            grad_output = torch.ones_like(outputs)
        
        # Zero existing gradients
        for tensor in inputs.values():
            if tensor.requires_grad and tensor.grad is not None:
                tensor.grad.zero_()
        
        # Compute gradients
        outputs.backward(grad_output, retain_graph=True)
        
        # Collect gradients
        gradients = {}
        for name, tensor in inputs.items():
            if tensor.requires_grad:
                gradients[name] = tensor.grad.clone() if tensor.grad is not None else None
        
        return gradients
    
    def get_parameters(self) -> Optional[Dict[str, torch.nn.Parameter]]:
        """Get operator parameters if applicable.
        
        Returns:
            Dictionary of parameters, or None if operator is stateless
        """
        return None
    
    def validate_output_shape(self, inputs: Dict[str, torch.Tensor],
                             outputs: torch.Tensor) -> bool:
        """Validate that output shape is correct.
        
        Args:
            inputs: Input tensors
            outputs: Output tensor
            
        Returns:
            True if output shape is valid
        """
        # Default implementation: check that batch dimension is preserved
        first_input = list(inputs.values())[0]
        return outputs.shape[0] == first_input.shape[0]
    
    def validate_output_type(self, outputs: torch.Tensor) -> bool:
        """Validate that output type is correct.
        
        Args:
            outputs: Output tensor
            
        Returns:
            True if output type is valid
        """
        return isinstance(outputs, torch.Tensor)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests.
        
        Returns:
            Dictionary containing all test results
        """
        print(f"\n{'='*60}")
        print(f"Running all tests for: {self.name}")
        print(f"{'='*60}\n")
        
        results = {
            'operator_name': self.name,
            'config': self.config,
            'tests': {}
        }
        
        # Run each validation layer
        results['tests']['mathematical'] = self.validate_mathematical()
        results['tests']['numerical'] = self.validate_numerical()
        results['tests']['functional'] = self.validate_functional()
        results['tests']['performance'] = self.validate_performance()
        
        self.test_results = results
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def validate_mathematical(self) -> Dict[str, Any]:
        """Validate mathematical correctness.
        
        Returns:
            Test results dictionary
        """
        from .validators import MathematicalValidator
        validator = MathematicalValidator(self.config)
        return validator.validate(self)
    
    def validate_numerical(self) -> Dict[str, Any]:
        """Validate numerical stability.
        
        Returns:
            Test results dictionary
        """
        from .validators import NumericalValidator
        validator = NumericalValidator(self.config)
        return validator.validate(self)
    
    def validate_functional(self) -> Dict[str, Any]:
        """Validate functional compliance.
        
        Returns:
            Test results dictionary
        """
        from .validators import FunctionalValidator
        validator = FunctionalValidator(self.config)
        return validator.validate(self)
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance characteristics.
        
        Returns:
            Test results dictionary
        """
        from .validators import PerformanceValidator
        validator = PerformanceValidator(self.config)
        return validator.validate(self)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary.
        
        Args:
            results: Test results dictionary
        """
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        
        all_passed = True
        for test_name, test_results in results['tests'].items():
            passed = test_results.get('passed', False)
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name:20s}: {status}")
            if not passed:
                all_passed = False
        
        print(f"{'='*60}")
        overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
        print(f"Overall: {overall}")
        print(f"{'='*60}\n")

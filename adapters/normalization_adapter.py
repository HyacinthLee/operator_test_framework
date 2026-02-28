"""
Normalization operator adapters for testing.
"""

import torch
from typing import List, Tuple
import sys
sys.path.append('/root/.openclaw/workspace')
from operator_test_framework.core.adapter import OperatorTestAdapter


class LayerNormAdapter(OperatorTestAdapter):
    """Test adapter for LayerNorm operator."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__("LayerNorm", "normalization")
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Create reference module
        self.ref_module = torch.nn.LayerNorm(normalized_shape, eps=eps)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch native LayerNorm."""
        return self.ref_module(x)
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom LayerNorm implementation.
        
        For demonstration - replace with your optimized version.
        """
        # Manual implementation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters if they exist
        if self.ref_module.elementwise_affine:
            x_norm = x_norm * self.ref_module.weight + self.ref_module.bias
        
        return x_norm
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test cases for LayerNorm."""
        test_cases = []
        
        shapes = [
            (1, self.normalized_shape),
            (4, self.normalized_shape),
            (8, 16, self.normalized_shape),
            (2, 4, 8, self.normalized_shape),
        ]
        
        for shape in shapes:
            # Standard case
            x = torch.randn(*shape)
            test_cases.append((x,))
            
            # With gradient
            x_grad = torch.randn(*shape, requires_grad=True)
            test_cases.append((x_grad,))
            
            # Near-zero variance (edge case)
            x_small_var = torch.ones(*shape) + torch.randn(*shape) * 1e-7
            test_cases.append((x_small_var,))
            
            # Large values
            x_large = torch.randn(*shape) * 1000
            test_cases.append((x_large,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify LayerNorm properties."""
        x = inputs[0]
        
        # Property 1: Output shape matches input
        if output.shape != x.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: For standard input, normalized output should have ~0 mean and ~1 std
        # (only check if not using affine transformation)
        if not self.ref_module.elementwise_affine:
            mean = output.mean(dim=-1)
            std = output.std(dim=-1, unbiased=False)
            
            if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-5):
                return False
            if not torch.allclose(std, torch.ones_like(std), atol=1e-4):
                return False
        
        return True


class RMSNormAdapter(OperatorTestAdapter):
    """Test adapter for RMSNorm operator."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__("RMSNorm", "normalization")
        self.dim = dim
        self.eps = eps
        self.weight = torch.ones(dim)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMSNorm implementation.
        Reference: https://arxiv.org/abs/1910.07467
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight.to(x.device)
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Custom RMSNorm - replace with your implementation."""
        return self.reference_implementation(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test cases for RMSNorm."""
        test_cases = []
        
        shapes = [
            (1, self.dim),
            (4, self.dim),
            (8, 16, self.dim),
        ]
        
        for shape in shapes:
            x = torch.randn(*shape)
            test_cases.append((x,))
            
            x_grad = torch.randn(*shape, requires_grad=True)
            test_cases.append((x_grad,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify RMSNorm properties."""
        x = inputs[0]
        
        # Property 1: Output shape matches input
        if output.shape != x.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: RMSNorm preserves sign (unlike LayerNorm)
        # Output should have same sign as input (when weight > 0)
        if not torch.all((output >= 0) == (x >= 0)):
            return False
        
        return True

"""
Transformer Engine adapter for testing.
Tests NVIDIA Transformer Engine operators with FP8 precision.
"""

import torch
import sys
sys.path.append('/root/.openclaw/workspace')

from operator_test_framework.core.adapter import OperatorTestAdapter
from typing import List, Tuple, Optional, Dict

# Try to import Transformer Engine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: Transformer Engine not installed. Tests will be skipped.")


class TransformerEngineLinearAdapter(OperatorTestAdapter):
    """
    Test adapter for Transformer Engine Linear layer.
    
    Tests FP8 linear layer against PyTorch native implementation.
    """
    
    def __init__(
        self,
        in_features: int = 768,
        out_features: int = 3072,
        fp8_format: str = "E4M3",
        device: str = 'cuda'
    ):
        super().__init__("TE_Linear", "transformer_engine")
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine not available")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for Transformer Engine")
        
        # Create FP8 recipe
        if fp8_format == "E4M3":
            fp8_format_enum = recipe.Format.E4M3
        elif fp8_format == "HYBRID":
            fp8_format_enum = recipe.Format.HYBRID
        else:
            fp8_format_enum = recipe.Format.E4M3
        
        self.fp8_recipe = recipe.DelayedScaling(
            margin=0,
            fp8_format=fp8_format_enum
        )
        
        # Create TE module
        self.te_module = te.Linear(in_features, out_features, bias=True)
        self.te_module.to(device)
        self.te_module.train()
        
        # Reference: PyTorch native Linear
        self.ref_module = torch.nn.Linear(in_features, out_features, bias=True)
        self.ref_module.to(device)
        
        # Copy weights for fair comparison
        with torch.no_grad():
            self.ref_module.weight.copy_(self.te_module.weight)
            if self.te_module.bias is not None:
                self.ref_module.bias.copy_(self.te_module.bias)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference: PyTorch native FP32 Linear.
        """
        return self.ref_module(x.float())
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Test: TE FP8 Linear.
        """
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            return self.te_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """
        Generate test inputs for TE Linear.
        """
        test_cases = []
        
        batch_sizes = [1, 16, 32]
        seq_lengths = [1, 128, 512]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Standard case
                x = torch.randn(batch, seq_len, self.in_features, device=self.device)
                test_cases.append((x,))
                
                # With gradient
                x_grad = torch.randn(
                    batch, seq_len, self.in_features,
                    device=self.device, requires_grad=True
                )
                test_cases.append((x_grad,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """
        Verify TE Linear output properties.
        """
        x = inputs[0]
        
        # Property 1: Output shape is correct
        expected_shape = x.shape[:-1] + (self.out_features,)
        if output.shape != expected_shape:
            return False
        
        # Property 2: Output is finite (no NaN/Inf)
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Output dtype matches input (or is FP8 internal)
        # TE may output FP16 or FP32 depending on configuration
        
        return True
    
    def get_fp8_scale(self) -> float:
        """Get current FP8 scaling factor."""
        return self.fp8_recipe.scale


class TransformerEngineLayerNormAdapter(OperatorTestAdapter):
    """
    Test adapter for Transformer Engine LayerNorm.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        eps: float = 1e-5,
        device: str = 'cuda'
    ):
        super().__init__("TE_LayerNorm", "transformer_engine")
        self.hidden_size = hidden_size
        self.device = device
        
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine not available")
        
        # Create TE LayerNorm
        self.te_module = te.LayerNorm(hidden_size, eps=eps)
        self.te_module.to(device)
        
        # Reference: PyTorch native LayerNorm
        self.ref_module = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.ref_module.to(device)
        
        # Copy parameters
        with torch.no_grad():
            self.ref_module.weight.copy_(self.te_module.weight)
            self.ref_module.bias.copy_(self.te_module.bias)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Reference: PyTorch native LayerNorm."""
        return self.ref_module(x.float())
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Test: TE LayerNorm."""
        return self.te_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test inputs."""
        test_cases = []
        
        shapes = [
            (16, 128, self.hidden_size),
            (32, 512, self.hidden_size),
            (8, 2048, self.hidden_size),
        ]
        
        for shape in shapes:
            x = torch.randn(*shape, device=self.device)
            test_cases.append((x,))
            
            # Near-zero variance case
            x_small_var = torch.ones(*shape, device=self.device)
            x_small_var += torch.randn(*shape, device=self.device) * 1e-7
            test_cases.append((x_small_var,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify LayerNorm properties."""
        x = inputs[0]
        
        # Property 1: Shape preserved
        if output.shape != x.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: For standard input, output should have ~0 mean and ~1 std
        # (last dimension)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False
        if not torch.allclose(std, torch.ones_like(std), atol=1e-3):
            return False
        
        return True


class TransformerEngineTransformerLayerAdapter(OperatorTestAdapter):
    """
    Test adapter for complete Transformer Engine TransformerLayer.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        ffn_hidden_size: int = 3072,
        device: str = 'cuda'
    ):
        super().__init__("TE_TransformerLayer", "transformer_engine")
        self.hidden_size = hidden_size
        self.device = device
        
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine not available")
        
        # FP8 recipe
        self.fp8_recipe = recipe.DelayedScaling(
            margin=0,
            fp8_format=recipe.Format.E4M3
        )
        
        # Create TE TransformerLayer
        self.te_module = te.TransformerLayer(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_attention_heads=num_attention_heads,
            self_attn_mask_type="padding"
        )
        self.te_module.to(device)
        self.te_module.train()
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference: PyTorch native TransformerEncoderLayer.
        Note: This is approximate as TE uses custom implementation.
        """
        # For end-to-end testing, we mainly check output properties
        # rather than exact numerical match
        return x  # Placeholder
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Test: TE TransformerLayer."""
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
            return self.te_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test inputs."""
        test_cases = []
        
        batch_sizes = [2, 8]
        seq_lengths = [128, 512]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                x = torch.randn(
                    batch, seq_len, self.hidden_size,
                    device=self.device, requires_grad=True
                )
                test_cases.append((x,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify TransformerLayer output properties."""
        x = inputs[0]
        
        # Property 1: Shape preserved
        if output.shape != x.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Output should be different from input (transformation occurred)
        # Allow small differences due to residual connections
        if torch.allclose(output, x, rtol=1e-3, atol=1e-4):
            return False
        
        return True


# Utility functions for TE testing
def check_te_availability() -> bool:
    """Check if Transformer Engine is available."""
    return TE_AVAILABLE and torch.cuda.is_available()


def get_te_version() -> Optional[str]:
    """Get Transformer Engine version."""
    if not TE_AVAILABLE:
        return None
    try:
        return te.__version__
    except AttributeError:
        return "unknown"


def print_te_info():
    """Print Transformer Engine information."""
    print("=" * 60)
    print("Transformer Engine Information")
    print("=" * 60)
    print(f"Available: {check_te_availability()}")
    print(f"Version: {get_te_version()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Check GPU architecture for FP8 support
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 9:  # Hopper or newer
            print(f"FP8 Support: Yes (SM {capability[0]}{capability[1]})")
        elif capability[0] == 8 and capability[1] >= 9:  # Ada
            print(f"FP8 Support: Yes (SM {capability[0]}{capability[1]})")
        else:
            print(f"FP8 Support: No (SM {capability[0]}{capability[1]})")
    else:
        print(f"CUDA Available: False")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    print_te_info()
    
    if not check_te_availability():
        print("Transformer Engine not available. Skipping tests.")
        exit(0)
    
    # Test TE Linear
    print("\nTesting TE Linear...")
    adapter = TransformerEngineLinearAdapter(768, 3072)
    results = adapter.run_full_test_suite(verbose=True)
    print(f"\nTE Linear: {results['passed']}/{results['total']} passed")
    
    # Test TE LayerNorm
    print("\nTesting TE LayerNorm...")
    adapter = TransformerEngineLayerNormAdapter(768)
    results = adapter.run_full_test_suite(verbose=True)
    print(f"\nTE LayerNorm: {results['passed']}/{results['total']} passed")

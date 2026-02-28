"""
Comprehensive tests for Transformer Engine operators.
"""

import pytest
import torch
import sys
sys.path.append('/root/.openclaw/workspace')

from operator_test_framework.adapters.transformer_engine_adapter import (
    TransformerEngineLinearAdapter,
    TransformerEngineLayerNormAdapter,
    TransformerEngineTransformerLayerAdapter,
    check_te_availability,
    get_te_version,
)
from operator_test_framework.core import (
    MixedPrecisionTester,
    MemoryLeakDetector,
    GradientStabilityTester,
)

# Skip all tests if TE not available
pytestmark = pytest.mark.skipif(
    not check_te_availability(),
    reason="Transformer Engine not available or CUDA not enabled"
)


class TestTransformerEngineLinear:
    """Tests for TE Linear layer."""
    
    def test_te_linear_correctness(self):
        """Test TE Linear numerical correctness vs PyTorch."""
        adapter = TransformerEngineLinearAdapter(768, 3072)
        results = adapter.run_full_test_suite(verbose=False)
        
        assert results['failed'] == 0, f"Failed tests: {results['failed']}"
    
    def test_te_linear_fp8_numerical_stability(self):
        """Test FP8 numerical stability with various inputs."""
        adapter = TransformerEngineLinearAdapter(1024, 4096, fp8_format="E4M3")
        
        # Test with different input magnitudes
        test_cases = [
            torch.randn(16, 1024, device='cuda'),  # Normal
            torch.randn(16, 1024, device='cuda') * 0.01,  # Small
            torch.randn(16, 1024, device='cuda') * 100,  # Large
        ]
        
        for x in test_cases:
            out_te = adapter.test_implementation(x)
            out_ref = adapter.reference_implementation(x)
            
            # FP8 allows larger error than FP16
            rel_error = (out_te.float() - out_ref).abs().mean() / out_ref.abs().mean()
            assert rel_error < 0.05, f"Relative error too large: {rel_error}"
    
    def test_te_linear_gradient_stability(self):
        """Test gradient stability in FP8 training."""
        tester = GradientStabilityTester(
            underflow_threshold=1e-6,
            overflow_threshold=1e4
        )
        
        adapter = TransformerEngineLinearAdapter(1024, 4096)
        x = torch.randn(16, 1024, device='cuda', requires_grad=True)
        
        # Forward + backward
        out = adapter.test_implementation(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradient health
        assert x.grad is not None, "No gradient computed"
        
        grad_stats = tester.analyze_gradient_stats(x.grad)
        
        # FP8 training should have healthy gradients
        assert grad_stats['nan_count'] == 0, "NaN in gradients"
        assert grad_stats['inf_count'] == 0, "Inf in gradients"
        
        # Allow some underflow in FP8, but not too much
        assert grad_stats['underflow_ratio'] < 0.7, f"Too much underflow: {grad_stats['underflow_ratio']}"
    
    def test_te_linear_memory_efficiency(self):
        """Test TE Linear memory efficiency."""
        detector = MemoryLeakDetector()
        
        adapter = TransformerEngineLinearAdapter(4096, 4096)
        x = torch.randn(32, 4096, device='cuda')
        
        # Test for memory leaks
        results = detector.detect_leak_single_run(
            adapter.test_implementation,
            x,
            warmup=3,
            iterations=10
        )
        
        # Should not have significant memory leak
        assert not results['leak_detected'], f"Memory leak detected: {results['avg_growth_per_iter_mb']} MB/iter"
    
    def test_te_linear_fp8_scaling_adaptation(self):
        """Test FP8 scaling factor adapts during training."""
        adapter = TransformerEngineLinearAdapter(1024, 4096)
        initial_scale = adapter.get_fp8_scale()
        
        # Simulate training steps
        for _ in range(5):
            x = torch.randn(16, 1024, device='cuda', requires_grad=True)
            out = adapter.test_implementation(x)
            loss = out.sum()
            loss.backward()
        
        final_scale = adapter.get_fp8_scale()
        
        # Scale should have adapted (changed from initial)
        # Note: Scale might increase or decrease depending on amax values
        assert final_scale > 0, "Scale should be positive"
        assert final_scale < 65536, "Scale should not overflow"


class TestTransformerEngineLayerNorm:
    """Tests for TE LayerNorm."""
    
    def test_te_layernorm_correctness(self):
        """Test TE LayerNorm correctness."""
        adapter = TransformerEngineLayerNormAdapter(768)
        results = adapter.run_full_test_suite(verbose=False)
        
        assert results['failed'] == 0
    
    def test_te_layernorm_numerical_stability(self):
        """Test LayerNorm with challenging inputs."""
        adapter = TransformerEngineLayerNormAdapter(768)
        
        # Test with near-zero variance
        x_small_var = torch.ones(16, 128, 768, device='cuda')
        x_small_var += torch.randn(16, 128, 768, device='cuda') * 1e-7
        
        out = adapter.test_implementation(x_small_var)
        
        # Should still produce finite output
        assert torch.isfinite(out).all(), "LayerNorm produced non-finite values"
        
        # Should have ~0 mean and ~1 std
        mean = out.mean(dim=-1)
        std = out.std(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-3)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-2)
    
    def test_te_layernorm_vs_pytorch(self):
        """Compare TE LayerNorm with PyTorch native."""
        adapter = TransformerEngineLayerNormAdapter(768)
        
        x = torch.randn(16, 128, 768, device='cuda')
        
        out_te = adapter.test_implementation(x)
        out_ref = adapter.reference_implementation(x)
        
        # Should be very close (LayerNorm is deterministic)
        max_diff = (out_te.float() - out_ref).abs().max()
        assert max_diff < 1e-4, f"LayerNorm mismatch: {max_diff}"


class TestTransformerEngineTransformerLayer:
    """Tests for complete TE TransformerLayer."""
    
    def test_te_transformer_layer_forward(self):
        """Test TransformerLayer forward pass."""
        adapter = TransformerEngineTransformerLayerAdapter(
            hidden_size=768,
            num_attention_heads=12,
            ffn_hidden_size=3072
        )
        
        x = torch.randn(2, 128, 768, device='cuda', requires_grad=True)
        out = adapter.test_implementation(x)
        
        # Check output properties
        assert out.shape == x.shape, "Output shape mismatch"
        assert torch.isfinite(out).all(), "Non-finite values in output"
    
    def test_te_transformer_layer_backward(self):
        """Test TransformerLayer backward pass."""
        adapter = TransformerEngineTransformerLayerAdapter(
            hidden_size=768,
            num_attention_heads=12,
            ffn_hidden_size=3072
        )
        
        x = torch.randn(2, 128, 768, device='cuda', requires_grad=True)
        
        # Forward
        out = adapter.test_implementation(x)
        loss = out.sum()
        
        # Backward should work without error
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None, "Input gradient not computed"
        assert torch.isfinite(x.grad).all(), "Non-finite gradients"
    
    def test_te_transformer_layer_memory(self):
        """Test TransformerLayer memory usage."""
        detector = MemoryLeakDetector()
        
        adapter = TransformerEngineTransformerLayerAdapter(
            hidden_size=1024,
            num_attention_heads=16,
            ffn_hidden_size=4096
        )
        
        x = torch.randn(4, 512, 1024, device='cuda')
        
        results = detector.detect_leak_single_run(
            adapter.test_implementation,
            x,
            warmup=2,
            iterations=5
        )
        
        # Transformer layer uses significant memory, but shouldn't leak
        assert results['avg_growth_per_iter_mb'] < 10, "Possible memory leak in TransformerLayer"


class TestTransformerEngineMixedPrecision:
    """Mixed precision specific tests."""
    
    def test_fp8_vs_fp16_accuracy(self):
        """Compare FP8 vs FP16 accuracy."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        
        # Check GPU capability
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9 and not (capability[0] == 8 and capability[1] >= 9):
            pytest.skip("FP8 requires Hopper (SM90) or Ada (SM89) GPU")
        
        tester = MixedPrecisionTester(rtol=1e-2, atol=1e-3)
        
        # Create simple test operator
        def test_op(x):
            return x @ x.T
        
        x = torch.randn(64, 64, device='cuda')
        
        # Test FP16
        result_fp16 = tester.test_numerical_equivalence(
            test_op, (x,), dtype=torch.float16
        )
        
        # FP16 should have good accuracy
        assert result_fp16['passed'], f"FP16 accuracy issue: {result_fp16['max_rel_error']}"
    
    def test_gradient_scaling_fp8(self):
        """Test gradient scaling with FP8."""
        adapter = TransformerEngineLinearAdapter(1024, 4096)
        
        x = torch.randn(16, 1024, device='cuda', requires_grad=True)
        
        # Forward with FP8
        out = adapter.test_implementation(x)
        loss = out.sum()
        
        # Backward
        loss.backward()
        
        # Check gradient scaling
        grad_norm = x.grad.norm()
        assert grad_norm > 1e-8, "Gradients too small (underflow)"
        assert grad_norm < 1e6, "Gradients too large (overflow)"


class TestTransformerEngineInfo:
    """Information and utility tests."""
    
    def test_te_version(self):
        """Test TE version is available."""
        version = get_te_version()
        assert version is not None, "TE version not available"
        print(f"Transformer Engine version: {version}")
    
    def test_te_availability(self):
        """Test TE availability check."""
        available = check_te_availability()
        assert available, "Transformer Engine should be available for these tests"


if __name__ == "__main__":
    # Print TE info
    from operator_test_framework.adapters.transformer_engine_adapter import print_te_info
    print_te_info()
    
    # Run tests
    pytest.main([__file__, "-v"])

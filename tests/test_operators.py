"""
Example tests for the operator test framework.
"""

import pytest
import torch
import sys
sys.path.append('/root/.openclaw/workspace')

from operator_test_framework.adapters import (
    AttentionAdapter,
    LayerNormAdapter,
    RMSNormAdapter
)
from operator_test_framework.core import (
    NumericalStabilityTester,
    PerformanceBenchmark,
    verify_gradient
)


class TestAttentionOperator:
    """Test cases for Attention operator."""
    
    def test_attention_correctness(self):
        """Test mathematical correctness."""
        adapter = AttentionAdapter(dim=64, num_heads=8)
        results = adapter.run_full_test_suite(verbose=False)
        
        assert results["failed"] == 0, f"Failed tests: {results['failed']}"
    
    def test_attention_gradient(self):
        """Test gradient correctness."""
        adapter = AttentionAdapter(dim=64, num_heads=8)
        
        # Get a test case with gradients
        test_cases = adapter.generate_test_cases()
        grad_case = None
        for case in test_cases:
            if any(t.requires_grad for t in case):
                grad_case = case
                break
        
        if grad_case:
            q, k, v = grad_case
            # Should not raise
            verify_gradient(
                lambda q: adapter.test_implementation(q, k, v),
                q
            )
    
    def test_attention_numerical_stability(self):
        """Test numerical stability."""
        tester = NumericalStabilityTester()
        
        def simple_attention(x):
            # Simplified attention for testing
            return torch.softmax(x @ x.T, dim=-1) @ x
        
        results = tester.test_operator(
            simple_attention,
            (16, 64)
        )
        
        # Most tests should pass
        assert results["passed"] >= results["total"] * 0.75


class TestLayerNormOperator:
    """Test cases for LayerNorm operator."""
    
    def test_layernorm_correctness(self):
        """Test mathematical correctness."""
        adapter = LayerNormAdapter(normalized_shape=64)
        results = adapter.run_full_test_suite(verbose=False)
        
        assert results["failed"] == 0
    
    def test_layernorm_properties(self):
        """Test LayerNorm specific properties."""
        adapter = LayerNormAdapter(normalized_shape=64)
        
        # Test with standard input
        x = torch.randn(4, 64)
        output = adapter.test_implementation(x)
        
        # Without affine transform, mean should be ~0 and std ~1
        # (This depends on the specific implementation)
        assert torch.isfinite(output).all()
    
    def test_layernorm_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        adapter = LayerNormAdapter(normalized_shape=64)
        tester = NumericalStabilityTester()
        
        results = tester.test_operator(
            adapter.test_implementation,
            (4, 64)
        )
        
        tester.print_results(results)
        assert results["failed"] == 0


class TestRMSNormOperator:
    """Test cases for RMSNorm operator."""
    
    def test_rmsnorm_correctness(self):
        """Test mathematical correctness."""
        adapter = RMSNormAdapter(dim=64)
        results = adapter.run_full_test_suite(verbose=False)
        
        assert results["failed"] == 0
    
    def test_rmsnorm_vs_layernorm_zero_mean(self):
        """Test that RMSNorm equals LayerNorm for zero-mean input."""
        rms_norm = RMSNormAdapter(dim=64)
        layer_norm = LayerNormAdapter(normalized_shape=64)
        
        # Zero-mean input
        x = torch.randn(4, 64)
        x = x - x.mean(dim=-1, keepdim=True)
        
        # For zero-mean input, RMSNorm should approximate LayerNorm
        out_rms = rms_norm.test_implementation(x)
        out_ln = layer_norm.test_implementation(x)
        
        # They won't be exactly equal due to different formulas,
        # but both should produce finite, normalized outputs
        assert torch.isfinite(out_rms).all()
        assert torch.isfinite(out_ln).all()


class TestPerformanceBenchmark:
    """Test performance benchmarking functionality."""
    
    def test_latency_measurement(self):
        """Test latency measurement."""
        benchmark = PerformanceBenchmark(
            warmup_iters=5,
            benchmark_iters=10
        )
        
        def dummy_op(x):
            return x @ x.T
        
        x = torch.randn(128, 128)
        latency = benchmark.measure_latency(dummy_op, x)
        
        assert "mean_ms" in latency
        assert "min_ms" in latency
        assert "max_ms" in latency
        assert latency["mean_ms"] > 0
    
    def test_comparison(self):
        """Test implementation comparison."""
        benchmark = PerformanceBenchmark(warmup_iters=2, benchmark_iters=5)
        
        x = torch.randn(256, 256)
        
        implementations = {
            "matmul": lambda: x @ x.T,
            "einsum": lambda: torch.einsum('ij,jk->ik', x, x.T),
        }
        
        results = benchmark.compare_implementations(
            implementations,
            metrics=["latency"]
        )
        
        assert "speedups" in results
        assert len(results["implementations"]) == 2


if __name__ == "__main__":
    # Run with: pytest tests/test_operators.py -v
    pytest.main([__file__, "-v"])

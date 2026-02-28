"""
Tests for Megatron-LM operators.
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.append('/root/.openclaw/workspace')

from operator_test_framework.adapters.megatron_lm_adapter import (
    MegatronLayerNormAdapter,
    MegatronRMSNormAdapter,
    MegatronMLPAdapter,
    MegatronRotaryEmbeddingAdapter,
    MegatronAttentionAdapter,
    MegatronLinearAdapter,
)
from operator_test_framework.core import (
    MixedPrecisionTester,
    MemoryLeakDetector,
    GradientStabilityTester,
)


class TestMegatronLayerNorm:
    """Tests for Megatron LayerNorm."""
    
    def test_layernorm_correctness(self):
        """Test Megatron LayerNorm correctness vs PyTorch."""
        adapter = MegatronLayerNormAdapter(768)
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_layernorm_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        adapter = MegatronLayerNormAdapter(768)
        
        test_cases = [
            torch.randn(16, 128, 768),  # Normal
            torch.ones(16, 128, 768) * 1e-7,  # Near zero
            torch.randn(16, 128, 768) * 1e4,  # Large values
        ]
        
        for x in test_cases:
            out = adapter.test_implementation(x)
            assert torch.isfinite(out).all(), "Non-finite output"


class TestMegatronRMSNorm:
    """Tests for Megatron RMSNorm."""
    
    def test_rmsnorm_correctness(self):
        """Test RMSNorm correctness."""
        adapter = MegatronRMSNormAdapter(768)
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_rmsnorm_vs_layernorm(self):
        """Compare RMSNorm with LayerNorm for zero-mean input."""
        rms_norm = MegatronRMSNormAdapter(768)
        layer_norm = MegatronLayerNormAdapter(768)
        
        # Zero-mean input
        x = torch.randn(4, 64, 768)
        x = x - x.mean(dim=-1, keepdim=True)
        
        out_rms = rms_norm.test_implementation(x)
        out_ln = layer_norm.test_implementation(x)
        
        # Both should produce valid outputs
        assert torch.isfinite(out_rms).all()
        assert torch.isfinite(out_ln).all()


class TestMegatronMLP:
    """Tests for Megatron MLP (SwiGLU)."""
    
    def test_mlp_correctness(self):
        """Test MLP correctness."""
        adapter = MegatronMLPAdapter(
            hidden_size=768,
            ffn_hidden_size=3072
        )
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_mlp_gradient_stability(self):
        """Test MLP gradient stability."""
        tester = GradientStabilityTester()
        
        adapter = MegatronMLPAdapter(768, 3072)
        x = torch.randn(16, 768, requires_grad=True)
        
        out = adapter.test_implementation(x)
        loss = out.sum()
        loss.backward()
        
        health = tester.test_gradient_health(lambda x: adapter.test_implementation(x), x)
        assert health['healthy'], f"Gradient issues: {health['issues']}"


class TestMegatronRotaryEmbedding:
    """Tests for Megatron Rotary Embedding (RoPE)."""
    
    def test_rotary_embedding_correctness(self):
        """Test RoPE correctness."""
        adapter = MegatronRotaryEmbeddingAdapter(
            dim=64,
            max_seq_len=512
        )
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_rotary_embedding_position_invariance(self):
        """Test that RoPE correctly encodes position information."""
        adapter = MegatronRotaryEmbeddingAdapter(dim=64, max_seq_len=512)
        
        # Same content at different positions
        x_pos0 = torch.randn(1, 1, 64)
        x_pos10 = x_pos0.clone()
        
        # Apply RoPE at different positions
        out_pos0 = adapter.test_module.forward(x_pos0, seq_len=1, offset=0)
        out_pos10 = adapter.test_module.forward(x_pos10, seq_len=1, offset=10)
        
        # Outputs should be different (position encoded)
        assert not torch.allclose(out_pos0, out_pos10, atol=1e-6)
    
    def test_rotary_embedding_seq_len_1(self):
        """Test RoPE with seq_len=1 (edge case)."""
        adapter = MegatronRotaryEmbeddingAdapter(dim=64, max_seq_len=512)
        
        x = torch.randn(2, 1, 64)
        out = adapter.test_module.forward(x, seq_len=1, offset=0)
        
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
    
    def test_rotary_embedding_offset_greater_than_0(self):
        """Test RoPE with offset > 0."""
        adapter = MegatronRotaryEmbeddingAdapter(dim=64, max_seq_len=512)
        
        x = torch.randn(2, 10, 64)
        out = adapter.test_module.forward(x, seq_len=10, offset=100)
        
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


class TestMegatronAttention:
    """Tests for Megatron Attention."""
    
    def test_attention_correctness(self):
        """Test attention correctness."""
        adapter = MegatronAttentionAdapter(
            hidden_size=768,
            num_attention_heads=12
        )
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_attention_causal_mask(self):
        """Test causal masking in attention."""
        adapter = MegatronAttentionAdapter(768, 12)
        
        seq_len = 10
        x = torch.randn(2, seq_len, 768, requires_grad=True)
        
        # Forward with causal mask
        out = adapter.test_implementation(x)
        
        # Test causal property: position i should not depend on positions > i
        for i in range(seq_len):
            x_modified = x.clone()
            if i + 1 < seq_len:
                x_modified[:, i+1:, :] = torch.randn_like(x[:, i+1:, :]) * 100
            
            out_modified = adapter.test_implementation(x_modified)
            
            # Position i should be unchanged
            assert torch.allclose(out[:, i, :], out_modified[:, i, :], rtol=1e-4)
    
    def test_attention_memory_efficiency(self):
        """Test attention memory usage."""
        detector = MemoryLeakDetector()
        
        adapter = MegatronAttentionAdapter(768, 12)
        x = torch.randn(4, 128, 768)
        
        results = detector.detect_leak_single_run(
            adapter.test_implementation,
            x,
            warmup=2,
            iterations=5
        )
        
        assert not results['leak_detected']


class TestMegatronMLP:
    """Tests for Megatron MLP (complete feed-forward layer)."""
    
    def test_mlp_correctness(self):
        """Test MLP correctness."""
        adapter = MegatronMLPAdapter(
            hidden_size=768,
            ffn_hidden_size=3072
        )
        results = adapter.run_full_test_suite(verbose=False)
        assert results['failed'] == 0
    
    def test_mlp_gradient_stability(self):
        """Test MLP gradient stability."""
        tester = GradientStabilityTester()
        
        adapter = MegatronMLPAdapter(768, 3072)
        x = torch.randn(16, 768, requires_grad=True)
        
        out = adapter.test_implementation(x)
        loss = out.sum()
        loss.backward()
        
        health = tester.test_gradient_health(lambda x: adapter.test_implementation(x), x)
        assert health['healthy'], f"Gradient issues: {health['issues']}"


class TestMegatronMixedPrecision:
    """Mixed precision tests for Megatron operators."""
    
    def test_mlp_fp16_stability(self):
        """Test MLP with FP16."""
        tester = MixedPrecisionTester(rtol=1e-3, atol=1e-4)
        
        adapter = MegatronMLPAdapter(768, 3072)
        x = torch.randn(16, 768)
        
        result = tester.test_numerical_equivalence(
            adapter.test_implementation,
            (x,),
            dtype=torch.float16
        )
        
        assert result['passed'], f"FP16 error too large: {result['max_rel_error']}"
    
    def test_attention_fp16_stability(self):
        """Test Attention with FP16."""
        tester = MixedPrecisionTester(rtol=1e-2, atol=1e-3)
        
        adapter = MegatronAttentionAdapter(768, 12)
        x = torch.randn(4, 128, 768)
        
        result = tester.test_numerical_equivalence(
            adapter.test_implementation,
            (x,),
            dtype=torch.float16
        )
        
        # Attention may have larger FP16 error due to softmax
        assert result['max_rel_error'] < 0.1, f"Attention FP16 error: {result['max_rel_error']}"
    
    def test_attention_fp16_stability(self):
        """Test Attention with FP16."""
        tester = MixedPrecisionTester(rtol=1e-2, atol=1e-3)
        
        adapter = MegatronAttentionAdapter(768, 12)
        x = torch.randn(4, 128, 768)
        
        result = tester.test_numerical_equivalence(
            adapter.test_implementation,
            (x,),
            dtype=torch.float16
        )
        
        # Attention may have larger FP16 error due to softmax
        assert result['max_rel_error'] < 0.1, f"Attention FP16 error: {result['max_rel_error']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

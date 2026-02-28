"""
Megatron-LM operator adapter for testing.
Tests Megatron-LM specific implementations of core transformer operators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import sys
sys.path.append('/root/.openclaw/workspace')

from operator_test_framework.core.adapter import OperatorTestAdapter


class MegatronLayerNormAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM LayerNorm.
    
    Megatron-LM uses a custom LayerNorm implementation that may differ
    from PyTorch native in terms of precision and fusion optimizations.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5, no_persist_layer_norm: bool = True):
        super().__init__("Megatron_LayerNorm", "normalization")
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.no_persist_layer_norm = no_persist_layer_norm
        
        # Create reference module (PyTorch native)
        self.ref_module = nn.LayerNorm(normalized_shape, eps=eps)
        
        # Create test module (Megatron-style implementation)
        self.test_module = MegatronLayerNorm(normalized_shape, eps=eps)
        
        # Copy parameters for fair comparison
        with torch.no_grad():
            self.test_module.weight.copy_(self.ref_module.weight)
            self.test_module.bias.copy_(self.ref_module.bias)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch native LayerNorm as reference."""
        return self.ref_module(x)
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Megatron-style LayerNorm implementation."""
        return self.test_module(x)
    
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
            
            # Mixed precision simulation (FP16 range)
            x_fp16_range = torch.randn(*shape) * 1e3
            test_cases.append((x_fp16_range,))
        
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
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False
        if not torch.allclose(std, torch.ones_like(std), atol=1e-3):
            return False
        
        return True


class MegatronRMSNormAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM RMSNorm.
    
    RMSNorm is commonly used in LLaMA and other modern LLMs.
    Formula: x / sqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__("Megatron_RMSNorm", "normalization")
        self.dim = dim
        self.eps = eps
        
        # Create test module
        self.test_module = MegatronRMSNorm(dim, eps=eps)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference RMSNorm implementation.
        Formula: x / sqrt(mean(x^2) + eps) * weight
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.test_module.weight.to(x.device)
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Megatron-style RMSNorm implementation."""
        return self.test_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test cases for RMSNorm."""
        test_cases = []
        
        shapes = [
            (1, self.dim),
            (4, self.dim),
            (8, 16, self.dim),
            (2, 4, 8, self.dim),
        ]
        
        for shape in shapes:
            # Standard case
            x = torch.randn(*shape)
            test_cases.append((x,))
            
            # With gradient
            x_grad = torch.randn(*shape, requires_grad=True)
            test_cases.append((x_grad,))
            
            # Near-zero input
            x_small = torch.randn(*shape) * 1e-6
            test_cases.append((x_small,))
            
            # Large values
            x_large = torch.randn(*shape) * 100
            test_cases.append((x_large,))
        
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
        if (self.test_module.weight > 0).all():
            if not torch.all((output >= 0) == (x >= 0)):
                return False
        
        # Property 4: RMS of output should be ~1 (ignoring weight)
        rms_out = torch.sqrt(torch.mean(output ** 2, dim=-1))
        expected_rms = self.test_module.weight.abs().mean()
        if not torch.allclose(rms_out.mean(), expected_rms, atol=1e-3):
            return False
        
        return True


class MegatronMLPAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM MLP with SwiGLU activation.
    
    SwiGLU: swish(xW1) * (xW2)
    Where swish(x) = x * sigmoid(beta * x), beta=1 for Swish/SiLU
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        ffn_hidden_size: int = 3072,
        activation: str = "swiglu"
    ):
        super().__init__("Megatron_MLP", "mlp")
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.activation = activation
        
        # Create test module
        self.test_module = MegatronSwiGLU(hidden_size, ffn_hidden_size)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference SwiGLU implementation.
        SwiGLU(x) = (x @ W_gate) * swish(x @ W_up) @ W_down
        Or: SiLU(xW1) * (xW2) @ W3
        """
        # Get parameters from test module
        w_gate = self.test_module.w_gate.weight
        w_up = self.test_module.w_up.weight
        w_down = self.test_module.w_down.weight
        
        b_gate = self.test_module.w_gate.bias
        b_up = self.test_module.w_up.bias
        b_down = self.test_module.w_down.bias
        
        # Compute gate and up projections
        gate = F.linear(x, w_gate, b_gate)
        up = F.linear(x, w_up, b_up)
        
        # SwiGLU: SiLU(gate) * up
        activated = F.silu(gate) * up
        
        # Down projection
        output = F.linear(activated, w_down, b_down)
        
        return output
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Megatron-style SwiGLU implementation."""
        return self.test_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test cases for MLP."""
        test_cases = []
        
        batch_sizes = [1, 4, 8]
        seq_lengths = [1, 16, 128]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Standard case
                x = torch.randn(batch, seq_len, self.hidden_size)
                test_cases.append((x,))
                
                # With gradient
                x_grad = torch.randn(batch, seq_len, self.hidden_size, requires_grad=True)
                test_cases.append((x_grad,))
                
                # Large activations
                x_large = torch.randn(batch, seq_len, self.hidden_size) * 10
                test_cases.append((x_large,))
                
                # Small activations
                x_small = torch.randn(batch, seq_len, self.hidden_size) * 0.01
                test_cases.append((x_small,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify MLP properties."""
        x = inputs[0]
        
        # Property 1: Output shape is (batch, seq, hidden_size)
        expected_shape = x.shape[:-1] + (self.hidden_size,)
        if output.shape != expected_shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Zero input should produce non-zero output (due to bias)
        # Only check if biases exist
        if self.test_module.w_gate.bias is not None:
            x_zero = torch.zeros_like(x)
            out_zero = self.test_module(x_zero)
            if torch.allclose(out_zero, torch.zeros_like(out_zero)):
                return False
        
        return True


class MegatronRotaryEmbeddingAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM Rotary Position Embedding (RoPE).
    
    RoPE applies rotation to query/key vectors based on position.
    """
    
    def __init__(
        self,
        dim: int = 64,
        max_seq_len: int = 2048,
        base: float = 10000.0
    ):
        super().__init__("Megatron_RotaryEmbedding", "embedding")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Create test module
        self.test_module = MegatronRotaryEmbedding(dim, max_seq_len, base)
    
    def reference_implementation(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Reference RoPE implementation.
        
        Args:
            x: Input tensor of shape (batch, seq, n_heads, head_dim) or (batch, n_heads, seq, head_dim)
            seq_len: Sequence length
            offset: Position offset for cached sequences
        """
        # Compute rotation frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Position indices
        positions = torch.arange(offset, offset + seq_len, device=x.device).float()
        
        # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
        freqs = torch.outer(positions, inv_freq.to(x.device))
        
        # Create complex rotation: cos + i*sin
        cos = torch.cos(freqs)  # (seq_len, dim/2)
        sin = torch.sin(freqs)  # (seq_len, dim/2)
        
        # Apply rotary embedding
        # x shape could be (batch, seq, n_heads, head_dim) or (batch, n_heads, seq, head_dim)
        if x.dim() == 4:
            if x.shape[1] == seq_len:  # (batch, seq, n_heads, head_dim)
                x1, x2 = x[..., 0::2], x[..., 1::2]
                cos = cos.unsqueeze(1)  # (seq, 1, dim/2)
                sin = sin.unsqueeze(1)
                y1 = x1 * cos - x2 * sin
                y2 = x1 * sin + x2 * cos
                return torch.stack([y1, y2], dim=-1).flatten(-2)
            else:  # (batch, n_heads, seq, head_dim)
                x1, x2 = x[..., 0::2], x[..., 1::2]
                cos = cos.unsqueeze(0)  # (1, seq, dim/2)
                sin = sin.unsqueeze(0)
                y1 = x1 * cos - x2 * sin
                y2 = x1 * sin + x2 * cos
                return torch.stack([y1, y2], dim=-1).flatten(-2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
    
    def test_implementation(
        self,
        x: torch.Tensor,
        seq_len: int,
        offset: int = 0
    ) -> torch.Tensor:
        """Megatron-style Rotary Embedding implementation."""
        return self.test_module(x, seq_len, offset)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor, int, int]]:
        """Generate test cases for Rotary Embedding."""
        test_cases = []
        
        batch_sizes = [1, 4]
        n_heads = [8, 16]
        seq_lengths = [16, 64, 128]
        
        for batch in batch_sizes:
            for n_head in n_heads:
                for seq_len in seq_lengths:
                    # Standard case: (batch, seq, n_heads, head_dim)
                    x = torch.randn(batch, seq_len, n_head, self.dim)
                    test_cases.append((x, seq_len, 0))
                    
                    # With gradient
                    x_grad = torch.randn(batch, seq_len, n_head, self.dim, requires_grad=True)
                    test_cases.append((x_grad, seq_len, 0))
                    
                    # Alternative layout: (batch, n_heads, seq, head_dim)
                    x_alt = torch.randn(batch, n_head, seq_len, self.dim)
                    test_cases.append((x_alt, seq_len, 0))
                    
                    # With offset (for cached generation)
                    if seq_len > 8:
                        x_offset = torch.randn(batch, seq_len, n_head, self.dim)
                        test_cases.append((x_offset, seq_len, 64))
        
        return test_cases
    
    def verify_properties(
        self,
        output: torch.Tensor,
        inputs: Tuple[torch.Tensor, int, int]
    ) -> bool:
        """Verify Rotary Embedding properties."""
        x, seq_len, offset = inputs
        
        # Property 1: Output shape matches input
        if output.shape != x.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Norm preservation - RoPE preserves vector norm
        input_norm = torch.norm(x, dim=-1)
        output_norm = torch.norm(output, dim=-1)
        if not torch.allclose(input_norm, output_norm, atol=1e-5):
            return False
        
        # Property 4: Relative position encoding property
        # For positions i and j, the rotation angle difference should be consistent
        if seq_len >= 2:
            # Check that different positions have different encodings
            pos_0 = output[:, 0, ...] if output.shape[1] == seq_len else output[..., 0, :]
            pos_1 = output[:, 1, ...] if output.shape[1] == seq_len else output[..., 1, :]
            if torch.allclose(pos_0, pos_1):
                return False
        
        return True


class MegatronAttentionAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM Attention (basic version, non-parallel).
    
    Tests the core attention computation without tensor parallelism.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_query_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__("Megatron_Attention", "attention")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups or num_attention_heads
        self.attention_dropout = attention_dropout
        self.causal = causal
        
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Create test module
        self.test_module = MegatronAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            attention_dropout=attention_dropout,
            causal=causal
        )
    
    def reference_implementation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reference attention implementation.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional mask
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        qkv = F.linear(hidden_states, self.test_module.qkv.weight, self.test_module.qkv.bias)
        
        # Split into Q, K, V
        # For grouped query attention
        num_kv_heads = self.num_query_groups
        q_size = self.num_attention_heads * self.head_dim
        kv_size = num_kv_heads * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        
        # Reshape for multi-head attention
        # (batch, seq, num_heads * head_dim) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle grouped query attention - repeat K, V if needed
        if num_kv_heads < self.num_attention_heads:
            num_repeats = self.num_attention_heads // num_kv_heads
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, num_heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = F.linear(attn_output, self.test_module.o_proj.weight, self.test_module.o_proj.bias)
        
        return output
    
    def test_implementation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Megatron-style Attention implementation."""
        return self.test_module(hidden_states, attention_mask)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Generate test cases for Attention."""
        test_cases = []
        
        batch_sizes = [1, 4]
        seq_lengths = [16, 64, 128]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Standard case
                x = torch.randn(batch, seq_len, self.hidden_size)
                test_cases.append((x, None))
                
                # With gradient
                x_grad = torch.randn(batch, seq_len, self.hidden_size, requires_grad=True)
                test_cases.append((x_grad, None))
                
                # With attention mask
                mask = torch.randn(batch, 1, 1, seq_len) * -1e4  # Simulated padding mask
                x_mask = torch.randn(batch, seq_len, self.hidden_size)
                test_cases.append((x_mask, mask))
        
        return test_cases
    
    def verify_properties(
        self,
        output: torch.Tensor,
        inputs: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> bool:
        """Verify Attention properties."""
        hidden_states, attention_mask = inputs
        
        # Property 1: Output shape matches input
        if output.shape != hidden_states.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Causal property - position i only attends to positions <= i
        if self.causal:
            batch_size, seq_len, _ = hidden_states.shape
            
            # Modify future values and check position i is unchanged
            for i in range(min(3, seq_len)):
                modified = hidden_states.clone()
                if i + 1 < seq_len:
                    modified[:, i+1:, :] = torch.randn_like(modified[:, i+1:, :]) * 100
                
                out_original = self.test_implementation(hidden_states, attention_mask)
                out_modified = self.test_implementation(modified, attention_mask)
                
                # Position i should be unchanged
                if not torch.allclose(out_original[:, i, :], out_modified[:, i, :], rtol=1e-4, atol=1e-5):
                    return False
        
        return True


class MegatronLinearAdapter(OperatorTestAdapter):
    """
    Test adapter for Megatron-LM Linear layer (non-parallel version).
    
    Tests the basic linear transformation without tensor parallelism.
    """
    
    def __init__(
        self,
        in_features: int = 768,
        out_features: int = 3072,
        bias: bool = True,
        skip_bias_add: bool = False
    ):
        super().__init__("Megatron_Linear", "linear")
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.skip_bias_add = skip_bias_add
        
        # Create test module
        self.test_module = MegatronLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            skip_bias_add=skip_bias_add
        )
        
        # Reference module
        self.ref_module = nn.Linear(in_features, out_features, bias=bias)
        
        # Copy parameters
        with torch.no_grad():
            self.ref_module.weight.copy_(self.test_module.weight)
            if bias:
                self.ref_module.bias.copy_(self.test_module.bias)
    
    def reference_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch native Linear as reference."""
        return self.ref_module(x)
    
    def test_implementation(self, x: torch.Tensor) -> torch.Tensor:
        """Megatron-style Linear implementation."""
        return self.test_module(x)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor]]:
        """Generate test cases for Linear."""
        test_cases = []
        
        batch_sizes = [1, 4, 16]
        seq_lengths = [1, 16, 128]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Standard case
                x = torch.randn(batch, seq_len, self.in_features)
                test_cases.append((x,))
                
                # With gradient
                x_grad = torch.randn(batch, seq_len, self.in_features, requires_grad=True)
                test_cases.append((x_grad,))
                
                # 2D input
                x_2d = torch.randn(batch * seq_len, self.in_features)
                test_cases.append((x_2d,))
                
                # Large values
                x_large = torch.randn(batch, seq_len, self.in_features) * 100
                test_cases.append((x_large,))
                
                # Small values
                x_small = torch.randn(batch, seq_len, self.in_features) * 0.01
                test_cases.append((x_small,))
        
        return test_cases
    
    def verify_properties(self, output: torch.Tensor, inputs: Tuple[torch.Tensor]) -> bool:
        """Verify Linear properties."""
        x = inputs[0]
        
        # Property 1: Output shape is correct
        expected_shape = x.shape[:-1] + (self.out_features,)
        if output.shape != expected_shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Linearity property
        # f(a*x + b*y) = a*f(x) + b*f(y)
        a, b = 2.0, 3.0
        x1 = torch.randn_like(x)
        x2 = torch.randn_like(x)
        
        left = self.test_module(a * x1 + b * x2)
        right = a * self.test_module(x1) + b * self.test_module(x2)
        
        if not torch.allclose(left, right, rtol=1e-4, atol=1e-5):
            return False
        
        # Property 4: Zero input should produce bias (if bias exists)
        if self.bias:
            x_zero = torch.zeros_like(x)
            out_zero = self.test_module(x_zero)
            expected_bias = self.test_module.bias
            
            # Check that output equals bias broadcasted
            if not torch.allclose(out_zero, expected_bias.view(1, 1, -1).expand_as(out_zero), rtol=1e-4):
                return False
        
        return True


# ============================================================================
# Megatron-style implementations (simplified versions for testing)
# ============================================================================

class MegatronLayerNorm(nn.Module):
    """Megatron-style LayerNorm implementation."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual LayerNorm implementation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class MegatronRMSNorm(nn.Module):
    """Megatron-style RMSNorm implementation."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight


class MegatronSwiGLU(nn.Module):
    """Megatron-style SwiGLU MLP implementation."""
    
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        
        # SwiGLU has two up-projections: gate and up
        self.w_gate = nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.w_up = nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.w_down = nn.Linear(ffn_hidden_size, hidden_size, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        activated = gate * up
        return self.w_down(activated)


class MegatronRotaryEmbedding(nn.Module):
    """Megatron-style Rotary Position Embedding implementation."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for common sequence lengths
        self._cached_seq_len = 0
        self._cached_cos = None
        self._cached_sin = None
    
    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embedding to input.
        
        Args:
            x: Input tensor (batch, seq, n_heads, head_dim) or (batch, n_heads, seq, head_dim)
            seq_len: Sequence length
            offset: Position offset
        """
        # Compute or retrieve cached cos/sin
        if seq_len + offset > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = max(seq_len + offset, self.max_seq_len)
            positions = torch.arange(0, self._cached_seq_len, device=x.device).float()
            freqs = torch.outer(positions, self.inv_freq.to(x.device))
            self._cached_cos = torch.cos(freqs)
            self._cached_sin = torch.sin(freqs)
        
        # Get cos/sin for the relevant positions
        cos = self._cached_cos[offset:offset + seq_len]  # (seq_len, dim/2)
        sin = self._cached_sin[offset:offset + seq_len]
        
        # Apply rotary embedding
        # Handle different input layouts
        if x.dim() == 4:
            if x.shape[1] == seq_len:  # (batch, seq, n_heads, head_dim)
                x1, x2 = x[..., 0::2], x[..., 1::2]
                cos = cos.unsqueeze(1)  # (seq, 1, dim/2)
                sin = sin.unsqueeze(1)
                y1 = x1 * cos - x2 * sin
                y2 = x1 * sin + x2 * cos
                return torch.stack([y1, y2], dim=-1).flatten(-2)
            else:  # (batch, n_heads, seq, head_dim)
                x1, x2 = x[..., 0::2], x[..., 1::2]
                cos = cos.unsqueeze(0)  # (1, seq, dim/2)
                sin = sin.unsqueeze(0)
                y1 = x1 * cos - x2 * sin
                y2 = x1 * sin + x2 * cos
                return torch.stack([y1, y2], dim=-1).flatten(-2)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")


class MegatronAttention(nn.Module):
    """Megatron-style Attention implementation (basic, non-parallel)."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_query_groups: Optional[int] = None,
        attention_dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups or num_attention_heads
        self.attention_dropout = attention_dropout
        self.causal = causal
        
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # QKV projection - fused for efficiency
        q_size = hidden_size
        kv_size = self.head_dim * self.num_query_groups
        self.qkv = nn.Linear(hidden_size, q_size + 2 * kv_size, bias=True)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv(hidden_states)
        
        # Split into Q, K, V
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_query_groups * self.head_dim
        
        q = qkv[..., :q_size]
        k = qkv[..., q_size:q_size + kv_size]
        v = qkv[..., q_size + kv_size:]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_query_groups, self.head_dim).transpose(1, 2)
        
        # Handle grouped query attention
        if self.num_query_groups < self.num_attention_heads:
            num_repeats = self.num_attention_heads // self.num_query_groups
            k = k.repeat_interleave(num_repeats, dim=1)
            v = v.repeat_interleave(num_repeats, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Dropout
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return output


class MegatronLinear(nn.Module):
    """Megatron-style Linear layer (non-parallel version)."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        skip_bias_add: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_bias_add = skip_bias_add
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight, None if self.skip_bias_add else self.bias)
        return output

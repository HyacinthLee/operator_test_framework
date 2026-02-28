"""
Attention operator adapter for testing.
"""

import torch
import math
from typing import List, Tuple, Optional
import sys
sys.path.append('/root/.openclaw/workspace')
from core.adapter import OperatorTestAdapter


class AttentionAdapter(OperatorTestAdapter):
    """
    Test adapter for Attention operators.
    
    Supports testing various Attention implementations including:
    - Standard PyTorch attention
    - Custom optimized attention
    - FlashAttention variants
    """
    
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        causal: bool = False,
        scale: Optional[float] = None
    ):
        super().__init__("Attention", "attention")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.scale = scale or 1.0 / math.sqrt(self.head_dim)
    
    def reference_implementation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Reference: PyTorch native scaled dot-product attention.
        """
        # Reshape for multi-head: (batch, seq, dim) -> (batch, heads, seq, head_dim)
        batch_size, seq_len, _ = query.shape
        
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return output
    
    def test_implementation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Test implementation - replace with your custom attention.
        
        For demonstration, this uses the reference implementation.
        In practice, replace this with your optimized version.
        """
        # TODO: Replace with your custom attention implementation
        return self.reference_implementation(query, key, value)
    
    def generate_test_cases(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate test cases for attention."""
        test_cases = []
        
        batch_sizes = [1, 4]
        seq_lengths = [16, 64, 256]
        
        for batch in batch_sizes:
            for seq_len in seq_lengths:
                # Standard case
                q = torch.randn(batch, seq_len, self.dim)
                k = torch.randn(batch, seq_len, self.dim)
                v = torch.randn(batch, seq_len, self.dim)
                test_cases.append((q, k, v))
                
                # Requires grad for gradient testing
                q_grad = torch.randn(batch, seq_len, self.dim, requires_grad=True)
                k_grad = torch.randn(batch, seq_len, self.dim, requires_grad=True)
                v_grad = torch.randn(batch, seq_len, self.dim, requires_grad=True)
                test_cases.append((q_grad, k_grad, v_grad))
                
                # Different seq lengths (cross-attention scenario)
                if seq_len > 16:
                    kv_len = seq_len // 2
                    q_cross = torch.randn(batch, seq_len, self.dim)
                    k_cross = torch.randn(batch, kv_len, self.dim)
                    v_cross = torch.randn(batch, kv_len, self.dim)
                    test_cases.append((q_cross, k_cross, v_cross))
        
        return test_cases
    
    def verify_properties(
        self,
        output: torch.Tensor,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> bool:
        """Verify attention-specific properties."""
        query, key, value = inputs
        
        # Property 1: Output shape matches query shape
        if output.shape != query.shape:
            return False
        
        # Property 2: Output is finite
        if not torch.isfinite(output).all():
            return False
        
        # Property 3: Causal property - position i only attends to positions <= i
        if self.causal:
            batch_size, seq_len, _ = query.shape
            for i in range(min(5, seq_len)):  # Check first 5 positions
                # Modify future values in value
                v_modified = value.clone()
                if i + 1 < seq_len:
                    v_modified[:, i+1:, :] = torch.randn_like(v_modified[:, i+1:, :]) * 100
                
                output_modified = self.test_implementation(query, key, v_modified)
                
                # Position i should be unchanged
                if not torch.allclose(output[:, i, :], output_modified[:, i, :], rtol=1e-4):
                    return False
        
        return True


class FlashAttentionAdapter(AttentionAdapter):
    """
    Adapter for testing FlashAttention implementations.
    """
    
    def __init__(self, dim: int = 64, num_heads: int = 8, causal: bool = False):
        super().__init__(dim, num_heads, causal)
        self.name = "FlashAttention"
    
    def test_implementation(self, query, key, value):
        """
        Placeholder for FlashAttention implementation.
        
        To use:
        1. Install flash-attn: pip install flash-attn
        2. Uncomment and use the flash_attn_func
        """
        # try:
        #     from flash_attn import flash_attn_func
        #     return flash_attn_func(query, key, value, causal=self.causal)
        # except ImportError:
        #     print("flash-attn not installed, using reference implementation")
        #     return self.reference_implementation(query, key, value)
        
        # For now, use reference
        return self.reference_implementation(query, key, value)

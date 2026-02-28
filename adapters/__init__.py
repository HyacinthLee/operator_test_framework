"""
Adapter module initialization.
"""

from .attention_adapter import AttentionAdapter, FlashAttentionAdapter
from .normalization_adapter import LayerNormAdapter, RMSNormAdapter

try:
    from .transformer_engine_adapter import (
        TransformerEngineLinearAdapter,
        TransformerEngineLayerNormAdapter,
        TransformerEngineTransformerLayerAdapter,
    )
    TE_ADAPTERS_AVAILABLE = True
except ImportError:
    TE_ADAPTERS_AVAILABLE = False

__all__ = [
    "AttentionAdapter",
    "FlashAttentionAdapter",
    "LayerNormAdapter",
    "RMSNormAdapter",
]

if TE_ADAPTERS_AVAILABLE:
    __all__.extend([
        "TransformerEngineLinearAdapter",
        "TransformerEngineLayerNormAdapter",
        "TransformerEngineTransformerLayerAdapter",
    ])

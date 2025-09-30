"""
FlashAttention 2 implementation for Order Book Transformer.
This module provides the core attention mechanism using FlashAttention 2.

Author: Track A Developer
Interface Version: 1.0
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from flash_attn import flash_attn_func, flash_attn_varlen_func

class FlashOrderBookAttention(nn.Module):
    """
    Order Book Attention using FlashAttention 2 exclusively.
    
    Interface Contract:
    - Input: [batch_size, seq_len, d_model]
    - Output: (output, None) where output is [batch_size, seq_len, d_model]
    - Attention weights are NOT returned for efficiency
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, 
                 use_varlen_attention: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_varlen_attention = use_varlen_attention
        
        # Fused QKV projection for maximum efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                rotary_emb: Optional[nn.Module] = None) -> Tuple[torch.Tensor, None]:
        """Standard forward pass with optional rotary embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            rotary_emb: Optional rotary embedding module to apply to Q and K
            
        Returns:
            Tuple of (output, None) where output is [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Store original dtype for output
        original_dtype = x.dtype
        
        # Project to Q, K, V first (keeps original dtype)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)  # Each is [B, L, H, D]
        
        # Apply rotary embeddings if provided (before dtype conversion)
        if rotary_emb is not None:
            # Rotary embeddings expect [batch, heads, seq, dim] format
            q_rot = q.permute(0, 2, 1, 3)  # [B, H, L, D]
            k_rot = k.permute(0, 2, 1, 3)  # [B, H, L, D]
            
            # Apply rotary embeddings
            q_rot = rotary_emb(q_rot)
            k_rot = rotary_emb(k_rot)
            
            # Permute back to [B, L, H, D]
            q = q_rot.permute(0, 2, 1, 3)
            k = k_rot.permute(0, 2, 1, 3)
        
        # Always use bf16 for Flash Attention (no fallback)
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16) 
            v = v.to(torch.bfloat16)
        
        # Choose attention implementation based on configuration
        if self.use_varlen_attention:
            # Use variable-length attention with explicit sample boundaries
            output = self._forward_varlen(q, k, v, batch_size, seq_len)
        else:
            # Use standard FlashAttention
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.d_k),
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False
            )
            # Reshape output
            output = output.reshape(batch_size, seq_len, self.d_model)
        
        # Convert back to original dtype before projection to match linear layer weights
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        
        # Project output
        output = self.out_proj(output)
        
        return output, None
    
    def _forward_varlen(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Forward pass using variable-length attention with explicit sample boundaries.
        This ensures complete isolation between samples in the batch.
        
        Args:
            q, k, v: Query, Key, Value tensors [batch_size, seq_len, n_heads, head_dim]
            batch_size: Number of samples in batch
            seq_len: Sequence length per sample
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Reshape to concatenated format for varlen attention
        # Concatenate all sequences into one long sequence
        q_concat = q.reshape(batch_size * seq_len, self.n_heads, self.d_k)
        k_concat = k.reshape(batch_size * seq_len, self.n_heads, self.d_k)
        v_concat = v.reshape(batch_size * seq_len, self.n_heads, self.d_k)
        
        # Create cumulative sequence lengths
        # This defines the boundaries between samples
        # cu_seqlens[i] to cu_seqlens[i+1] defines sample i's sequence
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len,
            dtype=torch.int32, device=q.device
        )
        
        # Apply variable-length FlashAttention with explicit boundaries
        # Each sample can only attend within its own sequence boundaries
        output = flash_attn_varlen_func(
            q_concat, k_concat, v_concat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=1.0 / math.sqrt(self.d_k),
            causal=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False
        )
        
        # Reshape back to batch format
        output = output.reshape(batch_size, seq_len, self.d_model)
        
        return output


class EfficientFeatureAttention(nn.Module):
    """
    Efficient alternative to feature-dimension attention.
    Uses grouped convolutions and gating for feature mixing.
    
    Interface Contract:
    - Input: [batch_size, seq_len, d_model]
    - Output: [batch_size, seq_len, d_model]
    """
    
    def __init__(self, d_model: int, n_groups: int = 8):
        super().__init__()
        self.n_groups = n_groups
        self.group_conv = nn.Conv1d(d_model, d_model, kernel_size=1, groups=n_groups)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Apply grouped convolution for feature mixing
        x_conv = x.transpose(1, 2)
        x_mixed = self.group_conv(x_conv).transpose(1, 2)
        
        # Apply gating mechanism
        gate_values = self.gate(x.mean(dim=1, keepdim=True))
        
        # Ensure dtypes match before addition
        if x_mixed.dtype != x.dtype:
            x_mixed = x_mixed.to(x.dtype)
        if gate_values.dtype != x.dtype:
            gate_values = gate_values.to(x.dtype)
        
        return x + x_mixed * gate_values
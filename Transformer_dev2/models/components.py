import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import FlashAttention components
from .flash_attention import FlashOrderBookAttention

# Import TFT components
from .tft_components import GatedResidualNetwork, PositionwiseGRN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for Transformers.
    
    RoPE encodes absolute positional information with rotation matrix and 
    naturally incorporates relative position dependency in self-attention.
    
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Create inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache for fast lookup
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )
        
    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        """Precompute cos and sin values for rotary embeddings."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim] 
               or [seq_len, batch_size, num_heads, head_dim]
            seq_len: Sequence length (if different from x.shape)
            
        Returns:
            Tensor with rotary embeddings applied
        """
        # Support both [batch, heads, seq, dim] and [seq, batch, heads, dim] formats
        if x.ndim == 4:
            if seq_len is None:
                # Determine seq_len based on tensor format
                # For [batch, heads, seq, dim]: seq is at index 2
                # For [seq, batch, heads, dim]: seq is at index 0
                # We can distinguish by checking if dim 2 is likely seq_len (larger)
                if x.shape[2] > x.shape[0]:  # Likely [batch, heads, seq, dim]
                    seq_len = x.shape[2]
                else:  # Likely [seq, batch, heads, dim]
                    seq_len = x.shape[0]
        else:
            raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
            
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
            
        return self.apply_rotary_pos_emb(x, self.cos_cached[:seq_len], self.sin_cached[:seq_len])
    
    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary position embeddings to input tensors."""
        # cos and sin are already [seq_len, dim]
        # x is [batch, heads, seq, dim] or [seq, batch, heads, dim]
        
        # Handle different input shapes
        if x.ndim == 4 and x.shape[2] > x.shape[0]:  # [batch, heads, seq, dim] format
            # Reshape cos and sin to match x's shape
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        else:  # [seq, batch, heads, dim] format - less common
            cos = cos.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1, dim]
            sin = sin.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1, dim]
            
        # Apply rotation using the formula: x_rot = x * cos + rotate_half(x) * sin
        x_embed = (x * cos) + (self.rotate_half(x) * sin)
        return x_embed




class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, 
                 activation: str = 'gelu'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x




class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer using FlashAttention 2.
    
    Interface Contract:
    - Must maintain compatibility with existing transformer.py
    - Input format: [seq_len, batch_size, d_model] (note: seq_len first)
    - Output format: (output, None) where output is [seq_len, batch_size, d_model]
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, activation: str = 'gelu', 
                 layer_norm_eps: float = 1e-5, use_grn: bool = False,
                 use_varlen_attention: bool = True):
        super().__init__()
        
        # Use FlashAttention exclusively with optional varlen for sample isolation
        self.self_attn = FlashOrderBookAttention(d_model, n_heads, dropout, 
                                                  use_varlen_attention=use_varlen_attention)
        
        # Choose between GRN and standard FFN based on config
        if use_grn:
            self.feed_forward = PositionwiseGRN(d_model, d_ff, dropout)
        else:
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        
        # Pre-norm architecture for better stability with FlashAttention
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                rotary_emb: Optional[nn.Module] = None) -> Tuple[torch.Tensor, None]:
        # Input: [seq_len, batch_size, d_model]
        # FlashAttention expects [batch_size, seq_len, d_model], so transpose for attention only
        
        # Pre-norm + self-attention + residual
        src_transposed = src.transpose(0, 1)  # [batch_size, seq_len, d_model]
        src2 = self.norm1(src_transposed)
        attn_output, _ = self.self_attn(src2, src_mask, rotary_emb=rotary_emb)
        src_transposed = src_transposed + self.dropout1(attn_output)
        
        # Pre-norm + feed-forward + residual
        src2 = self.norm2(src_transposed)
        ff_output = self.feed_forward(src2)
        src_transposed = src_transposed + self.dropout2(ff_output)
        
        # Transpose back to original format
        src = src_transposed.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        return src, None


class OrderBookEmbedding(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale convolutional feature extractor for capturing patterns at different temporal scales.
    """

    def __init__(self, d_model: int, kernel_sizes: list = [3, 5, 7], dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes

        # Create multiple conv layers with different kernel sizes
        # Calculate output channels for each conv layer to ensure sum equals d_model
        num_kernels = len(kernel_sizes)
        base_channels = d_model // num_kernels
        remainder = d_model % num_kernels

        self.conv_layers = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # Distribute remainder channels to first few layers
            out_channels = base_channels + (1 if i < remainder else 0)
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2  # Same padding to maintain sequence length
                )
            )

        # Combine features from different scales
        self.combine_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Multi-scale features of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Transpose for conv1d: [batch_size, d_model, seq_len]
        x_transposed = x.transpose(1, 2)

        # Apply convolutions at different scales
        multi_scale_features = []
        for conv in self.conv_layers:
            conv_out = conv(x_transposed)  # [batch_size, d_model//n, seq_len]
            multi_scale_features.append(conv_out)

        # Concatenate features from different scales
        combined = torch.cat(multi_scale_features, dim=1)  # [batch_size, d_model, seq_len]

        # Transpose back: [batch_size, seq_len, d_model]
        combined = combined.transpose(1, 2)

        # Project and normalize
        output = self.combine_projection(combined)
        output = self.activation(output)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


"""
Temporal Fusion Transformer (TFT) components for enhancing the transformer architecture.
These components are adapted from the TFT paper while maintaining compatibility with
binary classification and the existing data pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit - core gating mechanism used in GRN."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split input through two paths
        hidden = self.dropout(self.fc1(x))
        gate = self.sigmoid(self.fc2(x))
        return hidden * gate


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Core building block of TFT.
    Replaces standard feedforward networks with gating mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
        use_time_distributed: bool = True,
        return_gate: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model
        self.return_gate = return_gate
        self.use_time_distributed = use_time_distributed
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # First linear projection (optional - if d_hidden != d_model)
        if self.d_model != self.d_hidden:
            self.fc1 = nn.Linear(d_model, d_hidden)
        else:
            self.fc1 = None
            
        # ELU activation
        self.elu = nn.ELU()
        
        # Second linear projection
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        
        # Gated Linear Unit
        self.glu = GatedLinearUnit(d_hidden, dropout)
        
        # Final projection back to d_model
        if self.d_hidden != d_model:
            self.fc3 = nn.Linear(d_hidden, d_model)
        else:
            self.fc3 = None
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model] or [batch_size, d_model]
            context: Optional context tensor for conditional GRN
            
        Returns:
            Output tensor with same shape as input
        """
        # Store original input for residual connection
        residual = x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # First projection (if needed)
        if self.fc1 is not None:
            x = self.fc1(x)
            
        # ELU activation
        x = self.elu(x)
        
        # Second projection
        x = self.fc2(x)
        x = self.elu(x)
        
        # Gated linear unit
        x = self.glu(x)
        
        # Final projection (if needed)
        if self.fc3 is not None:
            x = self.fc3(x)
            
        # Dropout
        x = self.dropout(x)
        
        # Residual connection
        return x + residual


class TemporalFusionDecoder(nn.Module):
    """
    Temporal Fusion Decoder - Combines temporal patterns for final prediction.
    Uses multi-head attention to aggregate information across time steps.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_grn: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_grn = use_grn
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Multi-head attention for temporal fusion
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Position-wise GRN or FFN
        if use_grn:
            self.position_wise = GatedResidualNetwork(
                d_model=d_model,
                d_hidden=d_model * 4,
                dropout=dropout
            )
        else:
            # Fallback to standard FFN
            self.position_wise = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            
        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable query for temporal aggregation
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            - Fused output tensor of shape [batch_size, seq_len, d_model]
            - Attention weights of shape [batch_size, n_heads, 1, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Expand temporal query to batch size
        query = self.temporal_query.expand(batch_size, -1, -1)
        
        # Apply temporal attention
        # Use the learned query to attend to all time steps
        attended, attention_weights = self.temporal_attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=mask,
            need_weights=True
        )
        
        # Residual connection and normalization
        x_attended = self.norm1(attended + query)
        
        # Position-wise transformation
        if self.use_grn:
            output = self.position_wise(x_attended)
        else:
            output = x_attended + self.position_wise(x_attended)
            
        output = self.norm2(output)
        
        # Expand the aggregated representation back to sequence length
        # This allows the classification head to use the temporally fused information
        output = output.expand(-1, seq_len, -1)
        
        # Combine with original sequence via residual connection
        output = output + x
        
        return output, attention_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention with enhanced visibility into attention patterns.
    Can be used as a drop-in replacement for standard attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_grn: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # GRN for value transformation (TFT-style)
        if use_grn:
            self.value_grn = GatedResidualNetwork(
                d_model=d_model,
                d_hidden=d_model,
                dropout=dropout
            )
        else:
            self.value_grn = nn.Identity()
            
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query, key, value: Input tensors of shape [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            - Output tensor of shape [batch_size, seq_len, d_model]
            - Attention weights (if available)
        """
        # Apply GRN to values for better feature extraction
        value_transformed = self.value_grn(value)
        
        # Apply attention
        output, weights = self.attention(
            query=query,
            key=key,
            value=value_transformed,
            attn_mask=mask,
            need_weights=True
        )
        
        # Store weights for interpretability
        self.last_attention_weights = weights.detach()
        
        # Residual connection and normalization
        output = self.norm(output + query)
        
        return output, weights


class PositionwiseGRN(nn.Module):
    """
    Position-wise Gated Residual Network.
    Drop-in replacement for PositionwiseFeedForward in transformer layers.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.grn = GatedResidualNetwork(
            d_model=d_model,
            d_hidden=d_ff,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model] or
               [batch_size, seq_len, d_model]
               
        Returns:
            Output tensor with same shape as input
        """
        # GRN handles both 2D and 3D inputs naturally
        return self.grn(x)
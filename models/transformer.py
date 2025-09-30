import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.checkpoint import checkpoint

# Import all components from unified components.py
from .components import (
    TransformerEncoderLayer,
    OrderBookEmbedding,
    PositionalEncoding,
    RotaryEmbedding
)

# Import TFT components
from .tft_components import TemporalFusionDecoder

# Config imports removed - using dict-like access



class OrderBookTransformer(nn.Module):
    def __init__(self, config: Optional[Dict] = None, binary_classification_config=None):
        super().__init__()

        # Config is now expected to be passed in as a dict
        if config is None:
            raise ValueError("Config must be provided")
        self.config = config
        self.binary_classification_config = binary_classification_config
        self.gradient_checkpointing = False

        # Store key dimensions as attributes for easy access
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers

        # Determine classification mode (default to binary for backward compatibility)
        self.classification_mode = getattr(config, 'classification_mode', 'binary')
        if self.classification_mode not in ['binary', 'ternary']:
            raise ValueError(f"Invalid classification_mode: {self.classification_mode}. Must be 'binary' or 'ternary'")
        
        # Input embedding layer
        # n_features can be set dynamically or from config
        # If not in config, it will be inferred from the first forward pass
        self.input_features = getattr(config, 'n_features', None)

        # We'll create the embedding layer lazily if n_features is not provided
        if self.input_features is not None:
            self.embedding = OrderBookEmbedding(
                input_dim=self.input_features,
                d_model=config.d_model,
                dropout=config.dropout
            )
        else:
            # Will be created on first forward pass
            self.embedding = None
            self._embedding_config = {
                'd_model': config.d_model,
                'dropout': config.dropout
            }
        
        # Choose position encoding based on configuration
        self.use_rope = config.use_rope if hasattr(config, 'use_rope') else True  # Default to RoPE
        
        # Only create the positional encoding module that will be used
        # This prevents DDP unused parameters issues
        if self.use_rope:
            head_dim = config.d_model // config.n_heads
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_sequence_length if hasattr(config, 'max_sequence_length') and config.max_sequence_length else 4096,
                base=config.rope_theta if hasattr(config, 'rope_theta') else 10000.0
            )
            self.pos_encoder = None
        else:
            self.rotary_emb = None
            self.pos_encoder = PositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length if hasattr(config, 'max_sequence_length') and config.max_sequence_length else 1024,
                dropout=config.dropout
            )
        
        # Check if we should use TFT enhancements
        self.use_grn = config.use_grn if hasattr(config, 'use_grn') else False
        self.use_temporal_fusion = config.use_temporal_fusion if hasattr(config, 'use_temporal_fusion') else False
        
        # Check if we should use variable-length attention for sample isolation
        self.use_varlen_attention = config.use_varlen_attention if hasattr(config, 'use_varlen_attention') else True
        
        # Transformer encoder layers using FlashAttention
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.feedforward_dim,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                use_grn=self.use_grn,  # Pass GRN flag to encoder layers
                use_varlen_attention=self.use_varlen_attention  # Pass varlen flag for sample isolation
            )
            for _ in range(config.n_layers)
        ])
        
        
        # Only create Temporal Fusion Decoder if it will be used
        # This prevents DDP unused parameters issues
        if self.use_temporal_fusion:
            self.temporal_decoder = TemporalFusionDecoder(
                d_model=config.d_model,
                n_heads=max(config.n_heads // 2, 1),  # Use fewer heads for efficiency
                dropout=config.dropout,
                use_grn=self.use_grn
            )
        else:
            self.temporal_decoder = None

        # Classification head based on mode
        num_classes = 3 if self.classification_mode == 'ternary' else 1
        self.prediction_head = ClassificationHead(
            config.d_model,
            dropout=config.dropout,
            num_classes=num_classes,
            classification_mode=self.classification_mode
        )
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the transformer encoder layers."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the transformer encoder layers."""
        self.gradient_checkpointing = False
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer model.
        
        TEMPORAL ORDERING CONVENTION:
        The input tensor x has shape [batch_size, seq_len, features] where the sequence dimension
        follows reverse chronological order:
        - x[:, 0, :] = Most recent timestep (current time, offset=0 from reference position)
        - x[:, 1, :] = 1 timestep in the past
        - x[:, 2, :] = 2 timesteps in the past
        - ...
        - x[:, -1, :] = Oldest timestep (maximum lookback)
        
        This ordering is established by the window creation process where sample_offsets=[0,1,2,...]
        maps to windows[:, i, :] = data[position - offset], making index 0 the current position.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
            mask: Optional attention mask
            
        Returns:
            Dict containing 'logits' and 'hidden_states'
        """
        # Input shape: [batch_size, seq_len, n_features]
        assert x.dim() == 3, f"Expected 3D input tensor, got {x.dim()}D"
        batch_size, seq_len, n_features = x.shape

        # Create embedding layer on first forward pass if not already created
        if self.embedding is None:
            self.input_features = n_features
            self.embedding = OrderBookEmbedding(
                input_dim=n_features,
                d_model=self._embedding_config['d_model'],
                dropout=self._embedding_config['dropout']
            )
            # Move to same device as input
            self.embedding = self.embedding.to(x.device)
        else:
            assert n_features == self.input_features, (
                f"Input features {n_features} != expected {self.input_features}"
            )

        assert seq_len <= self.config.max_sequence_length, (
            f"Sequence length {seq_len} > max_sequence_length {self.config.max_sequence_length}"
        )


        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        assert x.shape == (batch_size, seq_len, self.d_model), (
            f"After embedding shape {x.shape} != expected ({batch_size}, {seq_len}, {self.d_model})"
        )
        
        
        # Transpose for transformer (expects seq_len first)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        assert x.shape == (seq_len, batch_size, self.d_model), (
            f"After transpose shape {x.shape} != expected ({seq_len}, {batch_size}, {self.d_model})"
        )
        
        # Add positional encoding (if not using RoPE)
        if not self.use_rope and self.pos_encoder is not None:
            x = self.pos_encoder(x)
        
        # Pass through transformer encoder layers
        # Note: FlashAttention doesn't return attention weights for efficiency
        for i, layer in enumerate(self.encoder_layers):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(x, mask):
                        return module(x, src_mask=mask, rotary_emb=self.rotary_emb if self.use_rope and self.rotary_emb is not None else None)
                    return custom_forward
                
                x, _ = checkpoint(create_custom_forward(layer), x, mask, use_reentrant=False)
            else:
                x, _ = layer(x, src_mask=mask, rotary_emb=self.rotary_emb if self.use_rope and self.rotary_emb is not None else None)
        
        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        assert x.shape == (batch_size, seq_len, self.d_model), (
            f"After final transpose shape {x.shape} != expected ({batch_size}, {seq_len}, {self.d_model})"
        )
        
        # Apply temporal fusion decoder if enabled
        if self.use_temporal_fusion and self.temporal_decoder is not None:
            x, _ = self.temporal_decoder(x, mask=mask)
        
        # Generate predictions based on task
        predictions = self.prediction_head(x)

        # Verify output shape based on classification mode
        expected_shape = (batch_size, 3) if self.classification_mode == 'ternary' else (batch_size, 1)
        assert predictions.shape == expected_shape, (
            f"Predictions shape {predictions.shape} != expected {expected_shape}"
        )

        # Store last output for callback access (detached to prevent memory leak)
        output = {
            'logits': predictions,  # [batch_size, num_classes]
            'hidden_states': x.detach() if self.training else x
        }
        # Store output for callback access (always store, not just during evaluation)
        self.last_forward_output = output

        # Return classification output
        return output
    


class ClassificationHead(nn.Module):
    """
    Classification head supporting both binary and ternary classification.

    Binary mode: Single output for buy/sell decision
    Ternary mode: Three outputs for Hold/Buy/Sell decision

    TEMPORAL ORDERING CONVENTION:
    - x[:, 0, :] = Most recent timestep (current time, offset=0)
    - x[:, 1, :] = 1 timestep ago
    - x[:, -1, :] = Oldest timestep (maximum lookback)

    The model uses the most recent timestep as a query to compute attention weights
    over all historical timesteps, allowing it to adaptively focus on relevant
    temporal patterns for prediction.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, num_classes: int = 1,
                 classification_mode: str = 'binary'):
        super().__init__()

        # Attention mechanism: Use most recent timestep to query historical data
        # Full dimension attention for maximum expressiveness in final decision
        self.attention_query = nn.Linear(d_model, d_model)
        self.attention_key = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

        # Classification layer based on mode
        self.classification_mode = classification_mode
        self.num_classes = num_classes
        self.fc = nn.Linear(d_model, num_classes, bias=False)  # Direct projection to logits
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected 3D input tensor, got {x.dim()}D"
        batch_size, seq_len, d_model = x.shape
        
        # Temporal attention aggregation:
        # Use the most recent timestep (index 0) as a query to attend over all historical data.
        # This allows the model to dynamically weight different historical timesteps based on
        # their relevance to the current prediction.
        
        # Compute attention scores
        queries = self.attention_query(x[:, :1, :])  # Most recent timestep as query [batch_size, 1, d_model]
        keys = self.attention_key(x)  # All timesteps (0=most recent to -1=oldest) as keys [batch_size, seq_len, d_model]
        
        # Compute attention weights (scale by sqrt(d_model) for stability)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (d_model ** 0.5)  # [batch_size, 1, seq_len]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Store attention weights for debugging (detached to avoid memory issues)
        self._last_attention_weights = attention_weights.detach()
        
        # Apply attention to get weighted representation
        x_aggregated = torch.matmul(attention_weights, x).squeeze(1)  # [batch_size, d_model]
        
        
        # Apply dropout for regularization
        x = self.dropout(x_aggregated)

        # Projection to logits based on classification mode
        logits = self.fc(x)  # [batch_size, num_classes]
        assert logits.shape == (batch_size, self.num_classes), f"Logits shape {logits.shape} != expected ({batch_size}, {self.num_classes})"

        return logits



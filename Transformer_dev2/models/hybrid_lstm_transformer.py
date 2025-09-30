"""
Hybrid LSTM-Transformer Model for Order Book Prediction.

This model combines the sequential processing power of LSTM with the
long-range dependency modeling of Transformers. The LSTM acts as the
primary feature extractor, while the Transformer refines these features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from torch.utils.checkpoint import checkpoint
import math

# Import components from existing modules
from .components import (
    OrderBookEmbedding,
    TransformerEncoderLayer,
    PositionalEncoding,
    MultiScaleFeatureExtractor,
    RotaryEmbedding
)

# Import TFT components
from .tft_components import TemporalFusionDecoder

# Config imports removed - using dict-like access


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder for sequential processing.
    
    This module processes the input sequences with a bidirectional LSTM,
    capturing both forward and backward temporal dependencies.
    """
    
    def __init__(
        self,
        d_model: int,
        lstm_hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )
        
        # Project concatenated forward/backward hidden states back to d_model
        self.projection = nn.Linear(lstm_hidden_size * 2, d_model)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize LSTM weights
        self._init_lstm_weights()
    
    def _init_lstm_weights(self):
        """Initialize LSTM weights using Xavier uniform initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the BiLSTM encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional padding mask
            
        Returns:
            - Output tensor of shape [batch_size, seq_len, d_model]
            - Tuple of (hidden_state, cell_state) for last timestep
        """
        # Pack padded sequence if mask is provided
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Project back to d_model dimension
        output = self.projection(lstm_out)
        
        # Layer norm and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output, (h_n, c_n)


class HybridLSTMTransformer(nn.Module):
    """
    Hybrid LSTM-Transformer model for order book prediction.
    
    Architecture:
    1. Embedding layer
    2. Bidirectional LSTM (primary processor)
    3. Reduced Transformer layers (refinement)
    4. Optional Temporal Fusion Decoder
    5. Classification head for binary prediction
    """
    
    def __init__(self, config: Optional[Dict] = None, binary_classification_config=None):
        super().__init__()
        
        # Config is now expected to be passed in as a dict
        if config is None:
            raise ValueError("Config must be provided")
        self.config = config
        self.binary_classification_config = binary_classification_config
        self.gradient_checkpointing = False
        
        # Store key dimensions
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        
        # Hybrid architecture parameters
        # Handle None case for lstm_hidden_size
        lstm_hidden_from_config = config.lstm_hidden_size if hasattr(config, 'lstm_hidden_size') else None
        self.lstm_hidden_size = lstm_hidden_from_config if lstm_hidden_from_config is not None else config.d_model // 2
        
        self.lstm_num_layers = config.lstm_num_layers if hasattr(config, 'lstm_num_layers') else 2
        self.transformer_num_layers = config.transformer_num_layers if hasattr(config, 'transformer_num_layers') else 3
        
        # Input embedding layer
        if config.n_features is None:
            raise ValueError("n_features must be set in config")
        input_features = config.n_features
        self.embedding = OrderBookEmbedding(
            input_dim=input_features,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # Bidirectional LSTM Encoder (Primary Processor)
        self.bilstm_encoder = BiLSTMEncoder(
            d_model=config.d_model,
            lstm_hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            dropout=config.dropout
        )
        
        
        # Position encoding (applied after LSTM)
        # LSTM already encodes positional information, so this is optional
        self.use_positional_encoding = config.use_positional_encoding_after_lstm if hasattr(config, 'use_positional_encoding_after_lstm') else False
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=config.d_model,
                max_len=config.max_sequence_length if hasattr(config, 'max_sequence_length') else 1024,
                dropout=config.dropout
            )
        else:
            self.pos_encoder = None
        
        # Transformer encoder layers (Reduced count for refinement)
        self.use_grn = config.use_grn if hasattr(config, 'use_grn') else True
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.feedforward_dim,
                dropout=config.dropout,
                activation=config.activation,
                layer_norm_eps=config.layer_norm_eps,
                use_grn=self.use_grn
            )
            for _ in range(self.transformer_num_layers)
        ])
        
        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(
            d_model=config.d_model,
            kernel_sizes=[3, 5, 7]
        )
        
        # Temporal Fusion Decoder (if enabled)
        self.use_temporal_fusion = config.use_temporal_fusion if hasattr(config, 'use_temporal_fusion') else True
        if self.use_temporal_fusion:
            self.temporal_decoder = TemporalFusionDecoder(
                d_model=config.d_model,
                n_heads=max(config.n_heads // 2, 1),
                dropout=config.dropout,
                use_grn=self.use_grn
            )
        else:
            self.temporal_decoder = None
        
        # Classification head
        from .transformer import ClassificationHead
        if self.binary_classification_config:
            self.prediction_head = ClassificationHead(
                config.d_model,
                dropout=config.dropout
            )
        else:
            self.prediction_head = ClassificationHead(
                config.d_model,
                dropout=config.dropout
            )
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
            - 'logits': Binary classification logits [batch_size, 1]
            - 'hidden_states': Final hidden states
            - 'lstm_hidden': LSTM hidden states (for analysis)
        """
        batch_size, seq_len, n_features = x.shape
        
        # 1. Embedding
        x_embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # 2. BiLSTM Encoding (Primary Processing)
        lstm_out, (h_n, c_n) = self.bilstm_encoder(x_embedded, mask)
        
        # 3. Residual connection
        x = x_embedded + lstm_out
        
        # 4. Optional positional encoding
        if self.pos_encoder is not None:
            # Transpose for positional encoding (expects seq_len first)
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)
        
        # 5. Transformer refinement
        # Transpose for transformer (expects seq_len first)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        
        for i, layer in enumerate(self.transformer_layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(x, mask):
                        return module(x, src_mask=mask)
                    return custom_forward
                
                x, _ = checkpoint(create_custom_forward(layer), x, mask, use_reentrant=False)
            else:
                x, _ = layer(x, src_mask=mask)
        
        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 6. Multi-scale feature extraction
        if self.multi_scale is not None:
            multi_scale_features = self.multi_scale(x)
            x = x + multi_scale_features
        
        # 7. Temporal fusion decoder
        if self.temporal_decoder is not None:
            x, _ = self.temporal_decoder(x, mask=mask)
        
        # 8. Classification
        predictions = self.prediction_head(x)
        
        # Store outputs
        output = {
            'logits': predictions,  # [batch_size, 1]
            'hidden_states': x.detach() if self.training else x,
            'lstm_hidden': lstm_out.detach() if self.training else lstm_out
        }
        
        # Store for callback access
        self.last_forward_output = output
        
        return output
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
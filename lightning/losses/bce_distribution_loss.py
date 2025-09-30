"""BCE Loss with optional distribution matching for binary classification."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import logging

from .base_distribution_loss import BaseDistributionLoss
from .loss_registry import register_loss

logger = logging.getLogger(__name__)


@register_loss('bce')
class BCEDistributionLoss(BaseDistributionLoss):
    """
    Binary Cross Entropy loss with optional distribution matching for binary classification.
    
    This loss uses PyTorch's BCEWithLogitsLoss as the base loss function and adds
    optional batch distribution matching to ensure predictions match the actual 
    class distribution within each batch.
    
    The distribution matching supports both:
    - L1 (linear) loss: Simple ratio matching
    - F1-based loss: Considers both precision and recall for better performance
    """
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0, **kwargs):
        """
        Args:
            pos_weight: Weight for positive class in BCE loss (default None)
            label_smoothing: Label smoothing factor (0.0 = no smoothing, typical 0.1-0.2) (default 0.0)
            **kwargs: Additional arguments passed to BaseDistributionLoss
        """
        super().__init__(**kwargs)
        
        # Initialize BCE loss
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        
        # Store label smoothing parameter
        self.label_smoothing = label_smoothing
        
        # Log configuration
        logger.info(f"BCEDistributionLoss initialized with pos_weight={pos_weight}, label_smoothing={label_smoothing}")
    
    def compute_primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Compute the BCE loss with optional label smoothing.
        
        Args:
            pred: Raw logits [batch_size]
            target: Ground truth labels [batch_size]
            
        Returns:
            Tuple of (loss_tensor, 'bce_loss')
        """
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # Smooth the labels: target * (1 - epsilon) + (1 - target) * epsilon
            # This transforms: 0 -> epsilon, 1 -> 1-epsilon
            smoothed_target = target.float() * (1 - self.label_smoothing) + (1 - target.float()) * self.label_smoothing
        else:
            smoothed_target = target.float()
        
        # Compute BCE loss with smoothed targets
        bce_loss = self.bce_loss(pred, smoothed_target)
        
        return bce_loss, 'bce_loss'
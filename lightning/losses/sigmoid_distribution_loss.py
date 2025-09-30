"""Sigmoid Loss with optional distribution matching for binary classification."""

import torch
from typing import Tuple
import logging

from .base_distribution_loss import BaseDistributionLoss
from .loss_registry import register_loss

logger = logging.getLogger(__name__)


@register_loss('sigmoid')
class SigmoidDistributionLoss(BaseDistributionLoss):
    """
    Sigmoid-based loss with optional distribution matching for binary classification.
    
    The loss formula is: Loss = sigmoid(k × (0.5 - p) × (2y - 1))
    
    Where:
    - p is the predicted probability (sigmoid of logits)
    - y is the target label (0 or 1)
    - k is a scaling factor controlling the steepness of the loss
    
    This loss naturally penalizes predictions based on their distance from
    the decision boundary (0.5), with wrong predictions receiving higher loss.
    
    Additionally supports batch distribution matching to ensure predictions
    match the actual class distribution within each batch using either:
    - L1 (linear) loss: Simple ratio matching
    - F1-based loss: Considers both precision and recall for better performance
    """
    
    def __init__(self, k=5.0, **kwargs):
        """
        Args:
            k: Scaling factor for the sigmoid loss formula (default 5.0)
            **kwargs: Additional arguments passed to BaseDistributionLoss
        """
        super().__init__(**kwargs)
        self.k = k
        logger.info(f"SigmoidDistributionLoss initialized with k={k}")
    
    def compute_primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Compute the sigmoid-based loss.
        
        Args:
            pred: Raw logits [batch_size]
            target: Ground truth labels [batch_size]
            
        Returns:
            Tuple of (loss_tensor, 'sigmoid_loss')
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(pred)
        
        # Sigmoid Loss: Loss = sigmoid(k × (0.5 - p) × (2y - 1))
        # Convert target from {0,1} to {-1,1}
        target_signed = 2 * target.float() - 1  # (2y - 1)
        
        # Calculate distance from decision boundary
        distance_from_boundary = 0.5 - probs  # (0.5 - p)
        
        # Compute the input to sigmoid
        sigmoid_input = self.k * distance_from_boundary * target_signed
        
        # Apply sigmoid to get the loss
        sigmoid_loss = torch.sigmoid(sigmoid_input)
        
        # Average the sigmoid loss
        sigmoid_loss = sigmoid_loss.mean()
        
        return sigmoid_loss, 'sigmoid_loss'
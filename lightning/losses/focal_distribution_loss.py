"""Focal Loss with optional distribution matching for binary classification."""

import torch
from typing import Tuple
import logging

from .base_distribution_loss import BaseDistributionLoss
from .loss_registry import register_loss

logger = logging.getLogger(__name__)


@register_loss('focal')
class FocalDistributionLoss(BaseDistributionLoss):
    """
    Focal loss with optional distribution matching for binary classification.
    
    Focal loss is designed to address class imbalance by down-weighting easy examples
    and focusing on hard ones. The loss formula is:
    
    FL(pt) = -α(1-pt)^γ * log(pt)
    
    Where:
    - pt is the probability of the correct class
    - α is the class weight (alpha)
    - γ is the focusing parameter (gamma)
    
    Additionally supports batch distribution matching to ensure predictions
    match the actual class distribution within each batch using either:
    - L1 (linear) loss: Simple ratio matching
    - F1-based loss: Considers both precision and recall for better performance
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        """
        Args:
            alpha: Class weight for positive class (default 0.25 for imbalanced data)
            gamma: Focusing parameter - higher values focus more on hard examples (default 2.0)
            **kwargs: Additional arguments passed to BaseDistributionLoss
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        logger.info(f"FocalDistributionLoss initialized with alpha={alpha}, gamma={gamma}")
    
    def compute_primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Compute the focal loss.
        
        Args:
            pred: Raw logits [batch_size]
            target: Ground truth labels [batch_size]
            
        Returns:
            Tuple of (loss_tensor, 'focal_loss')
        """
        # More aggressive clamping for numerical stability with high gamma
        eps = 1e-4 if self.gamma > 2.0 else 1e-7
        
        # Convert logits to probabilities with clamping
        probs = torch.sigmoid(pred)
        probs = torch.clamp(probs, min=eps, max=1-eps)
        
        # Compute probability of correct class
        # pt = p if y=1, 1-p if y=0
        pt = torch.where(target == 1, probs, 1 - probs)
        
        # Additional clamping for pt to ensure numerical stability
        pt = torch.clamp(pt, min=eps)
        
        # Compute focal weight: (1-pt)^gamma
        # Use original gamma without clamping to allow proper gradient flow
        # Pre-weight clipping: ensure base is in safe range before power operation
        focal_base = torch.clamp(1 - pt, min=eps, max=1-eps)
        focal_weight = torch.pow(focal_base, self.gamma)
        
        # Compute cross entropy: -log(pt)
        # Use log1p for better numerical stability when pt is close to 1
        ce_loss = -torch.log(pt + eps)
        
        # Apply alpha weighting
        # alpha_t = alpha if y=1, 1-alpha if y=0
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss: -alpha_t * (1-pt)^gamma * log(pt)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Only prevent negative losses, don't cap upper range
        focal_loss = torch.clamp_min(focal_loss, 0.0)
        
        # Check for inf/nan and clamp if necessary
        if torch.isinf(focal_loss).any() or torch.isnan(focal_loss).any():
            logger.warning(f"Focal loss contains inf/nan. Stats: pt_min={pt.min():.6f}, pt_max={pt.max():.6f}, "
                         f"focal_weight_max={focal_weight.max():.6f}, ce_loss_max={ce_loss.max():.6f}")
            # Replace inf/nan with moderate value
            focal_loss = torch.where(torch.isfinite(focal_loss), focal_loss, torch.tensor(2.0, device=focal_loss.device))
        
        # Average the focal loss
        focal_loss = focal_loss.mean()
        
        return focal_loss, 'focal_loss'
    
    def _create_loss_dict(self, primary_loss: torch.Tensor, primary_loss_name: str,
                         distribution_loss: torch.Tensor, **extra_components) -> dict:
        """
        Override to add focal-specific parameters to the loss dictionary.
        
        Args:
            primary_loss: The focal loss value
            primary_loss_name: 'focal_loss'
            distribution_loss: The distribution matching loss
            **extra_components: Any additional components
            
        Returns:
            Dictionary with all loss components including focal parameters
        """
        # Get base dictionary from parent
        loss_dict = super()._create_loss_dict(primary_loss, primary_loss_name, 
                                              distribution_loss, **extra_components)
        
        # Add focal-specific parameters
        loss_dict.update({
            'focal_alpha': self.alpha,
            'focal_gamma': self.gamma
        })
        
        return loss_dict
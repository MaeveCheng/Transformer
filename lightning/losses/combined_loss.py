"""Combined loss function that uses multiple primary losses with distribution matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .loss_registry import register_loss
from .base_distribution_loss import BaseDistributionLoss

logger = logging.getLogger(__name__)


@register_loss('combined')
class CombinedLoss(nn.Module):
    """
    Combined loss that uses multiple primary losses (BCE, Focal, Sigmoid) 
    with optional distribution matching.
    
    Total loss = w_bce*BCE + w_focal*Focal + w_sigmoid*Sigmoid + w_distribution*Distribution
    
    This allows flexible combination of different loss functions with individual weights,
    enabling experimentation with different loss combinations without code changes.
    """
    
    def __init__(self,
                 # Primary loss weights
                 bce_weight: float = 1.0,
                 focal_weight: float = 0.0,
                 sigmoid_weight: float = 0.0,
                 # BCE parameters
                 bce_pos_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 # Focal parameters
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 # Sigmoid parameters
                 sigmoid_k: float = 5.0,
                 # Distribution parameters
                 match_batch_distribution: bool = True,
                 total_error_weight: float = 1.0,
                 balance_weight: float = 0.5,
                 variance_penalty_weight: float = 0.1,
                 target_prediction_ratio: float = 0.5,
                 balance_loss_type: str = 'squared',
                 balance_loss_k: float = 5.0,
                 balance_loss_gamma: float = 3.0,
                 use_hard_fp_fn: bool = False,
                 fp_fn_temperature: float = 0.1,
                 fp_fn_threshold: float = 0.5,
                 target_fp_rate: Optional[float] = None,
                 target_fn_rate: Optional[float] = None,
                 max_variance: float = 0.25,
                 **kwargs):
        """
        Args:
            bce_weight: Weight for BCE loss component
            focal_weight: Weight for Focal loss component
            sigmoid_weight: Weight for Sigmoid loss component
            bce_pos_weight: Positive class weight for BCE
            label_smoothing: Label smoothing factor for BCE
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            sigmoid_k: Scaling factor for sigmoid loss
            match_batch_distribution: Enable batch distribution matching
            ... (other distribution parameters)
        """
        super().__init__()
        
        # Store weights
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.sigmoid_weight = sigmoid_weight
        
        # Store label smoothing
        self.label_smoothing = label_smoothing
        
        # Initialize primary loss components
        # BCE Loss
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight, reduction='mean')
        
        # Focal Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Sigmoid Loss parameters
        self.sigmoid_k = sigmoid_k
        
        # Initialize distribution loss if needed
        if match_batch_distribution:
            # Import here to avoid circular dependency
            from .base_distribution_loss import BaseDistributionLoss
            
            # Create a minimal distribution loss handler
            self.distribution_handler = DistributionLossHandler(
                match_batch_distribution=match_batch_distribution,
                total_error_weight=total_error_weight,
                balance_weight=balance_weight,
                variance_penalty_weight=variance_penalty_weight,
                target_prediction_ratio=target_prediction_ratio,
                balance_loss_type=balance_loss_type,
                balance_loss_k=balance_loss_k,
                balance_loss_gamma=balance_loss_gamma,
                use_hard_fp_fn=use_hard_fp_fn,
                fp_fn_temperature=fp_fn_temperature,
                fp_fn_threshold=fp_fn_threshold,
                target_fp_rate=target_fp_rate,
                target_fn_rate=target_fn_rate,
                max_variance=max_variance
            )
        else:
            self.distribution_handler = None
        
        # Log configuration
        logger.info(f"CombinedLoss initialized with weights:")
        logger.info(f"  BCE weight: {bce_weight}")
        logger.info(f"  Focal weight: {focal_weight}")
        logger.info(f"  Sigmoid weight: {sigmoid_weight}")
        logger.info(f"  Distribution loss: {'enabled' if match_batch_distribution else 'disabled'}")
        logger.info(f"  Label smoothing: {label_smoothing}")
    
    def compute_bce_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss with optional label smoothing."""
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # Smooth the labels: target * (1 - epsilon) + (1 - target) * epsilon
            # This transforms: 0 -> epsilon, 1 -> 1-epsilon
            smoothed_target = target.float() * (1 - self.label_smoothing) + (1 - target.float()) * self.label_smoothing
        else:
            smoothed_target = target.float()
        
        return self.bce_loss(pred, smoothed_target)
    
    def compute_focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Convert logits to probabilities
        probs = torch.sigmoid(pred)
        
        # Compute focal loss
        # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        # Get p_t (probability of correct class)
        p_t = torch.where(target == 1, probs, 1 - probs)
        
        # Get alpha_t
        alpha_t = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        # Compute focal weight
        focal_weight = alpha_t * torch.pow(1 - p_t, self.focal_gamma)
        
        # Compute BCE part (using logits for numerical stability)
        bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()
    
    def compute_sigmoid_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid-based loss."""
        # Convert logits to probabilities
        probs = torch.sigmoid(pred)
        
        # Sigmoid Loss: Loss = sigmoid(k × (0.5 - p) × (2y - 1))
        # Convert target from {0,1} to {-1,1}
        target_signed = 2 * target.float() - 1  # (2y - 1)
        
        # Calculate distance from decision boundary
        distance_from_boundary = 0.5 - probs  # (0.5 - p)
        
        # Compute the input to sigmoid
        sigmoid_input = self.sigmoid_k * distance_from_boundary * target_signed
        
        # Apply sigmoid to get the loss
        sigmoid_loss = torch.sigmoid(sigmoid_input)
        
        return sigmoid_loss.mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing all loss components.
        
        Args:
            pred: Raw logits [batch_size] or [batch_size, 1]
            target: Ground truth labels [batch_size] or [batch_size, 1]
            
        Returns:
            Dictionary containing all loss components
        """
        # Prepare tensors (squeeze if needed)
        if pred.dim() > 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if target.dim() > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        
        # Initialize loss components
        loss_components = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Compute BCE loss if weight > 0
        if self.bce_weight > 0:
            bce_loss = self.compute_bce_loss(pred, target)
            loss_components['bce_loss'] = bce_loss
            total_loss = total_loss + self.bce_weight * bce_loss
        
        # Compute Focal loss if weight > 0
        if self.focal_weight > 0:
            focal_loss = self.compute_focal_loss(pred, target)
            loss_components['focal_loss'] = focal_loss
            total_loss = total_loss + self.focal_weight * focal_loss
        
        # Compute Sigmoid loss if weight > 0
        if self.sigmoid_weight > 0:
            sigmoid_loss = self.compute_sigmoid_loss(pred, target)
            loss_components['sigmoid_loss'] = sigmoid_loss
            total_loss = total_loss + self.sigmoid_weight * sigmoid_loss
        
        # Compute distribution loss if enabled
        if self.distribution_handler:
            probs = torch.sigmoid(pred)
            distribution_loss = self.distribution_handler.compute_distribution_loss(probs, target)
            loss_components['distribution_loss'] = distribution_loss
            total_loss = total_loss + distribution_loss
            
            # Add distribution components for logging
            if hasattr(self.distribution_handler, '_last_regularization_components'):
                components = self.distribution_handler._last_regularization_components
                if 'total_error_loss' in components:
                    loss_components['total_error_loss'] = components['total_error_loss']
                if 'balance_loss' in components:
                    loss_components['balance_loss'] = components['balance_loss']
                if 'variance_penalty' in components:
                    loss_components['variance_penalty'] = components['variance_penalty']
                # Add FP/FN rates if available
                if 'fp_rate' in components:
                    loss_components['fp_rate'] = components['fp_rate']
                if 'fn_rate' in components:
                    loss_components['fn_rate'] = components['fn_rate']
                # Add F1 components if available
                if 'soft_f1' in components:
                    loss_components['soft_f1'] = components['soft_f1']
                if 'soft_precision' in components:
                    loss_components['soft_precision'] = components['soft_precision']
                if 'soft_recall' in components:
                    loss_components['soft_recall'] = components['soft_recall']
                # Add confidence metrics if available
                if 'confidence' in components:
                    loss_components['confidence'] = components['confidence']
                if 'normalized_confidence' in components:
                    loss_components['normalized_confidence'] = components['normalized_confidence']
                # Add prediction rates if available
                if 'pred_rate' in components:
                    loss_components['pred_rate'] = components['pred_rate']
                if 'actual_positive_ratio' in components:
                    loss_components['actual_positive_ratio'] = components['actual_positive_ratio']
                # Add target rates if available
                if 'target_rate' in components:
                    loss_components['target_rate'] = components['target_rate']
                if 'target_fp_rate' in components:
                    loss_components['target_fp_rate'] = components['target_fp_rate']
                if 'target_fn_rate' in components:
                    loss_components['target_fn_rate'] = components['target_fn_rate']
                # Add soft/hard confusion matrix components if available
                for key in ['soft_tp', 'soft_fp', 'soft_fn', 'soft_tn',
                           'hard_tp', 'hard_fp', 'hard_fn', 'hard_tn',
                           'hard_f1', 'hard_precision', 'hard_recall',
                           'actual_hard_fp', 'actual_hard_fn']:
                    if key in components:
                        loss_components[key] = components[key]
                # Add regularization loss (same as distribution loss for backward compatibility)
                if 'regularization_loss' in components:
                    loss_components['regularization_loss'] = components['regularization_loss']
                    loss_components['fp_fn_balance_loss'] = components['regularization_loss']  # Backward compatibility
                # Add distribution matching loss (same as balance loss for backward compatibility)
                if 'balance_loss' in components:
                    loss_components['distribution_matching_loss'] = components['balance_loss']
        
        # Add total loss (use 'total' for backward compatibility with module.py)
        loss_components['total'] = total_loss
        loss_components['loss'] = total_loss  # Also keep 'loss' for compatibility
        
        # Calculate weighted primary loss (sum of all primary losses)
        weighted_primary = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if self.bce_weight > 0 and 'bce_loss' in loss_components:
            weighted_primary = weighted_primary + self.bce_weight * loss_components['bce_loss']
        if self.focal_weight > 0 and 'focal_loss' in loss_components:
            weighted_primary = weighted_primary + self.focal_weight * loss_components['focal_loss']
        if self.sigmoid_weight > 0 and 'sigmoid_loss' in loss_components:
            weighted_primary = weighted_primary + self.sigmoid_weight * loss_components['sigmoid_loss']
        
        # Add backward compatibility keys
        loss_components['weighted_primary_loss'] = weighted_primary
        loss_components['weighted_regularization_loss'] = loss_components.get('distribution_loss', 
                                                                              torch.tensor(0.0, device=pred.device))
        
        # Add individual weighted components for logging
        if self.bce_weight > 0 and 'bce_loss' in loss_components:
            loss_components['weighted_bce'] = self.bce_weight * loss_components['bce_loss']
        if self.focal_weight > 0 and 'focal_loss' in loss_components:
            loss_components['weighted_focal'] = self.focal_weight * loss_components['focal_loss']
        if self.sigmoid_weight > 0 and 'sigmoid_loss' in loss_components:
            loss_components['weighted_sigmoid'] = self.sigmoid_weight * loss_components['sigmoid_loss']
        
        return loss_components


class DistributionLossHandler(BaseDistributionLoss):
    """
    Handler for distribution loss computation.
    Inherits from BaseDistributionLoss to reuse the distribution matching logic.
    """
    
    def compute_primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Dummy implementation - not used in combined loss.
        Required by BaseDistributionLoss abstract class.
        """
        return torch.tensor(0.0, device=pred.device), 'dummy'
    
    def compute_distribution_loss(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute only the distribution/regularization loss.
        
        Args:
            probs: Predicted probabilities [batch_size]
            target: Ground truth labels [batch_size]
            
        Returns:
            Distribution loss value
        """
        # Use the base class's regularization computation
        regularization_loss = self._compute_regularization_losses(probs, target)
        return regularization_loss
"""Base class for losses with optional distribution matching for binary classification."""

import torch
import torch.nn as nn
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BaseDistributionLoss(nn.Module, ABC):
    """
    Abstract base class for losses with regularization components.
    
    Total loss = primary_loss_weight * primary_loss + regularization_loss
    
    Where regularization_loss = total_error_weight * total_error_loss + 
                               balance_weight * balance_loss + 
                               variance_penalty_weight * variance_penalty
    
    Regularization components:
    - Total error loss: Minimizes FP + FN rates
    - Balance loss: Enforces target prediction ratio (default 50/50)
    - Confidence penalty: Penalizes low confidence (all predictions near 0.5) while balance constraint ensures diversity
    
    Derived classes must implement:
    - compute_primary_loss(pred, target) -> (loss_tensor, loss_name)
    """
    
    def __init__(self,
                 match_batch_distribution=False,
                 primary_loss_weight=1.0,
                 # Regularization weights  
                 total_error_weight=1.0,
                 balance_weight=0.5,
                 variance_penalty_weight=0.1,
                 # Balance enforcement
                 target_prediction_ratio=0.5,
                 balance_loss_type='squared',
                 balance_loss_k=5.0,
                 balance_loss_gamma=3.0,
                 # Hard FP/FN parameters
                 use_hard_fp_fn=False,
                 fp_fn_temperature=0.1,
                 fp_fn_threshold=0.5,
                 # Target rate parameters
                 target_fp_rate=None,
                 target_fn_rate=None,
                 **kwargs):
        """
        Args:
            match_batch_distribution: Enable batch distribution matching and regularization losses (default False)
            primary_loss_weight: Weight for the primary loss (BCE, focal, sigmoid). Set very small to effectively disable (default 1.0)
            total_error_weight: Weight for minimizing total error rate (FP + FN) (default 1.0)
            balance_weight: Weight for balance enforcement loss (default 0.5)
            variance_penalty_weight: Weight for confidence penalty that discourages uncertain predictions near 0.5 (default 0.1)
            target_prediction_ratio: Target ratio for positive predictions (default 0.5 for balanced)
            balance_loss_type: Type of balance loss formula: 'exponential', 'focal', 'power', 'squared', 'absolute' (default 'squared')
            balance_loss_k: Scaling factor for exponential balance loss (default 5.0)
            balance_loss_gamma: Focusing parameter for focal balance loss (default 3.0)
            use_hard_fp_fn: Use hard FP/FN calculation with differentiable approximation (default False)
            fp_fn_temperature: Temperature for sigmoid approximation of hard decisions (default 0.1)
            fp_fn_threshold: Decision threshold for hard FP/FN (default 0.5)
            target_fp_rate: Target FP rate (None = use balanced formula, float = specific target) (default None)
            target_fn_rate: Target FN rate (None = use balanced formula, float = specific target) (default None)
        """
        super().__init__()
        
        # Enable regularization losses
        self.match_batch_distribution = match_batch_distribution
        
        # Regularization loss weights
        self.total_error_weight = total_error_weight
        self.balance_weight = balance_weight
        self.variance_penalty_weight = variance_penalty_weight
        
        # Balance enforcement
        self.target_prediction_ratio = target_prediction_ratio
        self.balance_loss_type = balance_loss_type
        self.balance_loss_k = balance_loss_k
        self.balance_loss_gamma = balance_loss_gamma
        
        # Hard FP/FN parameters
        self.use_hard_fp_fn = use_hard_fp_fn
        self.fp_fn_temperature = fp_fn_temperature
        self.fp_fn_threshold = fp_fn_threshold
        
        # Target rate parameters
        self.target_fp_rate = target_fp_rate
        self.target_fn_rate = target_fn_rate
        
        # Primary loss weight
        self.primary_loss_weight = primary_loss_weight
        
        
        # Log the weights being used (only once at initialization)
        if match_batch_distribution:
            logger.debug(f"BaseDistributionLoss initialized with regularization weights: "
                        f"total_error_weight={self.total_error_weight}, "
                        f"balance_weight={self.balance_weight}, "
                        f"variance_penalty_weight={self.variance_penalty_weight}, "
                        f"target_prediction_ratio={self.target_prediction_ratio}, "
                        f"balance_loss_type={self.balance_loss_type}, "
                        f"use_hard_fp_fn={self.use_hard_fp_fn}")
    
    @abstractmethod
    def compute_primary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """
        Compute the primary loss (e.g., BCE, focal, sigmoid).
        
        Args:
            pred: Raw logits [batch_size]
            target: Ground truth labels [batch_size]
            
        Returns:
            Tuple of (loss_tensor, loss_name)
            - loss_tensor: The computed loss value
            - loss_name: Name for the loss (e.g., 'bce_loss', 'focal_loss')
        """
        raise NotImplementedError("Derived classes must implement compute_primary_loss")
    
    def _prepare_tensors(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors to ensure correct shapes.
        
        Args:
            pred: Input predictions
            target: Input targets
            
        Returns:
            Tuple of (prepared_pred, prepared_target)
        """
        # Ensure pred is 1D (single logit per sample)
        if pred.dim() > 2:
            pred = pred.squeeze(-1)
        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
            
        # Ensure target has correct shape
        if target.dim() > 1:
            target = target.squeeze(-1)
            
        return pred, target
    
    def _compute_soft_f1_components(self, probs: torch.Tensor, targets: torch.Tensor, 
                                    epsilon: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute differentiable F1 score components using soft predictions.
        
        Args:
            probs: Predicted probabilities [batch_size]
            targets: Ground truth labels [batch_size]
            epsilon: Numerical stability constant
            
        Returns:
            Dictionary containing soft TP, FP, FN, TN, precision, recall, and F1
        """
        if epsilon is None:
            epsilon = 1e-7
            
        # Ensure tensors are float for computation
        probs = probs.float()
        targets = targets.float()
        
        # Clamp probabilities for numerical stability
        probs = torch.clamp(probs, min=epsilon, max=1-epsilon)
        
        # Compute soft confusion matrix components
        # These are differentiable w.r.t. probabilities
        soft_tp = (probs * targets).sum()
        soft_fp = (probs * (1 - targets)).sum()
        soft_fn = ((1 - probs) * targets).sum()
        soft_tn = ((1 - probs) * (1 - targets)).sum()
        
        # Compute soft precision and recall with numerical stability
        soft_precision = soft_tp / (soft_tp + soft_fp + epsilon)
        soft_recall = soft_tp / (soft_tp + soft_fn + epsilon)
        
        # Compute soft F1 score
        soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall + epsilon)
        
        return {
            'soft_tp': soft_tp,
            'soft_fp': soft_fp,
            'soft_fn': soft_fn,
            'soft_tn': soft_tn,
            'soft_precision': soft_precision,
            'soft_recall': soft_recall,
            'soft_f1': soft_f1
        }
    
    def _compute_hard_f1_components(self, probs: torch.Tensor, targets: torch.Tensor, 
                                   threshold: float = 0.5, temperature: float = 0.1,
                                   epsilon: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute hard FP/FN components with differentiable approximation using straight-through estimator.
        
        This method computes hard FP/FN for forward pass but uses temperature-scaled sigmoid
        for gradient computation to maintain differentiability.
        
        Args:
            probs: Predicted probabilities [batch_size]
            targets: Ground truth labels [batch_size]
            threshold: Decision threshold for hard predictions (default 0.5)
            temperature: Temperature for sigmoid approximation (default 0.1)
            epsilon: Numerical stability constant
            
        Returns:
            Dictionary containing hard TP, FP, FN, TN with gradients
        """
        if epsilon is None:
            epsilon = 1e-5
            
        # Ensure tensors are float for computation
        probs = probs.float()
        targets = targets.float()
        
        # Clamp probabilities for numerical stability
        probs = torch.clamp(probs, min=epsilon, max=1-epsilon)
        
        # Hard predictions for forward pass
        hard_preds = (probs > threshold).float()
        
        # Temperature-scaled sigmoid for gradient approximation
        # As temperature → 0, this approaches a step function
        sharp_probs = torch.sigmoid((probs - threshold) / temperature)
        
        # Straight-through estimator: use hard predictions for forward,
        # but gradient flows through sharp_probs
        preds_ste = hard_preds.detach() + (sharp_probs - sharp_probs.detach())
        
        # Compute confusion matrix components using STE predictions
        hard_tp = (preds_ste * targets).sum()
        hard_fp = (preds_ste * (1 - targets)).sum()
        hard_fn = ((1 - preds_ste) * targets).sum()
        hard_tn = ((1 - preds_ste) * (1 - targets)).sum()
        
        # Also compute actual hard values for monitoring
        actual_hard_tp = (hard_preds * targets).sum()
        actual_hard_fp = (hard_preds * (1 - targets)).sum()
        actual_hard_fn = ((1 - hard_preds) * targets).sum()
        actual_hard_tn = ((1 - hard_preds) * (1 - targets)).sum()
        
        # Compute precision and recall with numerical stability
        hard_precision = hard_tp / (hard_tp + hard_fp + epsilon)
        hard_recall = hard_tp / (hard_tp + hard_fn + epsilon)
        
        # Compute F1 score
        hard_f1 = 2 * hard_precision * hard_recall / (hard_precision + hard_recall + epsilon)
        
        return {
            'hard_tp': hard_tp,
            'hard_fp': hard_fp,
            'hard_fn': hard_fn,
            'hard_tn': hard_tn,
            'hard_precision': hard_precision,
            'hard_recall': hard_recall,
            'hard_f1': hard_f1,
            # Also include actual hard values for monitoring
            'actual_hard_tp': actual_hard_tp,
            'actual_hard_fp': actual_hard_fp,
            'actual_hard_fn': actual_hard_fn,
            'actual_hard_tn': actual_hard_tn
        }
    
    def _compute_regularization_loss(self, probs: torch.Tensor, targets: torch.Tensor, 
                                    actual_positive_ratio: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss with three components.
        
        Components:
        1. Total error minimization: Minimize FP + FN rates
        2. Balance loss: Enforce target prediction ratio (default 50/50)
        3. Confidence penalty: Penalize low confidence predictions (all near 0.5)
        
        Args:
            probs: Predicted probabilities
            targets: Ground truth labels
            actual_positive_ratio: The actual ratio of positive samples in the batch
            
        Returns:
            FP/FN balanced distribution loss value
        """
        # Choose between soft and hard FP/FN computation
        if self.use_hard_fp_fn:
            # Use hard FP/FN with differentiable approximation
            f1_components = self._compute_hard_f1_components(
                probs, targets, 
                threshold=self.fp_fn_threshold,
                temperature=self.fp_fn_temperature,
                epsilon=1e-5
            )
            # Use the STE versions for loss computation
            tp = f1_components['hard_tp']
            fp = f1_components['hard_fp']
            fn = f1_components['hard_fn']
            tn = f1_components['hard_tn']
            # Store actual hard values for logging
            actual_hard_fp = f1_components['actual_hard_fp']
            actual_hard_fn = f1_components['actual_hard_fn']
        else:
            # Use soft FP/FN (original behavior)
            f1_components = self._compute_soft_f1_components(probs, targets, epsilon=1e-5)
            tp = f1_components['soft_tp']
            fp = f1_components['soft_fp']
            fn = f1_components['soft_fn']
            tn = f1_components['soft_tn']
            actual_hard_fp = None
            actual_hard_fn = None
        
        # CRITICAL: Always normalize by LOCAL batch size for gradient computation
        # DDP will average the losses across GPUs automatically
        local_batch_size = float(probs.size(0))
        epsilon = 1e-5
        
        # Component 1: Expected errors per sample
        # These represent the average number of FP/FN per sample in the batch
        fp_rate = fp / (local_batch_size + epsilon)
        fn_rate = fn / (local_batch_size + epsilon)
        
        # Compute target rates based on the global distribution
        target_rates = self._compute_target_rates_from_global_distribution(actual_positive_ratio)
        target_fp_rate = target_rates['target_fp_rate']
        target_fn_rate = target_rates['target_fn_rate']
        
        # Loss components:
        # 1. Distribution matching - ensure model predicts correct proportion of positives
        # Calculate prediction rate based on whether using hard or soft predictions
        if self.use_hard_fp_fn:
            # For hard predictions, use the STE predictions
            # pred_rate = average of predictions (0 or 1)
            pred_rate = f1_components['hard_tp'] + f1_components['hard_fp']  # Total positive predictions
            pred_rate = pred_rate / (local_batch_size + epsilon)
        else:
            # For soft predictions, pred_rate is just mean probability
            pred_rate = probs.mean()
        
        # Balance enforcement loss - encourage predictions to match target ratio
        # Compute deviation
        deviation = torch.abs(pred_rate - self.target_prediction_ratio)
        
        # Apply different balance loss formulas based on configuration
        if self.balance_loss_type == 'exponential':
            # Exponential penalty: grows exponentially with deviation
            # Pre-weight clipping: clamp deviation before applying weight
            deviation_clamped = torch.clamp(deviation, min=0.0, max=2.0)  # Max deviation of 2.0
            # Clamp the exponential argument to prevent explosion
            # Maximum reasonable argument is 5 (exp(5) ≈ 148)
            exp_arg = torch.clamp(self.balance_loss_k * deviation_clamped, min=0.0, max=5.0)
            # Compute exponential with stable formula
            balance_loss = torch.expm1(exp_arg)  # expm1(x) = exp(x) - 1, more numerically stable
            # Additional safety: clamp the final loss value before weights are applied
            balance_loss = torch.clamp_min(balance_loss, 0.0)
        elif self.balance_loss_type == 'focal':
            # Focal-like balance loss: focuses on large deviations
            # Pre-weight clipping: clamp deviation to reasonable range
            deviation_clamped = torch.clamp(deviation, min=0.0, max=1.0)
            # Clamp gamma * deviation to prevent numerical issues in exp
            gamma_dev = torch.clamp(self.balance_loss_gamma * deviation_clamped, min=0.0, max=10.0)
            # Use stable formula: 1 - exp(-x) can lose precision for small x
            balance_loss = torch.expm1(-gamma_dev).abs() * deviation_clamped  # abs() ensures positive
            # Final safety clamp before weights are applied
            balance_loss = torch.clamp_min(balance_loss, 0.0)
        elif self.balance_loss_type == 'power':
            # Adaptive power function: exponent increases with deviation
            # Pre-weight clipping: clamp deviation to reasonable range
            deviation_clamped = torch.clamp(deviation, min=0.0, max=1.0)  # Max deviation of 1.0
            # Clamp exponent to reasonable range to prevent numerical issues
            exponent = torch.clamp(2 + 2 * deviation_clamped, min=2.0, max=4.0)  # Reduced multiplier from 4 to 2
            # Always use stable computation to avoid conditional branching
            # Add epsilon to avoid log(0)
            safe_deviation = torch.clamp(deviation_clamped, min=1e-8)
            # Compute in log space for numerical stability
            log_loss = exponent * torch.log(safe_deviation)
            # Clamp in log space to prevent explosion (exp(5) ≈ 148)
            balance_loss = torch.exp(torch.clamp(log_loss, min=-10.0, max=5.0))
            # Final safety clamp before weights are applied
            balance_loss = torch.clamp_min(balance_loss, 0.0)
        elif self.balance_loss_type == 'squared':
            # Standard squared loss
            balance_loss = deviation ** 2
        elif self.balance_loss_type == 'absolute':
            # Linear absolute loss
            balance_loss = deviation
        else:
            raise ValueError(f"Unknown balance_loss_type: {self.balance_loss_type}")
        
        # Confidence penalty implementation
        # When balance_weight enforces 50/50 predictions, the model already makes diverse decisions
        # We should REWARD confident predictions (far from 0.5), not penalize them
        # This encourages the model to be as confident as possible while meeting the balance constraint
        
        # Calculate average confidence: how far predictions are from 0.5
        # confidence = 0 when all predictions are 0.5 (maximally uncertain)
        # confidence = 0.5 when all predictions are 0 or 1 (maximally confident)
        confidence = torch.abs(probs - 0.5).mean()
        
        # Normalize to [0, 1] range where 1 is maximum confidence
        max_confidence = 0.5  # Maximum possible distance from 0.5
        normalized_confidence = confidence / max_confidence
        
        # Convert to a "penalty" for compatibility with existing code structure
        # Since we want to reward confidence, we create a penalty that decreases with confidence
        # When confidence is high (good), penalty is low
        # When confidence is low (bad), penalty is high
        min_confidence_ratio = 0.3  # We want at least 30% confidence (predictions not all at 0.5)
        
        # This penalty is high when confidence is low, zero when confidence is sufficient
        variance_penalty = torch.relu(min_confidence_ratio - normalized_confidence) / min_confidence_ratio
        
        
        # 2. Minimize total error
        total_error = fp_rate + fn_rate
        target_total_error = target_fp_rate + target_fn_rate
        # Use absolute difference for total error
        total_error_loss = torch.abs(total_error - target_total_error)
        
        # Combine all regularization components with their weights
        # regularization_loss = total_error_weight * total_error_loss + balance_weight * balance_loss + variance_penalty_weight * variance_penalty
        regularization_loss = (self.total_error_weight * total_error_loss + 
                              self.balance_weight * balance_loss + 
                              self.variance_penalty_weight * variance_penalty)
        
        # Final safety check: clamp total regularization loss to prevent training instability
        regularization_loss = torch.clamp_min(regularization_loss, 0.0)
        
        # Check for numerical issues and log warnings
        if torch.isnan(regularization_loss) or torch.isinf(regularization_loss):
            logger.warning(f"Regularization loss contains NaN/Inf! Components: "
                         f"total_error={total_error_loss.item():.4f}, "
                         f"balance={balance_loss.item():.4f}, "
                         f"variance={variance_penalty.item():.4f}")
            # Return a safe default value
            regularization_loss = torch.tensor(10.0, device=regularization_loss.device, dtype=regularization_loss.dtype)
        
        # Debug logging with comprehensive statistics
        mode_str = "Hard" if self.use_hard_fp_fn else "Soft"
        logger.debug(f"=== Regularization Loss Debug ({mode_str} mode) ===")
        logger.debug(f"Batch size: {local_batch_size}")
        logger.debug(f"{mode_str} confusion matrix components:")
        logger.debug(f"  tp: {tp.item():.4f}, fp: {fp.item():.4f}")
        logger.debug(f"  fn: {fn.item():.4f}, tn: {tn.item():.4f}")
        logger.debug(f"  Sum: {(tp + fp + fn + tn).item():.4f} (should ≈ {local_batch_size})")
        
        # Log actual hard values if using hard FP/FN
        if self.use_hard_fp_fn and actual_hard_fp is not None:
            logger.debug(f"Actual hard FP: {actual_hard_fp.item():.0f}, Actual hard FN: {actual_hard_fn.item():.0f}")
        logger.debug(f"Expected values per sample:")
        logger.debug(f"  fp_expected: {fp_rate.item():.4f} = {fp.item():.4f} / {local_batch_size}")
        logger.debug(f"  fn_expected: {fn_rate.item():.4f} = {fn.item():.4f} / {local_batch_size}")
        logger.debug(f"  actual_positive_ratio: {actual_positive_ratio.item():.4f}")
        logger.debug(f"Target expected values (Bayes optimal):")
        logger.debug(f"  target_fp_expected: {target_fp_rate.item():.4f}")
        logger.debug(f"  target_fn_expected: {target_fn_rate.item():.4f}")
        logger.debug(f"Distribution matching:")
        logger.debug(f"  pred_rate: {pred_rate.item():.4f} (model's positive prediction rate)")
        logger.debug(f"  target_ratio: {self.target_prediction_ratio:.4f} (target for balanced predictions)")
        logger.debug(f"  Probability stats - min: {probs.min():.6f}, max: {probs.max():.6f}, mean: {probs.mean():.6f}, std: {probs.std():.6f}")
        logger.debug(f"Regularization components:")
        logger.debug(f"  Balance loss ({self.balance_loss_type}): {balance_loss.item():.6f} (weight: {self.balance_weight}, deviation: {deviation.item():.4f})")
        logger.debug(f"  Average confidence: {confidence.item():.6f} (distance from 0.5)")
        logger.debug(f"  Normalized confidence: {normalized_confidence.item():.4f} (0=all at 0.5, 1=all at 0/1)")
        logger.debug(f"  Confidence penalty: {variance_penalty.item():.4f} (weight: {self.variance_penalty_weight})")
        logger.debug(f"Total error minimization:")
        logger.debug(f"  total_error: {total_error.item():.4f} = {fp_rate.item():.4f} + {fn_rate.item():.4f}")
        logger.debug(f"  target_total_error: {target_total_error.item():.4f}")
        logger.debug(f"  total_error_loss: {total_error_loss.item():.4f} = |{total_error.item():.4f} - {target_total_error.item():.4f}| (weight: {self.total_error_weight})")
        logger.debug(f"Final regularization loss: {self.total_error_weight} * {total_error_loss.item():.4f} + "
                    f"{self.balance_weight} * {balance_loss.item():.4f} + "
                    f"{self.variance_penalty_weight} * {variance_penalty.item():.4f} = "
                    f"{regularization_loss.item():.4f}")
        
        # Store components for logging
        self._last_regularization_components = {
            'regularization_loss': regularization_loss,
            'total_error_loss': total_error_loss,
            'balance_loss': balance_loss,
            'variance_penalty': variance_penalty,
            'confidence': confidence,
            'normalized_confidence': normalized_confidence,
            'fp': fp,
            'fn': fn,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'pred_rate': pred_rate,  # Model's positive prediction rate
            'target_ratio': self.target_prediction_ratio,  # Target for balanced predictions
            'target_fp_rate': target_fp_rate,
            'target_fn_rate': target_fn_rate,
            'actual_positive_ratio': actual_positive_ratio,
            'use_hard_fp_fn': self.use_hard_fp_fn
        }
        
        # Store actual hard values if available
        if self.use_hard_fp_fn and actual_hard_fp is not None:
            self._last_regularization_components['actual_hard_fp'] = actual_hard_fp
            self._last_regularization_components['actual_hard_fn'] = actual_hard_fn
        
        # Also store F1 components for monitoring
        self._last_f1_components = f1_components
        
        return regularization_loss
    
    def _compute_target_rates_from_global_distribution(self, actual_positive_ratio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute target FP/FN expected values based on configuration or global class distribution.
        
        If target rates are configured, use those values directly.
        Otherwise, compute balanced rates based on the actual distribution:
        - Target FP expected value = (1 - actual_positive_ratio) * actual_positive_ratio
        - Target FN expected value = actual_positive_ratio * (1 - actual_positive_ratio)
        
        Args:
            actual_positive_ratio: The actual positive class ratio (can be global)
            
        Returns:
            Dictionary with target expected values per sample
        """
        # Check if target rates are explicitly configured
        if self.target_fp_rate is not None or self.target_fn_rate is not None:
            # Use configured values (convert to tensor on same device as actual_positive_ratio)
            device = actual_positive_ratio.device
            dtype = actual_positive_ratio.dtype
            
            target_fp_rate = torch.tensor(
                self.target_fp_rate if self.target_fp_rate is not None else 0.0,
                device=device, dtype=dtype
            )
            target_fn_rate = torch.tensor(
                self.target_fn_rate if self.target_fn_rate is not None else 0.0,
                device=device, dtype=dtype
            )
        else:
            # Use balanced formula based on actual distribution
            # For a model that matches the distribution:
            # - FP: negative samples predicted as positive
            # - FN: positive samples predicted as negative
            
            # If we predict randomly with the true positive ratio:
            # E[FP per sample] = P(negative) * P(predict positive) = (1 - p) * p
            # E[FN per sample] = P(positive) * P(predict negative) = p * (1 - p)
            
            target_fp_rate = (1 - actual_positive_ratio) * actual_positive_ratio
            target_fn_rate = actual_positive_ratio * (1 - actual_positive_ratio)
        
        return {
            'target_fp_rate': target_fp_rate,
            'target_fn_rate': target_fn_rate,
            'target_error_rate': target_fp_rate + target_fn_rate  # Total error rate
        }
    
    def _compute_regularization_losses(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization losses using local batch statistics.
        
        Args:
            probs: Predicted probabilities
            targets: Ground truth labels
            
        Returns:
            Total regularization loss
        """
        if not self.match_batch_distribution:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
            
        # Always use local batch statistics
        actual_positive_ratio = targets.float().mean()
        
        # Compute all regularization components
        regularization_loss = self._compute_regularization_loss(
            probs, targets, actual_positive_ratio
        )
        
        return regularization_loss
    
    def _create_loss_dict(self, primary_loss: torch.Tensor, primary_loss_name: str,
                         regularization_loss: torch.Tensor, **extra_components) -> Dict[str, torch.Tensor]:
        """
        Create standardized loss dictionary.
        
        Args:
            primary_loss: The main loss value (BCE, focal, etc.)
            primary_loss_name: Name for the primary loss
            regularization_loss: The total regularization loss
            **extra_components: Any additional components to include
            
        Returns:
            Dictionary with all loss components
        """
        # Calculate total loss: primary + regularization
        weighted_primary = self.primary_loss_weight * primary_loss
        
        # Clamp primary loss to prevent extreme values
        # Note: This is AFTER weight is applied, so we need to consider primary_loss_weight
        # Only handle inf/nan, don't clamp normal values to allow gradient flow
        if torch.isinf(weighted_primary) or torch.isnan(weighted_primary):
            logger.warning(f"Primary loss contains inf/nan: replacing with safe value")
            weighted_primary = torch.tensor(10.0, device=weighted_primary.device, dtype=weighted_primary.dtype)
        
        total_loss = weighted_primary
        if self.match_batch_distribution:
            total_loss = total_loss + regularization_loss
        
        # Only handle inf/nan for total loss, don't clamp normal values
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            logger.warning(f"Total loss contains inf/nan. Primary: {weighted_primary.item() if torch.isfinite(weighted_primary) else 'inf/nan'}, "
                         f"Regularization: {regularization_loss.item() if torch.isfinite(regularization_loss) else 'inf/nan'}. Replacing with safe value")
            total_loss = torch.tensor(20.0, device=total_loss.device, dtype=total_loss.dtype)
        
        # For backward compatibility in logging
        weighted_regularization = (regularization_loss 
                                 if self.match_batch_distribution 
                                 else torch.tensor(0.0, device=primary_loss.device))
        
        # Prepare return dictionary with all components
        loss_dict = {
            'total': total_loss,
            primary_loss_name: primary_loss,  # Raw unweighted value for logging
            'weighted_primary_loss': weighted_primary,  # Weighted contribution
            'primary_loss_weight': self.primary_loss_weight,
            'regularization_loss': regularization_loss,  # Total regularization loss
            'weighted_regularization_loss': weighted_regularization,  # For backward compatibility
        }
        
        # Add any extra components
        loss_dict.update(extra_components)
        
        # Add regularization components
        if self.match_batch_distribution and hasattr(self, '_last_regularization_components'):
            reg_components = self._last_regularization_components
            loss_dict.update({
                # Main regularization components
                'total_error_loss': reg_components['total_error_loss'],
                'balance_loss': reg_components['balance_loss'],
                'variance_penalty': reg_components['variance_penalty'],
                'confidence': reg_components['confidence'],
                'normalized_confidence': reg_components['normalized_confidence'],
                # FP/FN rates
                'fp_rate': reg_components['fp_rate'],
                'fn_rate': reg_components['fn_rate'],
                # Distribution info
                'pred_rate': reg_components['pred_rate'],
                'target_ratio': reg_components['target_ratio'],
                'target_fp_rate': reg_components['target_fp_rate'],
                'target_fn_rate': reg_components['target_fn_rate'],
                'actual_positive_ratio': reg_components['actual_positive_ratio'],
                # Legacy names for backward compatibility
                'fp_fn_balance_loss': reg_components['regularization_loss'],
                'distribution_matching_loss': reg_components['balance_loss'],
                'distribution_loss': regularization_loss
            })
            # Also add basic confusion matrix components for monitoring
            if hasattr(self, '_last_f1_components'):
                f1_components = self._last_f1_components
                if self.use_hard_fp_fn:
                    # Using hard FP/FN
                    loss_dict.update({
                        'hard_tp': f1_components['hard_tp'],
                        'hard_fp': f1_components['hard_fp'],
                        'hard_fn': f1_components['hard_fn'],
                        'hard_tn': f1_components['hard_tn'],
                        'hard_precision': f1_components['hard_precision'],
                        'hard_recall': f1_components['hard_recall'],
                        'hard_f1': f1_components['hard_f1']
                    })
                    # Also add actual hard values if available
                    if 'actual_hard_fp' in f1_components:
                        loss_dict.update({
                            'actual_hard_fp': f1_components['actual_hard_fp'],
                            'actual_hard_fn': f1_components['actual_hard_fn']
                        })
                else:
                    # Using soft FP/FN (original behavior)
                    loss_dict.update({
                        'soft_tp': f1_components['soft_tp'],
                        'soft_fp': f1_components['soft_fp'],
                        'soft_fn': f1_components['soft_fn'],
                        'soft_tn': f1_components['soft_tn'],
                        'soft_precision': f1_components['soft_precision'],
                        'soft_recall': f1_components['soft_recall'],
                        'soft_f1': f1_components['soft_f1']
                    })
                
            # Log summary of loss calculation for debugging  
            logger.debug(f"=== Loss Summary ===")
            logger.debug(f"Primary loss ({primary_loss_name}): {primary_loss.item():.4f} (weight: {self.primary_loss_weight})")
            logger.debug(f"Regularization loss: {regularization_loss.item():.4f}")
            logger.debug(f"Total loss: {total_loss.item():.4f}")
        
        return loss_dict
    
    def compute_component_gradients(self, loss_dict: Dict[str, torch.Tensor], 
                                   model_parameters: torch.nn.ParameterList) -> Dict[str, float]:
        """
        Compute gradient norms for individual loss components.
        
        This method computes how each loss component contributes to the gradients
        of the model parameters, providing insight into which components drive learning.
        
        Args:
            loss_dict: Dictionary containing loss components
            model_parameters: Model parameters to compute gradients for
            
        Returns:
            Dictionary mapping component names to their gradient L2 norms
        """
        component_gradients = {}
        
        # Components to track gradients for
        gradient_components = [
            ('total', 'Total Loss'),
            ('weighted_primary_loss', 'Primary Loss (Weighted)'),
            ('regularization_loss', 'Regularization Loss'),
            ('total_error_loss', 'Total Error Loss'),
            ('balance_loss', 'Balance Loss'),
            ('variance_penalty', 'Confidence Penalty')
        ]
        
        # Convert parameters iterator to list to ensure we can iterate multiple times
        param_list = list(model_parameters)
        
        # Filter to only components that exist in loss_dict
        available_components = [(key, name) for key, name in gradient_components 
                               if key in loss_dict and torch.is_tensor(loss_dict[key]) and loss_dict[key].requires_grad]
        
        if not available_components:
            logger.debug("No available components with requires_grad=True")
            # Debug: show what's in loss_dict
            for key in loss_dict:
                if torch.is_tensor(loss_dict[key]):
                    logger.debug(f"  {key}: is_tensor=True, requires_grad={loss_dict[key].requires_grad}")
                else:
                    logger.debug(f"  {key}: is_tensor=False")
            return component_gradients
        
        # Store original gradients
        original_grads = []
        for param in param_list:
            if param.requires_grad:
                if param.grad is not None:
                    original_grads.append(param.grad.clone())
                else:
                    original_grads.append(None)
        
        # Compute gradient norm for each component
        for component_key, component_name in available_components:
            # Zero gradients
            for param in param_list:
                if param.requires_grad and param.grad is not None:
                    param.grad.zero_()
            
            # Compute gradients for this component only
            component_loss = loss_dict[component_key]
            if component_loss.requires_grad:
                try:
                    # Retain graph since we'll compute gradients multiple times
                    torch.autograd.backward(component_loss, retain_graph=True)
                    
                    # Compute gradient norm
                    total_norm = 0.0
                    param_count = 0
                    for param in param_list:
                        if param.requires_grad and param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                            param_count += 1
                    
                    if param_count > 0:
                        total_norm = total_norm ** 0.5
                        component_gradients[component_key] = total_norm
                        
                except Exception as e:
                    logger.debug(f"Failed to compute gradients for {component_key}: {e}")
        
        # Restore original gradients
        grad_idx = 0
        for param in param_list:
            if param.requires_grad:
                if grad_idx < len(original_grads) and original_grads[grad_idx] is not None:
                    param.grad = original_grads[grad_idx]
                elif param.grad is not None:
                    param.grad.zero_()
                grad_idx += 1
        
        return component_gradients
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing primary loss and optional distribution matching.
        
        Args:
            pred: Raw logits [batch_size] or [batch_size, 1]
            target: Ground truth labels [batch_size] or [batch_size, 1]
            
        Returns:
            Dictionary containing all loss components
        """
        # Debug tensor shapes
        logger.debug(f"BaseDistributionLoss forward - pred shape: {pred.shape}, target shape: {target.shape}")
        
        # Prepare tensors
        pred, target = self._prepare_tensors(pred, target)
        
        # Debug after preparation
        logger.debug(f"After preparation - pred shape: {pred.shape}, target shape: {target.shape}")
        
        # Compute primary loss (implemented by derived class)
        primary_loss, primary_loss_name = self.compute_primary_loss(pred, target)
        
        # Convert logits to probabilities for regularization
        probs = torch.sigmoid(pred)
        
        # Compute regularization losses
        regularization_loss = self._compute_regularization_losses(probs, target)
        
        # Create and return loss dictionary
        return self._create_loss_dict(primary_loss, primary_loss_name, regularization_loss)
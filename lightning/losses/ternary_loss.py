"""Ternary classification loss for 3-class trading predictions (Hold/Buy/Sell)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TernaryClassificationLoss(nn.Module):
    """
    Loss function for ternary classification (Hold=0, Buy=1, Sell=2).

    Supports multiple loss components:
    - Cross-entropy loss for classification
    - Class balancing with configurable weights
    - Confidence regularization
    - Trading-specific penalties
    """

    def __init__(self,
                 # Core loss parameters
                 loss_type: str = 'cross_entropy',  # 'cross_entropy' or 'focal'
                 class_weights: Optional[list] = None,  # [hold_weight, buy_weight, sell_weight]

                 # Focal loss parameters
                 focal_alpha: Optional[list] = None,  # Per-class alpha for focal loss
                 focal_gamma: float = 2.0,  # Focusing parameter for focal loss

                 # Regularization
                 label_smoothing: float = 0.0,  # Label smoothing factor
                 confidence_penalty_weight: float = 0.0,  # Penalize low confidence predictions
                 symmetric_penalty_weight: float = 0.0,  # Penalize buy/sell imbalance

                 # Distribution matching
                 target_distribution: Optional[list] = None,  # [hold_ratio, buy_ratio, sell_ratio]
                 distribution_weight: float = 0.0,  # Weight for distribution matching loss

                 **kwargs):
        """
        Initialize ternary classification loss.

        Args:
            loss_type: Type of base loss ('cross_entropy' or 'focal')
            class_weights: Weights for each class [hold, buy, sell]
            focal_alpha: Per-class weights for focal loss
            focal_gamma: Focusing parameter for focal loss
            label_smoothing: Label smoothing factor (0-1)
            confidence_penalty_weight: Weight for confidence penalty
            symmetric_penalty_weight: Weight for buy/sell symmetry enforcement
            target_distribution: Target class distribution [hold, buy, sell] (sums to 1)
            distribution_weight: Weight for distribution matching loss
        """
        super().__init__()

        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        # Class weights
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

        # Focal loss alpha
        if focal_alpha is not None:
            self.register_buffer('focal_alpha', torch.tensor(focal_alpha, dtype=torch.float32))
        else:
            self.focal_alpha = None

        # Regularization weights
        self.confidence_penalty_weight = confidence_penalty_weight
        self.symmetric_penalty_weight = symmetric_penalty_weight

        # Distribution matching
        if target_distribution is not None:
            assert abs(sum(target_distribution) - 1.0) < 1e-6, "Target distribution must sum to 1"
            self.register_buffer('target_distribution', torch.tensor(target_distribution, dtype=torch.float32))
        else:
            self.target_distribution = None
        self.distribution_weight = distribution_weight

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for addressing class imbalance.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Apply focal term
        focal_term = (1 - p_t) ** self.focal_gamma

        # Apply per-class alpha if provided
        if self.focal_alpha is not None:
            focal_alpha = self.focal_alpha.to(logits.device)
            alpha_t = focal_alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        return focal_loss.mean()

    def compute_confidence_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalize low confidence predictions (high entropy).
        Encourages the model to make decisive predictions.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(3.0, device=logits.device))  # Maximum entropy for 3 classes
        normalized_entropy = entropy / max_entropy
        return normalized_entropy.mean()

    def compute_symmetric_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Penalize imbalance between buy and sell predictions.
        Helps maintain market neutrality.
        """
        probs = F.softmax(logits, dim=-1)
        buy_prob = probs[:, 1].mean()  # Average buy probability
        sell_prob = probs[:, 2].mean()  # Average sell probability
        imbalance = torch.abs(buy_prob - sell_prob)
        return imbalance

    def compute_distribution_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Match predicted distribution to target distribution.
        """
        if self.target_distribution is None:
            return torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=-1)
        predicted_dist = probs.mean(dim=0)  # Average class probabilities

        # KL divergence between predicted and target distributions
        target_dist = self.target_distribution.to(logits.device)
        kl_div = F.kl_div(
            (predicted_dist + 1e-8).log(),
            target_dist,
            reduction='sum'
        )
        return kl_div

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute ternary classification loss.

        Args:
            logits: Model predictions of shape [batch_size, 3]
            targets: Ground truth labels of shape [batch_size] with values in {0, 1, 2}
            return_components: If True, return individual loss components

        Returns:
            Dictionary containing total loss and optionally individual components
        """
        batch_size = logits.shape[0]
        assert logits.shape == (batch_size, 3), f"Expected logits shape (batch_size, 3), got {logits.shape}"
        assert targets.shape == (batch_size,), f"Expected targets shape (batch_size,), got {targets.shape}"
        # Ensure targets are long type for cross_entropy
        targets = targets.long()
        assert torch.all((targets >= 0) & (targets <= 2)), "Targets must be in {0, 1, 2}"

        # Compute primary loss
        if self.loss_type == 'focal':
            primary_loss = self.focal_loss(logits, targets)
        else:  # cross_entropy
            # Ensure class_weights are on the same device as logits
            class_weights = self.class_weights.to(logits.device) if self.class_weights is not None else None

            if self.label_smoothing > 0:
                # Apply label smoothing
                primary_loss = F.cross_entropy(
                    logits, targets,
                    weight=class_weights,
                    label_smoothing=self.label_smoothing
                )
            else:
                primary_loss = F.cross_entropy(
                    logits, targets,
                    weight=class_weights
                )

        # Initialize total loss
        total_loss = primary_loss

        # Compute regularization losses
        components = {'primary': primary_loss}

        if self.confidence_penalty_weight > 0:
            confidence_penalty = self.compute_confidence_penalty(logits)
            total_loss = total_loss + self.confidence_penalty_weight * confidence_penalty
            components['confidence_penalty'] = confidence_penalty

        if self.symmetric_penalty_weight > 0:
            symmetric_penalty = self.compute_symmetric_penalty(logits)
            total_loss = total_loss + self.symmetric_penalty_weight * symmetric_penalty
            components['symmetric_penalty'] = symmetric_penalty

        if self.distribution_weight > 0:
            distribution_loss = self.compute_distribution_loss(logits)
            total_loss = total_loss + self.distribution_weight * distribution_loss
            components['distribution_loss'] = distribution_loss

        # Prepare output
        output = {'loss': total_loss}
        if return_components:
            output['components'] = components

        return output


class TernaryTradingLoss(TernaryClassificationLoss):
    """
    Extended ternary loss with trading-specific thresholds.
    """

    def __init__(self,
                 # Threshold-based parameters
                 buy_threshold: float = 0.6,  # Confidence threshold for buy
                 sell_threshold: float = 0.6,  # Confidence threshold for sell

                 # Parent class parameters
                 **kwargs):
        """
        Initialize trading-specific ternary loss.

        Args:
            buy_threshold: Minimum confidence for buy action
            sell_threshold: Minimum confidence for sell action
            **kwargs: Arguments for parent TernaryClassificationLoss
        """
        super().__init__(**kwargs)

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def compute_trading_metrics(self,
                                logits: torch.Tensor,
                                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute trading-specific metrics.

        Args:
            logits: Model predictions [batch_size, 3]
            targets: Ground truth labels [batch_size]

        Returns:
            Dictionary of trading metrics
        """
        probs = F.softmax(logits, dim=-1)

        # Convert to trading decisions based on thresholds
        hold_conf = probs[:, 0]
        buy_conf = probs[:, 1]
        sell_conf = probs[:, 2]

        # Apply thresholds - use argmax to ensure mutual exclusivity
        predicted_class = torch.argmax(probs, dim=-1)

        # Only trigger buy/sell if confidence exceeds threshold AND it's the highest probability
        buy_decisions = ((predicted_class == 1) & (buy_conf > self.buy_threshold)).float()
        sell_decisions = ((predicted_class == 2) & (sell_conf > self.sell_threshold)).float()
        hold_decisions = ((predicted_class == 0) |
                         ((predicted_class == 1) & (buy_conf <= self.buy_threshold)) |
                         ((predicted_class == 2) & (sell_conf <= self.sell_threshold))).float()

        # Compute metrics
        metrics = {}

        # Decision rate (how often model decides to trade)
        trade_rate = (buy_decisions.sum() + sell_decisions.sum()) / len(buy_decisions)
        metrics['trade_rate'] = trade_rate
        metrics['hold_rate'] = hold_decisions.mean()

        # Confidence metrics
        avg_buy_conf = buy_conf[buy_decisions == 1].mean() if buy_decisions.sum() > 0 else torch.tensor(0.0, device=logits.device)
        avg_sell_conf = sell_conf[sell_decisions == 1].mean() if sell_decisions.sum() > 0 else torch.tensor(0.0, device=logits.device)
        metrics['avg_buy_confidence'] = avg_buy_conf
        metrics['avg_sell_confidence'] = avg_sell_conf

        return metrics
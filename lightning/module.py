import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryJaccardIndex,
    BinaryMatthewsCorrCoef,
    BinaryCohenKappa,
    BinaryCalibrationError,
    BinaryFBetaScore,
    BinaryHammingDistance,
    BinaryHingeLoss,
    BinaryPrecisionAtFixedRecall,
    BinaryRecallAtFixedPrecision,
)
import time
from pytorch_lightning.utilities import rank_zero_only, grad_norm

# Config imports removed - using dict-like access

# Performance tracking removed - using PyTorch Lightning's built-in profiler instead

# Trading metrics removed - not used in this module

# Import refactored components
from .losses.loss_factory import create_loss_function

logger = logging.getLogger(__name__)




class OrderBookLightningModule(pl.LightningModule):
    """Lightning module for order book transformer model training.
    
    This module handles:
    - Training and validation steps
    - Optimizer and scheduler configuration
    - Direct metrics logging via torchmetrics
    - Mixed precision training support
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimization_config: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        binary_classification_config=None,
        gradient_tracking_config=None,
        data_config=None,
        checkpoint_config=None,
        class_weights: Optional[torch.Tensor] = None,
        metrics_display_interval: int = 100,
        clear_optimizer_on_resume: bool = False,
    ):
        """Initialize the Lightning module.
        
        Args:
            model: The transformer model (from Section 3)
            optimization_config: Optimization configuration dataclass
            model_config: Model configuration dataclass
            binary_classification_config: Binary classification configuration containing loss params
            checkpoint_config: Checkpoint loading configuration
            class_weights: Optional class weights for imbalanced data
            metrics_display_interval: Interval for displaying metrics during training
            clear_optimizer_on_resume: If True, clear optimizer state when loading checkpoint
        """
        super().__init__()
        
        # Config is now expected to be passed in as dicts
        if optimization_config is None:
            raise ValueError("optimization_config must be provided")
        if model_config is None:
            raise ValueError("model_config must be provided")
            
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.optimization_config = optimization_config
        self.model_config = model_config
        self.binary_classification_config = binary_classification_config
        self.data_config = data_config
        self.checkpoint_config = checkpoint_config
        self.clear_optimizer_on_resume = clear_optimizer_on_resume
        
        # Track current portfolio for optimizer adjustment
        self.current_portfolio = None
        self.portfolio_optimizer_configs = {}
        self.profiles = getattr(data_config, 'profiles', []) if data_config else []
        
        # Extract commonly used parameters
        self.learning_rate = optimization_config.learning_rate
        self.weight_decay = optimization_config.weight_decay
        # Note: gradient_clip_val moved to TrainingConfig and handled by Lightning trainer
        
        # Dynamic loss weights for auto-calibration
        if binary_classification_config:
            self.prediction_threshold = binary_classification_config.prediction_threshold if hasattr(binary_classification_config, 'prediction_threshold') else 0.5
            logger.info(f"Binary classification config loaded: "
                       f"match_batch_distribution={getattr(binary_classification_config, 'match_batch_distribution', True)}, "
                       f"prediction_threshold={self.prediction_threshold}")
        else:
            self.prediction_threshold = 0.5
            logger.warning("No binary classification config provided - using defaults")
        
        # Determine classification mode
        self.classification_mode = getattr(model_config, 'classification_mode', 'binary')

        # Create loss function using factory based on mode
        self.criterion = create_loss_function(binary_classification_config, self.classification_mode)
        
        # Store loss function reference for callbacks
        self.loss_fn = self.criterion
        
        # Initialize optimized metric collections
        # Using MetricCollection for efficiency:
        # - Reduces from 60 individual metrics to 3 collections
        # - Training uses only fast metrics for speed
        # - Validation/Test include all metrics including expensive ones
        self._setup_metrics()
        
        # Track metrics display
        self.batch_count = 0  # Count batches processed
        self.display_interval = metrics_display_interval  # Display metrics every N batches
        self.last_display_batch = 0  # Last batch when metrics were displayed
        self.accumulated_samples = 0  # Total samples since last display
        
        # Track loss accumulation for averaging
        self.accumulated_loss = 0.0  # Sum of losses since last display
        self.loss_count = 0  # Number of loss values accumulated
        
        # Track individual loss components
        self.accumulated_primary_loss = 0.0
        self.accumulated_distribution_loss = 0.0
        
        # Track F1 metrics for averaging
        self.accumulated_soft_f1 = 0.0
        self.accumulated_soft_precision = 0.0
        self.accumulated_soft_recall = 0.0
        
        # Track FP/FN components for averaging
        self.accumulated_fp_fn_balance_loss = 0.0
        self.accumulated_total_error_loss = 0.0
        self.accumulated_distribution_matching_loss = 0.0
        self.accumulated_fp_rate = 0.0
        self.accumulated_fn_rate = 0.0
        
        # Track confidence penalty components
        self.accumulated_variance_penalty = 0.0
        
        # Track training timing
        self.training_start_time = time.time()  # Time when training started
        
        # Accumulators for batch-averaged ratios (replacing EMA)
        self.total_positives = 0.0  # 累计实际正样本数
        self.total_predictions = 0.0  # 累计预测概率和
        self.total_samples = 0  # 累计样本总数
        
        # Gradient tracking configuration
        self.gradient_tracking_config = gradient_tracking_config
        if gradient_tracking_config:
            self.track_component_gradients = gradient_tracking_config.enable_component_tracking
            self.component_gradient_freq = gradient_tracking_config.component_gradient_freq
        else:
            # Default values if no config provided
            self.track_component_gradients = True
            self.component_gradient_freq = 10
        
        self._last_loss_dict = None  # Store last loss dictionary for gradient computation
        self._last_component_gradients = {}  # Store component gradients for display
        
    
    def _setup_metrics(self):
        """Set up metric collections for efficient metric computation."""
        # Define core metrics based on classification mode
        def get_core_metrics():
            if self.classification_mode == 'ternary':
                from torchmetrics.classification import (
                    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,
                    MulticlassF1Score, MulticlassAUROC, MulticlassAveragePrecision,
                    MulticlassConfusionMatrix, MulticlassMatthewsCorrCoef,
                    MulticlassJaccardIndex
                )
                return {
                    'accuracy': MulticlassAccuracy(num_classes=3),
                    'precision': MulticlassPrecision(num_classes=3, average='macro'),
                    'recall': MulticlassRecall(num_classes=3, average='macro'),
                    'f1': MulticlassF1Score(num_classes=3, average='macro'),
                    'auroc': MulticlassAUROC(num_classes=3, average='macro'),
                    'avg_precision': MulticlassAveragePrecision(num_classes=3, average='macro'),
                    'confusion_matrix': MulticlassConfusionMatrix(num_classes=3),
                    'mcc': MulticlassMatthewsCorrCoef(num_classes=3),
                    'jaccard': MulticlassJaccardIndex(num_classes=3, average='macro'),
                }
            else:
                return {
                    'accuracy': BinaryAccuracy(),
                    'precision': BinaryPrecision(),
                    'recall': BinaryRecall(),
                    'f1': BinaryF1Score(),
                    'auroc': BinaryAUROC(),
                    'avg_precision': BinaryAveragePrecision(),
                    'confusion_matrix': BinaryConfusionMatrix(),
                    'mcc': BinaryMatthewsCorrCoef(),
                    'jaccard': BinaryJaccardIndex(),
                }
        
        # Define cheap additional metrics
        def get_cheap_metrics():
            if self.classification_mode == 'ternary':
                from torchmetrics.classification import (
                    MulticlassFBetaScore, MulticlassHammingDistance
                )
                return {
                    'fbeta': MulticlassFBetaScore(num_classes=3, beta=2.0, average='macro'),  # F2 score
                    'hamming': MulticlassHammingDistance(num_classes=3, average='macro'),
                }
            else:
                return {
                    'fbeta': BinaryFBetaScore(beta=2.0),  # F2 score
                    'hamming': BinaryHammingDistance(),
                }
        
        # Define expensive metrics (only for val/test)
        def get_expensive_metrics():
            if self.classification_mode == 'ternary':
                from torchmetrics.classification import (
                    MulticlassCohenKappa, MulticlassCalibrationError,
                    MulticlassHingeLoss
                )
                return {
                    'cohen_kappa': MulticlassCohenKappa(num_classes=3),
                    'calibration_error': MulticlassCalibrationError(num_classes=3, n_bins=10),
                    'hinge': MulticlassHingeLoss(num_classes=3),
                    # Note: Fixed recall/precision metrics not available for multiclass
                }
            else:
                return {
                    'cohen_kappa': BinaryCohenKappa(),
                    'calibration_error': BinaryCalibrationError(),
                    'hinge': BinaryHingeLoss(),
                    'precision_at_recall': BinaryPrecisionAtFixedRecall(min_recall=0.8),
                    'recall_at_precision': BinaryRecallAtFixedPrecision(min_precision=0.8),
                }
        
        # Training metrics: only core + cheap metrics for speed
        self.train_metrics = MetricCollection({
            **get_core_metrics(),
            **get_cheap_metrics(),
        })
        
        # Validation metrics: all metrics
        self.val_metrics = MetricCollection({
            **get_core_metrics(),
            **get_cheap_metrics(),
            **get_expensive_metrics(),
        })
        
        # Create a mapping for phase-based access
        self.metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
        }
    
    def _update_metrics(self, phase: str, preds: torch.Tensor, targets: torch.Tensor, 
                       probs: Optional[torch.Tensor] = None) -> None:
        """Update metrics for a given phase.
        
        Args:
            phase: One of 'train' or 'val'
            preds: Binary predictions
            targets: Ground truth labels
            probs: Optional probabilities for metrics that need them
        """
        metrics = self.metrics[phase]
        
        # Update prediction-based metrics
        pred_metrics = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix',
                       'mcc', 'jaccard', 'fbeta', 'hamming', 'cohen_kappa']
        
        for name, metric in metrics.items():
            if name in pred_metrics:
                metric(preds, targets)
        
        # Update probability-based metrics
        if probs is not None:
            prob_metrics = ['auroc', 'avg_precision', 'calibration_error', 'hinge',
                           'precision_at_recall', 'recall_at_precision']
            
            for name, metric in metrics.items():
                if name in prob_metrics:
                    metric(probs, targets)
    
    def _compute_and_log_metrics(self, phase: str, on_step: bool = True, 
                                on_epoch: bool = False, prog_bar: bool = False,
                                prog_bar_metrics: Optional[List[str]] = None,
                                batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Compute and log metrics for a given phase.
        
        Args:
            phase: One of 'train' or 'val'
            on_step: Whether to log on step (always True for step-based training)
            on_epoch: Whether to log on epoch (always False for step-based training)
            prog_bar: Whether to show in progress bar (applies to all metrics)
            prog_bar_metrics: List of specific metrics to show in progress bar
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = self.metrics[phase]
        computed_metrics = metrics.compute()
        
        # Define default progress bar metrics per phase
        if prog_bar_metrics is None:
            if phase == 'train':
                prog_bar_metrics = ['accuracy', 'f1']
            elif phase == 'val':
                prog_bar_metrics = ['accuracy', 'f1', 'auroc']
            else:
                prog_bar_metrics = []
        
        # Handle special cases and log metrics
        for name, value in computed_metrics.items():
            # Skip complex metrics that return non-tensor values
            if name == 'confusion_matrix':
                continue
                
            # Handle metrics that return tuples (precision_at_recall, recall_at_precision)
            if isinstance(value, tuple):
                value = value[0]  # Take the first value (metric, not threshold)
            
            # Only log scalar tensors
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                # Determine if this metric should be in progress bar
                show_in_prog_bar = prog_bar or (name in prog_bar_metrics)
                if batch_size is not None:
                    # Let Lightning handle sync automatically for better performance
                    self.log(f'{phase}_{name}', value, on_step=on_step, on_epoch=on_epoch, prog_bar=show_in_prog_bar, batch_size=batch_size)
                else:
                    # Let Lightning handle sync automatically for better performance
                    self.log(f'{phase}_{name}', value, on_step=on_step, on_epoch=on_epoch, prog_bar=show_in_prog_bar)
        
        return computed_metrics
        
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, 306]
            mask: Optional attention mask
            
        Returns:
            Dictionary of predictions from the model
        """
        # Additional input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.warning("NaN or Inf in forward input, cleaning data")
            x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Forward pass
        output = self.model(x, mask=mask)
        
        # Check output for NaN/Inf
        if 'logits' in output:
            if torch.isnan(output['logits']).any() or torch.isinf(output['logits']).any():
                logger.warning("NaN or Inf in model logits, using nan_to_num")
                output['logits'] = torch.clamp_min(output['logits'], -50.0)
                output['logits'] = torch.nan_to_num(output['logits'], nan=0.0)
        
        return output
    
    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Prepare targets for classification.

        Args:
            targets: Raw target tensor, typically shape (batch_size, 1)

        Returns:
            Processed targets with shape (batch_size,) and dtype torch.long
        """
        # Ensure we have at least 1D tensor
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
        
        # Remove singleton dimensions but preserve batch dimension
        if targets.dim() > 1:
            # For shape (batch_size, 1), remove the last dimension
            # For shape (1, 1), this becomes (1,) not scalar
            targets = targets.view(targets.shape[0])
        
        # PHASE 3: Log invalid targets from data leakage fix
        if targets.dtype == torch.float:
            invalid_mask = targets < 0  # -1 indicates invalid target
            num_invalid = invalid_mask.sum().item()
            if num_invalid > 0:
                total_targets = len(targets)
                logger.info(f"[LEAKAGE FIX] Found {num_invalid}/{total_targets} invalid targets "
                           f"({100*num_invalid/total_targets:.1f}%) from boundary handling")
        
        # Convert based on classification mode
        if self.classification_mode == 'ternary':
            # Ternary classification: targets should already be 0, 1, or 2
            if targets.dtype == torch.float:
                # Round to nearest integer for ternary targets
                targets = torch.round(targets).long()
                # Clamp to valid range [0, 2]
                targets = torch.clamp(targets, 0, 2)
        else:
            # Binary classification: 1 if positive, 0 if negative/zero
            if targets.dtype == torch.float:
                targets = (targets > 0).long()
        
        # Ensure targets are long type for classification
        if targets.dtype != torch.long:
            targets = targets.long()
                
        return targets
    
    def _compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss based on task type.
        
        Args:
            predictions: Dictionary containing predictions ('logits' for binary classification)
            targets: Target labels for binary classification
            
        Returns:
            Loss value
        """
        # Skip loss computation if no targets provided (e.g., inference mode)
        if targets is None:
            # Return a dummy loss that still depends on the model output to satisfy DDP
            # This ensures gradients flow through all parameters
            logits = predictions['logits'] if isinstance(predictions, dict) else predictions
            return logits.sum() * 0.0  # Results in 0 loss but maintains gradient flow
        
        # PHASE 3: Filter out invalid targets from data leakage fix
        # First check for invalid targets before preparation
        original_targets = targets.clone()
        if targets.dim() > 1:
            targets_flat = targets.view(targets.shape[0])
        else:
            targets_flat = targets
        
        # Invalid targets are marked as -1
        valid_mask = targets_flat >= 0
        num_valid = valid_mask.sum().item()
        total_samples = len(targets_flat)
        
        if num_valid == 0:
            # No valid targets in this batch - return zero loss
            logger.warning(f"[LEAKAGE FIX] No valid targets in batch (all {total_samples} invalid)")
            logits = predictions['logits'] if isinstance(predictions, dict) else predictions
            return logits.sum() * 0.0
        
        if num_valid < total_samples:
            # Some invalid targets - filter them out
            num_invalid = total_samples - num_valid
            logger.debug(f"[LEAKAGE FIX] Filtering {num_invalid}/{total_samples} invalid targets in loss")
            
            # Filter both logits and targets
            logits = predictions['logits']
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            # Apply valid mask
            logits = logits[valid_mask]
            targets = original_targets[valid_mask]
        else:
            # All targets are valid - proceed normally
            logits = predictions['logits']
            
        # Prepare targets based on task type
        targets = self._prepare_targets(targets)
        
        # Handle different classification modes
        if self.classification_mode == 'ternary':
            # For ternary classification, logits should be [batch_size, 3]
            # Targets should be long integers [0, 1, 2]
            loss_output = self.criterion(logits, targets.long(), return_components=True)
            # Convert to dictionary format if needed
            if isinstance(loss_output, dict):
                loss_dict = loss_output
            else:
                loss_dict = {'total': loss_output, 'components': {}}
        else:
            # Binary classification - ensure logits are 1D
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            # Use SigmoidDistributionLoss directly (it returns a dictionary)
            loss_dict = self.criterion(logits, targets.float())
        
        # Extract total loss for backward pass
        # Ternary loss uses 'loss' key, binary loss uses 'total' key
        loss = loss_dict.get('loss', loss_dict.get('total'))
        
        # Store loss dict for gradient tracking
        self._last_loss_dict = loss_dict
        
        # Store loss components for logging
        if self.classification_mode == 'ternary':
            # Ternary loss components
            self.last_loss_components = {
                'total_loss': loss_dict['loss'].item() if 'loss' in loss_dict else loss_dict['total'].item()
            }
            # Add component losses if available
            if 'components' in loss_dict:
                for key, value in loss_dict['components'].items():
                    if isinstance(value, torch.Tensor):
                        self.last_loss_components[key] = value.item()

            # Log ternary loss components to TensorBoard
            batch_size = logits.shape[0]
            self.log('train/loss', loss, on_step=True, on_epoch=False, batch_size=batch_size)  # Use the extracted loss variable

            # Log individual components if available
            if 'components' in loss_dict:
                for key, value in loss_dict['components'].items():
                    if isinstance(value, torch.Tensor):
                        self.log(f'train/ternary_{key}', value, on_step=True, on_epoch=False, batch_size=batch_size)

            # Store batch statistics for ternary classification
            self.last_batch_stats = {
                'batch_size': logits.shape[0],
                'num_classes': 3,
                'classification_mode': 'ternary'
            }

            # Calculate and log class distribution for ternary
            if targets.dtype == torch.long:
                for class_idx in range(3):
                    class_count = (targets == class_idx).sum().item()
                    class_ratio = class_count / batch_size
                    class_name = ['hold', 'buy', 'sell'][class_idx]
                    self.log(f'train/{class_name}_ratio', class_ratio, on_step=True, on_epoch=False, batch_size=batch_size)

            return loss_dict  # Return the full dict, not just the loss
        else:
            # Binary loss components (original logic)
            # Determine which loss type we're using
            if 'sigmoid_loss' in loss_dict:
                primary_loss_key = 'sigmoid_loss'
            elif 'bce_loss' in loss_dict:
                primary_loss_key = 'bce_loss'
            elif 'focal_loss' in loss_dict:
                primary_loss_key = 'focal_loss'
            else:
                raise ValueError(f"Unknown loss type in loss_dict. Keys: {list(loss_dict.keys())}")

            self.last_loss_components = {
                'primary_loss': loss_dict['weighted_primary_loss'].item(),
                'distribution_loss': loss_dict['weighted_regularization_loss'].item(),
                'total_loss': loss_dict['total'].item()
            }
        
        # Store F1 components if available
        if 'soft_f1' in loss_dict:
            self.last_loss_components.update({
                'soft_f1': loss_dict['soft_f1'].item(),
                'soft_precision': loss_dict['soft_precision'].item(),
                'soft_recall': loss_dict['soft_recall'].item()
            })
        
        # Store FP/FN and KL divergence components if available  
        if 'total_error_loss' in loss_dict:
            self.last_loss_components.update({
                'total_error_loss': loss_dict['total_error_loss'].item(),
                'fp_rate': loss_dict['fp_rate'].item(),
                'fn_rate': loss_dict['fn_rate'].item()
            })
            
        # Store balance loss if available
        if 'balance_loss' in loss_dict:
            self.last_loss_components.update({
                'balance_loss': loss_dict['balance_loss'].item()
            })
        
        # Store confidence penalty components if available
        if 'variance_penalty' in loss_dict:
            self.last_loss_components.update({
                'variance_penalty': loss_dict['variance_penalty'].item()
            })
        
        # Store confidence components if available
        if 'confidence' in loss_dict:
            self.last_loss_components.update({
                'confidence': loss_dict['confidence'].item(),
                'normalized_confidence': loss_dict['normalized_confidence'].item()
            })
            
        # Store regularization loss (formerly fp_fn_balance_loss)
        if 'regularization_loss' in loss_dict:
            self.last_loss_components.update({
                'fp_fn_balance_loss': loss_dict['regularization_loss'].item(),  # Keep old name for compatibility
                'regularization_loss': loss_dict['regularization_loss'].item()
            })
            
        # Legacy support for distribution_matching_loss (maps to balance_loss)
        if 'balance_loss' in loss_dict:
            self.last_loss_components.update({
                'distribution_matching_loss': loss_dict['balance_loss'].item()  # Keep old name for compatibility
            })
        
        # Store batch statistics for distribution verification
        self.last_batch_stats = {
            'actual_positive_ratio': targets.float().mean().item(),
            'predicted_positive_ratio': torch.sigmoid(logits).mean().item(),
            'batch_size': logits.shape[0],
            'num_positives': targets.sum().item(),
            'num_negatives': (1 - targets).sum().item()
        }
        
        # Log individual loss components
        batch_size = logits.shape[0]
        # Log the primary loss (sigmoid/bce/focal) - both raw and weighted
        self.log(f'train/{primary_loss_key}', loss_dict[primary_loss_key], on_step=True, on_epoch=False, batch_size=batch_size)
        self.log('train/weighted_primary_loss', loss_dict['weighted_primary_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
        self.log('train/distribution_loss', loss_dict['weighted_regularization_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Log FP/FN balanced loss components if available
        if 'fp_fn_balance_loss' in loss_dict:
            self.log('train/fp_fn_balance_loss', loss_dict['fp_fn_balance_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'total_error_loss' in loss_dict:
                self.log('train/total_error_loss', loss_dict['total_error_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'distribution_matching_loss' in loss_dict:
                self.log('train/distribution_matching_loss', loss_dict['distribution_matching_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'fp_rate' in loss_dict:
                self.log('train/fp_rate', loss_dict['fp_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'fn_rate' in loss_dict:
                self.log('train/fn_rate', loss_dict['fn_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'target_fp_rate' in loss_dict:
                self.log('train/target_fp_rate', loss_dict['target_fp_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'target_fn_rate' in loss_dict:
                self.log('train/target_fn_rate', loss_dict['target_fn_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            
            # Log distribution matching metrics
            if 'pred_rate' in loss_dict:
                self.log('train/pred_rate', loss_dict['pred_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            if 'target_rate' in loss_dict:
                self.log('train/target_rate', loss_dict['target_rate'], on_step=True, on_epoch=False, batch_size=batch_size)
            
            # Log confusion matrix components based on whether using soft or hard FP/FN
            if 'soft_tp' in loss_dict:
                # Using soft FP/FN
                self.log('train/soft_tp', loss_dict['soft_tp'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/soft_fp', loss_dict['soft_fp'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/soft_fn', loss_dict['soft_fn'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/soft_tn', loss_dict['soft_tn'], on_step=True, on_epoch=False, batch_size=batch_size)
                
                # Log sum for verification
                soft_sum = loss_dict['soft_tp'] + loss_dict['soft_fp'] + loss_dict['soft_fn'] + loss_dict['soft_tn']
                self.log('train/soft_confusion_sum', soft_sum, on_step=True, on_epoch=False, batch_size=batch_size)
            elif 'hard_tp' in loss_dict:
                # Using hard FP/FN
                self.log('train/hard_tp', loss_dict['hard_tp'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/hard_fp', loss_dict['hard_fp'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/hard_fn', loss_dict['hard_fn'], on_step=True, on_epoch=False, batch_size=batch_size)
                self.log('train/hard_tn', loss_dict['hard_tn'], on_step=True, on_epoch=False, batch_size=batch_size)
                
                # Log actual hard values if available
                if 'actual_hard_fp' in loss_dict:
                    self.log('train/actual_hard_fp', loss_dict['actual_hard_fp'], on_step=True, on_epoch=False, batch_size=batch_size)
                    self.log('train/actual_hard_fn', loss_dict['actual_hard_fn'], on_step=True, on_epoch=False, batch_size=batch_size)
                
                # Log sum for verification
                hard_sum = loss_dict['hard_tp'] + loss_dict['hard_fp'] + loss_dict['hard_fn'] + loss_dict['hard_tn']
                self.log('train/hard_confusion_sum', hard_sum, on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Log F1 components based on whether using soft or hard FP/FN
        if 'soft_f1' in loss_dict:
            self.log('train/soft_f1', loss_dict['soft_f1'], on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/soft_precision', loss_dict['soft_precision'], on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/soft_recall', loss_dict['soft_recall'], on_step=True, on_epoch=False, batch_size=batch_size)
        elif 'hard_f1' in loss_dict:
            self.log('train/hard_f1', loss_dict['hard_f1'], on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/hard_precision', loss_dict['hard_precision'], on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/hard_recall', loss_dict['hard_recall'], on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Log confidence penalty and confidence components if available
        if 'variance_penalty' in loss_dict:
            self.log('train/variance_penalty', loss_dict['variance_penalty'], on_step=True, on_epoch=False, batch_size=batch_size)
        if 'confidence' in loss_dict:
            self.log('train/confidence', loss_dict['confidence'], on_step=True, on_epoch=False, batch_size=batch_size)
        if 'normalized_confidence' in loss_dict:
            self.log('train/normalized_confidence', loss_dict['normalized_confidence'], on_step=True, on_epoch=False, batch_size=batch_size)
        if 'balance_loss' in loss_dict:
            self.log('train/balance_loss', loss_dict['balance_loss'], on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Log distribution matching info if enabled
        if self.binary_classification_config is not None and getattr(self.binary_classification_config, 'match_batch_distribution', True):
            # Get probabilities for logging
            probs = torch.sigmoid(logits)
            
            # Calculate actual and predicted distributions for current batch
            actual_positive_ratio = targets.float().mean()
            predicted_positive_ratio = probs.mean()
            
            # Update batch accumulators
            actual_positives_batch = targets.float().sum().item()
            predicted_positives_batch = probs.sum().item()
            batch_samples = batch_size
            
            # Accumulate to total counters
            self.total_positives += actual_positives_batch
            self.total_predictions += predicted_positives_batch
            self.total_samples += batch_samples
            
            # Calculate cumulative averages
            actual_positive_ratio_avg = self.total_positives / self.total_samples if self.total_samples > 0 else 0.0
            predicted_positive_ratio_avg = self.total_predictions / self.total_samples if self.total_samples > 0 else 0.0
            
            # Calculate cumulative distribution error
            avg_distribution_error = abs(predicted_positive_ratio_avg - actual_positive_ratio_avg)
            
            # Log distribution info every 10 steps
            if self.global_step % 10 == 0:
                # Check if we're using global distribution matching
                is_global = (self.binary_classification_config is not None and 
                           getattr(self.binary_classification_config, 'global_distribution_matching', False) and
                           torch.distributed.is_available() and torch.distributed.is_initialized() and
                           torch.distributed.get_world_size() > 1)
                
                mode = "GLOBAL" if is_global else "LOCAL"
                
                # Get device info
                device_info = ""
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    world_size = torch.distributed.get_world_size()
                    device_info = f" [GPU {rank}/{world_size}]"
                
                # Log balance loss components if available
                if 'balance_loss' in loss_dict:
                    logger.info(f"Balance enforcement ({mode}){device_info} - step: {self.global_step}, "
                               f"balance_loss: {loss_dict['balance_loss'].item():.4f}, "
                               f"variance_penalty: {loss_dict['variance_penalty'].item():.4f}, "
                               f"confidence: {loss_dict['confidence'].item():.4f}, "
                               f"actual_positive_ratio(avg): {actual_positive_ratio_avg:.4f}, "
                               f"batch_size: {batch_size}")
                else:
                    logger.info(f"L1 Distribution matching ({mode}){device_info} - step: {self.global_step}, "
                               f"actual_positive_ratio(avg): {actual_positive_ratio_avg:.4f}, "
                               f"predicted_positive_ratio(avg): {predicted_positive_ratio_avg:.4f}, "
                               f"distribution_error(avg): {avg_distribution_error:.4f}, "
                               f"batch_size: {batch_size}")

            # Log distribution metrics
            self.log('train/actual_positive_ratio', actual_positive_ratio, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/predicted_positive_ratio', predicted_positive_ratio, on_step=True, on_epoch=False, batch_size=batch_size)
            self.log('train/distribution_error', torch.abs(predicted_positive_ratio - actual_positive_ratio), on_step=True, on_epoch=False, batch_size=batch_size)
        
        return loss
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get current batch distribution statistics for verification.
        
        Returns:
            Dictionary containing batch statistics
        """
        if hasattr(self, 'last_batch_stats'):
            return self.last_batch_stats.copy()
        return {}
    
    def _shared_eval_step(self, batch: Dict[str, torch.Tensor], phase: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Shared evaluation logic for test steps.
        
        Args:
            batch: Batch dictionary
            phase: 'test' only
            
        Returns:
            Tuple of (loss, outputs dictionary)
        """
        features = batch['features']
        # Support both 'targets' and 'labels' keys
        targets = batch.get('targets')
        if targets is None:
            targets = batch.get('labels')
        mask = batch.get('mask', None)
        
        # Forward pass
        predictions = self(features, mask=mask)
        
        # Single batch debugging - BEFORE loss computation
        if self.global_step == 0:
            logits = predictions['logits']
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            
            # Log comprehensive statistics
            print("\n" + "="*80)
            print("🔍 SINGLE BATCH DEBUGGING - BEFORE LOSS COMPUTATION")
            print("="*80)
            print(f"Batch size: {batch_size}")
            print(f"\nProbability Statistics:")
            if self.classification_mode == 'ternary':
                # For ternary, show statistics per class
                for cls_idx, cls_name in enumerate(['Hold', 'Buy', 'Sell']):
                    cls_probs = probs[:, cls_idx]
                    print(f"  {cls_name} class:")
                    print(f"    Min: {cls_probs.min().item():.6f}")
                    print(f"    Max: {cls_probs.max().item():.6f}")
                    print(f"    Mean: {cls_probs.mean().item():.6f}")
                    print(f"    Std: {cls_probs.std().item():.6f}")

                # Check predictions (argmax of probs)
                predictions = torch.argmax(probs, dim=1)
                print(f"\nPrediction distribution:")
                print(f"  Hold (0): {(predictions == 0).sum().item()} ({(predictions == 0).sum().item()/batch_size*100:.1f}%)")
                print(f"  Buy (1): {(predictions == 1).sum().item()} ({(predictions == 1).sum().item()/batch_size*100:.1f}%)")
                print(f"  Sell (2): {(predictions == 2).sum().item()} ({(predictions == 2).sum().item()/batch_size*100:.1f}%)")
            else:
                # For binary
                print(f"  Min: {probs.min().item():.6f}")
                print(f"  Max: {probs.max().item():.6f}")
                print(f"  Mean: {probs.mean().item():.6f}")
                print(f"  Std: {probs.std().item():.6f}")
                print(f"  Median: {probs.median().item():.6f}")

                # Check for extreme values
                near_zero = (probs < 0.01).sum().item()
                near_one = (probs > 0.99).sum().item()
                print(f"\nExtreme values:")
                print(f"  Probabilities < 0.01: {near_zero} ({near_zero/batch_size*100:.1f}%)")
                print(f"  Probabilities > 0.99: {near_one} ({near_one/batch_size*100:.1f}%)")

                # Histogram of probabilities
                hist_bins = 10
                hist, bin_edges = torch.histogram(probs.cpu(), bins=hist_bins)
                print(f"\nProbability histogram ({hist_bins} bins):")
                for i in range(hist_bins):
                    bar_length = int(hist[i].item() / batch_size * 50)
                    bar = "█" * bar_length
                    print(f"  [{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}]: {bar} {hist[i].item()} ({hist[i].item()/batch_size*100:.1f}%)")
            
            # Check actual targets distribution
            if targets is not None:
                targets_prepared = self._prepare_targets(targets)
                print(f"\nTarget distribution:")
                if self.classification_mode == 'ternary':
                    print(f"  Hold (0): {(targets_prepared == 0).sum().item()} ({(targets_prepared == 0).sum().item()/batch_size*100:.1f}%)")
                    print(f"  Buy (1): {(targets_prepared == 1).sum().item()} ({(targets_prepared == 1).sum().item()/batch_size*100:.1f}%)")
                    print(f"  Sell (2): {(targets_prepared == 2).sum().item()} ({(targets_prepared == 2).sum().item()/batch_size*100:.1f}%)")
                else:
                    print(f"  Zeros (sell): {(targets_prepared == 0).sum().item()} ({(targets_prepared == 0).sum().item()/batch_size*100:.1f}%)")
                    print(f"  Ones (buy): {(targets_prepared == 1).sum().item()} ({(targets_prepared == 1).sum().item()/batch_size*100:.1f}%)")
            
            # Check logits distribution
            print(f"\nLogit Statistics:")
            if self.classification_mode == 'ternary':
                # For ternary, flatten logits for overall statistics
                logit_flat = logits.flatten()
                print(f"  Min: {logit_flat.min().item():.6f}")
                print(f"  Max: {logit_flat.max().item():.6f}")
                print(f"  Mean: {logit_flat.mean().item():.6f}")
                print(f"  Std: {logit_flat.std().item():.6f}")
            else:
                # For binary, logits is 1D
                print(f"  Min: {logits.min().item():.6f}")
                print(f"  Max: {logits.max().item():.6f}")
                print(f"  Mean: {logits.mean().item():.6f}")
                print(f"  Std: {logits.std().item():.6f}")
            
            # Sample some actual values
            print(f"\nFirst 10 predictions:")
            for i in range(min(10, batch_size)):
                target_val = targets_prepared[i].item() if targets is not None else "N/A"
                if self.classification_mode == 'ternary':
                    # For ternary, show all 3 class logits and probs
                    print(f"  Sample {i}: logits={logits[i].tolist()}, probs={probs[i].tolist()}, target={target_val}")
                else:
                    # For binary, single logit and prob
                    print(f"  Sample {i}: logit={logits[i].item():.4f}, prob={probs[i].item():.4f}, target={target_val}")
                
            # Check input features diversity
            print(f"\nInput Features Analysis:")
            print(f"  Shape: {features.shape}")
            print(f"  Mean: {features.mean().item():.6f}")
            print(f"  Std: {features.std().item():.6f}")
            print(f"  Min: {features.min().item():.6f}")
            print(f"  Max: {features.max().item():.6f}")
            
            # Check if all windows are similar
            # Compare first and last window
            if batch_size > 1:
                first_window = features[0]
                last_window = features[-1]
                diff = (first_window - last_window).abs()
                print(f"\nWindow diversity check:")
                print(f"  Max difference between first and last window: {diff.max().item():.6f}")
                print(f"  Mean difference: {diff.mean().item():.6f}")
                
                # Check variance across batch dimension
                batch_variance = features.var(dim=0).mean().item()
                print(f"  Average variance across batch: {batch_variance:.6f}")
                
                # More detailed window comparison
                # Check if any windows are exactly identical
                for i in range(min(5, batch_size-1)):
                    if torch.equal(features[i], features[i+1]):
                        print(f"  ⚠️  WARNING: Window {i} and {i+1} are EXACTLY identical!")
                    else:
                        max_diff = (features[i] - features[i+1]).abs().max().item()
                        print(f"  Window {i} vs {i+1}: max_diff={max_diff:.6f}")
                
                # Check oldest timestep variance (features[:, -1, :] is the oldest/furthest back in time)
                oldest_hidden_variance = features[:, -1, :].var(dim=0).mean().item()
                print(f"  Oldest timestep variance across batch: {oldest_hidden_variance:.6f}")
                
                # Check if oldest timesteps are identical across samples
                oldest_timesteps = features[:, -1, :]  # Index -1 is the oldest data (max lookback)
                print(f"\nOldest timestep analysis:")
                for i in range(min(3, batch_size-1)):
                    if torch.equal(oldest_timesteps[i], oldest_timesteps[i+1]):
                        print(f"  ⚠️  CRITICAL: Oldest timestep of window {i} and {i+1} are EXACTLY identical!")
                    else:
                        max_diff = (oldest_timesteps[i] - oldest_timesteps[i+1]).abs().max().item()
                        mean_diff = (oldest_timesteps[i] - oldest_timesteps[i+1]).abs().mean().item()
                        print(f"  Oldest timestep {i} vs {i+1}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            
            print("="*80 + "\n", flush=True)
        
        # Compute loss
        loss = self._compute_loss(predictions, targets)
        
        # Log loss
        batch_size = features.shape[0]
        # Log loss with automatic sync (Lightning handles this efficiently)
        self.log(f'{phase}_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        
        # Prepare outputs
        outputs = {'loss': loss}
        
        # Prepare targets
        targets = self._prepare_targets(targets)
        
        # Classification metrics
        # Get probabilities from logits with numerical stability
        logits = predictions['logits']

        # Handle different classification modes
        if self.classification_mode == 'ternary':
            # Ternary classification
            # Logits shape: [batch_size, 3]
            probs = F.softmax(logits, dim=-1)  # Shape: [batch_size, 3]
            preds = torch.argmax(probs, dim=-1)  # Shape: [batch_size]
        else:
            # Binary classification
            # Ensure logits are 1D
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            # Only clamp extreme negatives to prevent underflow
            logits = torch.clamp_min(logits, -50.0)
            probs = torch.sigmoid(logits)
            # Additional safety check
            probs = torch.nan_to_num(probs, nan=self.prediction_threshold)
            preds = (probs > self.prediction_threshold).long()
        
        # Update and log metrics based on phase
        if phase == 'val':
            # Update metrics using the new system
            self._update_metrics('val', preds, targets, probs)
            
            # Compute and log all metrics (progress bar metrics are handled automatically)
            computed_metrics = self._compute_and_log_metrics('val', on_step=True, on_epoch=False, batch_size=features.shape[0])
            
            # Log confusion matrix components separately
            if 'confusion_matrix' in computed_metrics:
                cm = computed_metrics['confusion_matrix']
                # Ensure confusion matrix has correct shape
                if cm.dim() == 2 and cm.shape == (2, 2):
                    # For binary classification, confusion matrix is 2x2
                    tn, fp, fn, tp = cm.flatten()
                    self.log('val_true_positives', tp.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_true_negatives', tn.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_false_positives', fp.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_false_negatives', fn.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                elif cm.numel() == 4:
                    # Handle case where cm might already be flattened or have different shape
                    cm_flat = cm.view(-1)
                    tn, fp, fn, tp = cm_flat[0], cm_flat[1], cm_flat[2], cm_flat[3]
                    self.log('val_true_positives', tp.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_true_negatives', tn.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_false_positives', fp.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
                    self.log('val_false_negatives', fn.float(), on_step=True, on_epoch=False, batch_size=features.shape[0])
            
            
        
        return loss, outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic.
        
        Args:
            batch: Batch dictionary containing:
                - 'features': Input features [batch_size, seq_len, 307]
                - 'targets': Target dictionary
                - 'mask': Optional attention mask
                - 'file_path': Optional file path for tracking
            batch_idx: Index of the current batch
            
        Returns:
            Training loss
        """
        # Detect and adjust optimizer for current portfolio
        self._adjust_optimizer_for_portfolio(batch)
        
        features = batch['features']
        # Support both 'targets' and 'labels' keys
        targets = batch.get('targets')
        if targets is None:
            targets = batch.get('labels')
        mask = batch.get('mask', None)
        batch_size = features.shape[0]
        
        # Add input noise during early training to prevent collapse
        if self.training and self.global_step < 100:
            noise_scale = 0.01 * (1.0 - self.global_step / 100.0)  # Decay noise
            features = features + torch.randn_like(features) * noise_scale
            if self.global_step == 0:
                print(f"\n🎲 Adding input noise (scale={noise_scale:.4f}) to prevent collapse\n", flush=True)
        
        # Print feature information on first batch (only if trainer is attached)
        if self.global_step == 0 and batch_idx == 0:
            try:
                # This will raise RuntimeError if trainer is not attached
                if self.trainer:
                    self._log_feature_information(features, batch_idx)
            except RuntimeError:
                # Trainer not attached yet, skip logging feature info
                pass
        
        # Track chunk changes
        file_path = batch.get('file_path', None)
        chunk_start = batch.get('chunk_start', None)
        
        # Extract single file path if it's a list/tensor
        if file_path is not None:
            if isinstance(file_path, list):
                file_path = file_path[0] if len(file_path) > 0 else None
            elif torch.is_tensor(file_path):
                # If it's a tensor of strings, get the first element
                file_path = file_path[0] if file_path.numel() > 0 else None
        
        # Extract chunk start if it's a tensor or list
        if chunk_start is not None:
            if torch.is_tensor(chunk_start):
                chunk_start = chunk_start[0].item() if chunk_start.numel() > 0 else None
            elif isinstance(chunk_start, list):
                chunk_start = chunk_start[0] if len(chunk_start) > 0 else None
        
        # Create chunk key
        chunk_key = None
        if file_path is not None and chunk_start is not None:
            chunk_key = f"{file_path}::{chunk_start}"
                
        # Track batch count and samples
        self.batch_count += 1
        batch_size = len(targets) if targets is not None else features.shape[0]
        self.accumulated_samples += batch_size
        
        # Display metrics every N batches
        if self.batch_count % self.display_interval == 0:
            # Gather global metrics from all GPUs (all ranks must participate)
            global_batch_count, global_samples, global_accumulated_loss, global_loss_count, \
                global_primary_loss, global_distribution_loss, global_fp_fn_balance_loss, \
                global_total_error_loss, global_distribution_matching_loss, global_fp_rate, global_fn_rate, \
                global_variance_penalty = self._gather_global_metrics()
            
            # Only rank 0 displays the metrics
            if self.trainer.is_global_zero:
                self._display_accumulated_metrics(
                    batch_range=(self.last_display_batch + 1, self.batch_count),
                    samples=self.accumulated_samples,
                    phase='train',
                    global_batch_count=global_batch_count,
                    global_samples=global_samples,
                    global_accumulated_loss=global_accumulated_loss,
                    global_loss_count=global_loss_count,
                    global_primary_loss=global_primary_loss,
                    global_distribution_loss=global_distribution_loss,
                    global_fp_fn_balance_loss=global_fp_fn_balance_loss,
                    global_total_error_loss=global_total_error_loss,
                    global_distribution_matching_loss=global_distribution_matching_loss,
                    global_fp_rate=global_fp_rate,
                    global_fn_rate=global_fn_rate,
                    global_variance_penalty=global_variance_penalty
                )
            
            # All ranks reset their counters
            self.last_display_batch = self.batch_count
            self.accumulated_samples = 0
            self.accumulated_loss = 0.0
            self.loss_count = 0
            self.accumulated_primary_loss = 0.0
            self.accumulated_distribution_loss = 0.0
            self.accumulated_soft_f1 = 0.0
            self.accumulated_soft_precision = 0.0
            self.accumulated_soft_recall = 0.0
            self.accumulated_fp_fn_balance_loss = 0.0
            self.accumulated_total_error_loss = 0.0
            self.accumulated_distribution_matching_loss = 0.0
            self.accumulated_fp_rate = 0.0
            self.accumulated_fn_rate = 0.0
            self.accumulated_variance_penalty = 0.0
            # Reset metrics for next accumulation period
            self._reset_train_metrics()
        
        
        # Validate input data
        if torch.isnan(features).any():
            logger.warning(f"NaN values detected in features at batch {batch_idx}")
            # Replace NaN values with zeros
            features = torch.nan_to_num(features, nan=0.0)
        
        if torch.isinf(features).any():
            logger.warning(f"Inf values detected in features at batch {batch_idx}")
            # Remove clamping - let gradients flow naturally
            # Only replace inf/nan values if they exist
            if torch.isinf(features).any() or torch.isnan(features).any():
                features = torch.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Forward pass
        predictions = self.forward(features, mask=mask)
        
        # Compute loss
        loss_output = self._compute_loss(predictions, targets)

        # Extract loss value from dict if ternary mode
        if isinstance(loss_output, dict):
            loss = loss_output.get('loss', loss_output.get('total'))
        else:
            loss = loss_output

        # Clear GPU cache to ensure no cross-batch information leakage
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()

        # Add L2 regularization on logits to prevent them from exploding
        if 'logits' in predictions:
            logits = predictions['logits']
            logit_penalty = 0.01 * torch.mean(logits ** 2)
            loss = loss + logit_penalty
        
        # Check if loss is NaN
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at batch {batch_idx}")
            # Log more details about what caused the NaN
            logger.error(f"Logits stats - mean: {predictions['logits'].mean().item():.4f}, "
                        f"std: {predictions['logits'].std().item():.4f}, "
                        f"min: {predictions['logits'].min().item():.4f}, "
                        f"max: {predictions['logits'].max().item():.4f}")
            logger.error(f"Contains NaN in logits: {torch.isnan(predictions['logits']).any().item()}")
            logger.error(f"Contains Inf in logits: {torch.isinf(predictions['logits']).any().item()}")
            
            # Fail immediately on NaN loss - no fallback
            raise ValueError(
                f"NaN loss detected at batch {batch_idx}. "
                f"Logits stats - mean: {predictions['logits'].mean().item():.4f}, "
                f"std: {predictions['logits'].std().item():.4f}. "
                f"Check learning rate, data quality, and model stability."
            )
        
        # Log metrics - Lightning automatically syncs on_epoch=False in DDP
        # Only sync on_step if absolutely needed (expensive in DDP)
        # IMPORTANT: Log the ACTUAL loss being optimized (total_loss), not just BCE
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        
        # Log category scales every 100 steps if available
        if self.global_step % 100 == 0 and hasattr(self.model, 'embedding') and hasattr(self.model.embedding, 'category_scales'):
            for category, scale in self.model.embedding.category_scales.items():
                self.log(f'train/category_scale_{category}', scale.item(), on_step=True, batch_size=batch_size)
        
        # Accumulate the TOTAL loss for accurate averaging (weighted by batch size)
        self.accumulated_loss += loss.item() * batch_size
        self.loss_count += 1
        
        # Accumulate individual loss components if available
        if hasattr(self, 'last_loss_components') and self.last_loss_components:
            # Check which keys exist (binary vs ternary have different keys)
            if 'primary_loss' in self.last_loss_components:
                self.accumulated_primary_loss += self.last_loss_components['primary_loss'] * batch_size
            if 'distribution_loss' in self.last_loss_components:
                self.accumulated_distribution_loss += self.last_loss_components['distribution_loss'] * batch_size
            
            # Accumulate F1 metrics if available
            if 'soft_f1' in self.last_loss_components:
                self.accumulated_soft_f1 += self.last_loss_components['soft_f1'] * batch_size
                self.accumulated_soft_precision += self.last_loss_components['soft_precision'] * batch_size
                self.accumulated_soft_recall += self.last_loss_components['soft_recall'] * batch_size
            
            # Accumulate FP/FN components if available
            # Multiply by batch_size to match how other losses are accumulated
            if 'fp_fn_balance_loss' in self.last_loss_components:
                self.accumulated_fp_fn_balance_loss += self.last_loss_components['fp_fn_balance_loss'] * batch_size
                if 'total_error_loss' in self.last_loss_components:
                    self.accumulated_total_error_loss += self.last_loss_components['total_error_loss'] * batch_size
                if 'distribution_matching_loss' in self.last_loss_components:
                    self.accumulated_distribution_matching_loss += self.last_loss_components['distribution_matching_loss'] * batch_size
                if 'fp_rate' in self.last_loss_components:
                    self.accumulated_fp_rate += self.last_loss_components['fp_rate'] * batch_size
                if 'fn_rate' in self.last_loss_components:
                    self.accumulated_fn_rate += self.last_loss_components['fn_rate'] * batch_size
            
            # Accumulate confidence penalty components if available
            if 'variance_penalty' in self.last_loss_components:
                self.accumulated_variance_penalty += self.last_loss_components['variance_penalty'] * batch_size
        
        # PHASE 3: Filter invalid targets for metrics
        original_targets_for_metrics = targets.clone()
        if targets.dim() > 1:
            targets_flat = targets.view(targets.shape[0])
        else:
            targets_flat = targets
        
        # Check for invalid targets (-1 from boundary handling)
        valid_mask = targets_flat >= 0
        
        # Prepare targets for metrics
        targets = self._prepare_targets(targets)
        
        # Classification metrics
        # Get probabilities from logits for metrics
        logits = predictions['logits']

        # Handle different classification modes
        if self.classification_mode == 'ternary':
            # Ternary classification
            # Logits shape: [batch_size, 3]

            # Filter out invalid samples for metrics
            if not valid_mask.all():
                num_invalid = (~valid_mask).sum().item()
                logger.debug(f"[LEAKAGE FIX] Filtering {num_invalid} invalid samples from training metrics")
                logits = logits[valid_mask]
                targets = targets[valid_mask]
                # Skip metrics if no valid samples
                if len(targets) == 0:
                    logger.warning("[LEAKAGE FIX] No valid samples for metrics in this batch")
                    return loss

            # Get probabilities using softmax for ternary
            probs = F.softmax(logits, dim=-1)  # Shape: [batch_size, 3]
            # Get predictions as argmax
            preds = torch.argmax(probs, dim=-1)  # Shape: [batch_size]

        else:
            # Binary classification (original logic)
            # Ensure logits are 1D
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            # Filter out invalid samples for metrics
            if not valid_mask.all():
                num_invalid = (~valid_mask).sum().item()
                logger.debug(f"[LEAKAGE FIX] Filtering {num_invalid} invalid samples from training metrics")
                logits = logits[valid_mask]
                targets = targets[valid_mask]
                # Skip metrics if no valid samples
                if len(targets) == 0:
                    logger.warning("[LEAKAGE FIX] No valid samples for metrics in this batch")
                    return loss

            # Only clamp extreme negatives to prevent underflow
            logits = torch.clamp_min(logits, -50.0)
            probs = torch.sigmoid(logits)  # Probability of positive class
            # Additional safety check
            probs = torch.nan_to_num(probs, nan=self.prediction_threshold)
            preds = (probs > self.prediction_threshold).long()
        
        # Enhanced debug logging every 10 steps
        if self.global_step % 10 == 0:
            total = len(preds)

            if self.classification_mode == 'ternary':
                # Ternary classification stats
                pred_holds = (preds == 0).sum().item()
                pred_buys = (preds == 1).sum().item()
                pred_sells = (preds == 2).sum().item()
                target_holds = (targets == 0).sum().item()
                target_buys = (targets == 1).sum().item()
                target_sells = (targets == 2).sum().item()
            else:
                # Binary classification stats
                pred_ones = preds.sum().item()
                pred_zeros = (1 - preds).sum().item()
                target_ones = targets.sum().item()
                target_zeros = (1 - targets).sum().item()
            
            # Logit statistics
            if self.classification_mode == 'ternary':
                # For ternary, flatten logits to get overall statistics
                logit_flat = logits.flatten()
                logit_mean = logit_flat.mean().item()
                logit_std = logit_flat.std().item()
                logit_min = logit_flat.min().item()
                logit_max = logit_flat.max().item()
                unique_logits = torch.unique(logit_flat).numel()
                # For ternary, probs is 2D, so flatten it
                unique_probs = torch.unique(torch.round(probs.flatten() * 1000) / 1000).numel()
            else:
                # For binary, logits is 1D
                logit_mean = logits.mean().item()
                logit_std = logits.std().item()
                logit_min = logits.min().item()
                logit_max = logits.max().item()
                unique_logits = torch.unique(logits).numel()
                unique_probs = torch.unique(torch.round(probs * 1000) / 1000).numel()  # Round to 3 decimals
            
            # Attention weights analysis (if first batch)
            attention_stats = {}
            if hasattr(self.model, 'prediction_head') and hasattr(self.model.prediction_head, '_last_attention_weights'):
                attn_weights = self.model.prediction_head._last_attention_weights
                if attn_weights is not None:
                    attention_stats = {
                        'mean': attn_weights.mean().item(),
                        'std': attn_weights.std().item(),
                        'max': attn_weights.max().item(),
                        'min': attn_weights.min().item()
                    }
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Step {self.global_step} - COMPREHENSIVE DEBUG")
            logger.info(f"{'='*80}")
            if self.classification_mode == 'ternary':
                logger.info(f"Predictions: Hold={pred_holds}/{total} ({100*pred_holds/total:.1f}%), "
                           f"Buy={pred_buys}/{total} ({100*pred_buys/total:.1f}%), "
                           f"Sell={pred_sells}/{total} ({100*pred_sells/total:.1f}%)")
                logger.info(f"Targets: Hold={target_holds}/{total} ({100*target_holds/total:.1f}%), "
                           f"Buy={target_buys}/{total} ({100*target_buys/total:.1f}%), "
                           f"Sell={target_sells}/{total} ({100*target_sells/total:.1f}%)")
            else:
                logger.info(f"Predictions: 1s={pred_ones}/{total} ({100*pred_ones/total:.1f}%), "
                           f"0s={pred_zeros}/{total} ({100*pred_zeros/total:.1f}%)")
                logger.info(f"Targets: 1s={target_ones}/{total} ({100*target_ones/total:.1f}%), "
                           f"0s={target_zeros}/{total} ({100*target_zeros/total:.1f}%)")
            logger.info(f"Logit stats: mean={logit_mean:.4f}, std={logit_std:.4f}, "
                       f"min={logit_min:.4f}, max={logit_max:.4f}")
            if self.classification_mode == 'ternary':
                # For ternary, show per-class probability stats
                logger.info(f"Prob stats per class: Hold={probs[:,0].mean():.4f}, "
                           f"Buy={probs[:,1].mean():.4f}, Sell={probs[:,2].mean():.4f}")
            else:
                logger.info(f"Prob stats: mean={probs.mean():.4f}, std={probs.std():.4f}, "
                           f"min={probs.min():.4f}, max={probs.max():.4f}")
            logger.info(f"Unique values: {unique_logits} unique logits, {unique_probs} unique probs")
            
            if unique_logits == 1:
                logger.warning("⚠️ WARNING: All logits are IDENTICAL! Model outputting constant value!")
            elif unique_logits < 5:
                logger.warning(f"⚠️ WARNING: Only {unique_logits} unique logit values - very low diversity!")
            
            if attention_stats:
                logger.info(f"Attention weights: mean={attention_stats['mean']:.4f}, "
                           f"std={attention_stats['std']:.4f}, "
                           f"min={attention_stats['min']:.4f}, max={attention_stats['max']:.4f}")
            
            # Sample some actual values
            logger.info("Sample predictions (first 5):")
            for i in range(min(5, total)):
                if self.classification_mode == 'ternary':
                    # For ternary, show all 3 class logits and probs
                    logger.info(f"  [{i}] logits={logits[i].tolist()}, probs={probs[i].tolist()}, "
                               f"pred={preds[i].item()}, target={targets[i].item()}")
                else:
                    # For binary, single logit and prob
                    logger.info(f"  [{i}] logit={logits[i].item():.4f}, prob={probs[i].item():.4f}, "
                               f"pred={preds[i].item()}, target={targets[i].item()}")
            
            # Debug: Check input features diversity
            if self.global_step % 50 == 0:
                # Get mid_price column if it exists
                mid_price_idx = None
                for idx, col in enumerate(self.data_config.feature_columns):
                    if col == 'mid_price':
                        mid_price_idx = idx
                        break
                
                if mid_price_idx is not None:
                    # Check mid_price values across batch
                    mid_prices = features[:, :5, mid_price_idx]  # First 5 timesteps
                    logger.info(f"\nMid-price analysis (first 5 timesteps):")
                    logger.info(f"  Sample 0: {mid_prices[0].cpu().numpy()}")
                    
                    # Only compare samples if batch has at least 2 samples
                    if features.shape[0] > 1:
                        logger.info(f"  Sample 1: {mid_prices[1].cpu().numpy()}")
                        logger.info(f"  Variance across batch: {mid_prices.var().item():.6f}")
                        logger.info(f"  All identical? {torch.allclose(mid_prices[0], mid_prices[1])}")
                    else:
                        logger.info(f"  Batch has only 1 sample - no comparison available")
                        logger.info(f"  Single sample variance: {mid_prices[0].var().item():.6f}")
            
            logger.info(f"{'='*80}\n")
            
            # Check if predictions match targets exactly (potential data leak)
            correct = (preds == targets).sum().item()
            logger.info(f"Step {self.global_step} - Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        
        # Update metrics using the new system
        self._update_metrics('train', preds, targets, probs)
        
        # Compute and log metrics efficiently - log at step level based on log_every_n_steps
        computed_metrics = self._compute_and_log_metrics('train', on_step=True, on_epoch=False, batch_size=batch_size)
        
        # Additionally log confusion matrix components
        # Note: computed_metrics already available from _compute_and_log_metrics
        if 'confusion_matrix' in computed_metrics:
            cm = computed_metrics['confusion_matrix']

            if self.classification_mode == 'ternary':
                # For ternary classification, confusion matrix is 3x3
                if cm.dim() == 2 and cm.shape == (3, 3):
                    # Log full confusion matrix for ternary
                    for i in range(3):
                        for j in range(3):
                            actual_class = ['hold', 'buy', 'sell'][i]
                            predicted_class = ['hold', 'buy', 'sell'][j]
                            self.log(f'train_cm_{actual_class}_as_{predicted_class}',
                                   cm[i, j].float(), on_step=True, on_epoch=False,
                                   batch_size=batch_size, sync_dist=False)

                    # Calculate and log per-class TP, FP, FN, TN
                    for class_idx, class_name in enumerate(['hold', 'buy', 'sell']):
                        tp = cm[class_idx, class_idx]
                        fp = cm[:, class_idx].sum() - tp
                        fn = cm[class_idx, :].sum() - tp
                        tn = cm.sum() - tp - fp - fn

                        self.log(f'train_{class_name}_tp', tp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                        self.log(f'train_{class_name}_fp', fp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                        self.log(f'train_{class_name}_fn', fn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                        self.log(f'train_{class_name}_tn', tn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
            else:
                # Binary classification
                # Ensure confusion matrix has correct shape
                if cm.dim() == 2 and cm.shape == (2, 2):
                    # For binary classification, confusion matrix is 2x2
                    tn, fp, fn, tp = cm.flatten()
                    self.log('train_true_positives', tp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_true_negatives', tn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_false_positives', fp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_false_negatives', fn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                elif cm.numel() == 4:
                    # Handle case where cm might already be flattened or have different shape
                    cm_flat = cm.view(-1)
                    tn, fp, fn, tp = cm_flat[0], cm_flat[1], cm_flat[2], cm_flat[3]
                    self.log('train_true_positives', tp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_true_negatives', tn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_false_positives', fp.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
                    self.log('train_false_negatives', fn.float(), on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=False)
        
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, on_step=True, on_epoch=False)
        
        # Return loss with predictions for callbacks
        return {
            'loss': loss, 
            'predictions': predictions, 
            'targets': targets,
            'loss_components': getattr(self, 'last_loss_components', None)
        }
    
    def _analyze_gradient_contributions(self) -> Dict[str, Dict[str, float]]:
        """Analyze how different loss components contribute to the total gradient.
        
        This method looks at the current gradients and the loss values to estimate
        the relative contribution of each component.
        
        Returns:
            Dictionary mapping component names to gradient info
        """
        if not hasattr(self, '_last_loss_dict') or self._last_loss_dict is None:
            return {}
            
        # Get total gradient norm using Lightning's method
        # This ensures we use the exact same value that's logged
        norms = grad_norm(self, norm_type=2)
        if norms and "grad_2.0_norm_total" in norms:
            total_grad_norm = norms["grad_2.0_norm_total"]
        else:
            # Fallback to pre-computed norm if available
            if hasattr(self, '_pre_clip_grad_norm') and self._pre_clip_grad_norm is not None:
                total_grad_norm = self._pre_clip_grad_norm
            else:
                # Last resort: manual computation
                total_grad_norm = 0.0
                for param in self.parameters():
                    if param.requires_grad and param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm == 0:
            return {}
            
        # Analyze loss components
        component_grads = {}
        
        # Get loss values from the dictionary
        def get_value(key):
            val = self._last_loss_dict.get(key, 0)
            return val.item() if isinstance(val, torch.Tensor) else val
            
        # Get the main components (these should add up to total)
        total_loss = get_value('total')
        weighted_primary = get_value('weighted_primary_loss')
        regularization_loss = get_value('regularization_loss')
        
        # Get the sub-components of regularization loss
        total_error_loss = get_value('total_error_loss')
        balance_loss = get_value('balance_loss')
        variance_penalty = get_value('variance_penalty')
        
        # Get the weights used
        total_error_weight = getattr(self.criterion, 'total_error_weight', 0.1)
        balance_weight = getattr(self.criterion, 'balance_weight', 0.5)
        variance_weight = getattr(self.criterion, 'variance_penalty_weight', 0.1)
        
        if total_loss > 0:
            # Main components that add up to total
            component_grads['total'] = {
                'norm': total_grad_norm,
                'value': total_loss,
                'contribution_pct': 100.0
            }
            
            # Primary loss contribution
            if weighted_primary > 0:
                primary_ratio = weighted_primary / total_loss
                component_grads['weighted_primary_loss'] = {
                    'norm': total_grad_norm * primary_ratio,
                    'value': weighted_primary,
                    'contribution_pct': primary_ratio * 100
                }
            
            # Regularization loss contribution (total)
            if regularization_loss > 0:
                reg_ratio = regularization_loss / total_loss
                component_grads['regularization_loss'] = {
                    'norm': total_grad_norm * reg_ratio,
                    'value': regularization_loss,
                    'contribution_pct': reg_ratio * 100
                }
                
                # Sub-components of regularization (these show breakdown of regularization_loss)
                # These are weighted contributions to regularization_loss
                weighted_total_error = total_error_weight * total_error_loss
                weighted_balance = balance_weight * balance_loss
                weighted_variance = variance_weight * variance_penalty
                
                if weighted_total_error > 0:
                    # This is the contribution of total_error to the total gradient
                    error_contribution = (weighted_total_error / total_loss)
                    component_grads['total_error_loss'] = {
                        'norm': total_grad_norm * error_contribution,
                        'value': total_error_loss,  # Show unweighted value
                        'weighted_value': weighted_total_error,
                        'contribution_pct': error_contribution * 100,
                        'weight': total_error_weight
                    }
                
                if weighted_balance > 0:
                    balance_contribution = (weighted_balance / total_loss)
                    component_grads['balance_loss'] = {
                        'norm': total_grad_norm * balance_contribution,
                        'value': balance_loss,  # Show unweighted value
                        'weighted_value': weighted_balance,
                        'contribution_pct': balance_contribution * 100,
                        'weight': balance_weight
                    }
                
                if weighted_variance > 0:
                    var_contribution = (weighted_variance / total_loss)
                    component_grads['variance_penalty'] = {
                        'norm': total_grad_norm * var_contribution,
                        'value': variance_penalty,  # Show unweighted value
                        'weighted_value': weighted_variance,
                        'contribution_pct': var_contribution * 100,
                        'weight': variance_weight
                    }
                
        return component_grads
    
    def on_after_backward(self) -> None:
        """Called after loss.backward() and before optimizers are stepped.
        
        This is the correct place to log gradient norms since gradients
        are available after backward but before optimizer step.
        """
        # Log gradient norm using Lightning's utility
        if self.trainer.global_step > 0:  # Skip first step to avoid issues
            norms = grad_norm(self, norm_type=2)
            if norms and "grad_2.0_norm_total" in norms:
                self.log("grad_norm", norms["grad_2.0_norm_total"], 
                         on_step=True, on_epoch=False, prog_bar=True)
                # Store the pre-clipping gradient norm
                self._pre_clip_grad_norm = norms["grad_2.0_norm_total"]
                
                # DDP gradient synchronization diagnostic
                if self.trainer.global_step % 100 == 0 and self.trainer.world_size > 1:
                    # Check if gradients are synchronized properly across GPUs
                    # Get a sample parameter's gradient to verify DDP synchronization
                    params_list = list(self.parameters())
                    if len(params_list) > 0:
                        sample_param = params_list[0].grad
                        if sample_param is not None:
                            local_norm = sample_param.norm().item()
                            # After DDP sync, this should be approximately the same on all GPUs
                            logger.info(f"[GPU {self.trainer.global_rank}/{self.trainer.world_size}] "
                                       f"Sample param grad norm: {local_norm:.6f}, "
                                       f"Total grad norm: {self._pre_clip_grad_norm:.4f}")
                            
                            # Also log the shape for debugging
                            logger.debug(f"[GPU {self.trainer.global_rank}] "
                                        f"Sample param shape: {params_list[0].shape}, "
                                        f"Grad shape: {sample_param.shape}")
                
        # Track component gradients BEFORE clipping if enabled
        if (hasattr(self, 'track_component_gradients') and self.track_component_gradients):
            # Compute component gradients when:
            # 1. Every N steps as configured
            # 2. When we're about to display metrics (check if next batch will trigger display)
            next_batch_will_display = ((self.batch_count + 1) % self.display_interval == 0)
            should_compute = (self.global_step % self.component_gradient_freq == 0 or 
                            next_batch_will_display)
            
            if should_compute:
                # Compute gradient contribution of each loss component
                # We'll analyze the gradients that have already been computed
                component_grads = self._analyze_gradient_contributions()
                
                if component_grads:
                    # Log component gradients
                    for component_key, grad_info in component_grads.items():
                        self.log(f"grad_norm/{component_key}", grad_info['norm'],
                                on_step=True, on_epoch=False)
                    
                    # Store for display in metrics
                    self._last_component_gradients = {k: v['norm'] for k, v in component_grads.items()}
                    self._last_full_gradient_info = component_grads  # Store full info for display
                    self._gradient_info_step = self.global_step  # Track which step this is from
                    
                    # Debug logging
                    logger.debug(f"Component gradients analyzed at step {self.global_step}: {list(component_grads.keys())}")
    
    def on_before_optimizer_step(self, optimizer) -> None:
        """Called before optimizer step (and before gradient clipping).
        
        Note: This is actually called BEFORE gradient clipping, not after.
        We'll use it to detect when clipping will be applied.
        """
        # Store pre-step weight norms for monitoring
        if not hasattr(self, '_pre_step_weight_norms'):
            self._pre_step_weight_norms = {}
        
        # Sample weight norms from key layers before optimizer step
        with torch.no_grad():
            # Classification head weights
            if hasattr(self.model, 'prediction_head'):
                self._pre_step_weight_norms['prediction_head.fc'] = self.model.prediction_head.fc.weight.norm().item()
                if self.model.prediction_head.fc.bias is not None:
                    self._pre_step_weight_norms['prediction_head.fc_bias'] = self.model.prediction_head.fc.bias.norm().item()
                
                # Check attention weights
                self._pre_step_weight_norms['prediction_head.attention_query'] = self.model.prediction_head.attention_query.weight.norm().item()
            
            # First transformer layer
            if hasattr(self.model, 'encoder_layers') and len(self.model.encoder_layers) > 0:
                # FlashAttention uses qkv_proj instead of separate W_q, W_k, W_v
                self._pre_step_weight_norms['encoder_0.self_attn'] = self.model.encoder_layers[0].self_attn.qkv_proj.weight.norm().item()
        
        # Get current gradient norm
        norms = grad_norm(self, norm_type=2)
        if norms and "grad_2.0_norm_total" in norms:
            current_norm = norms["grad_2.0_norm_total"]
            
            # Check what the clip value is
            clip_val = self.trainer.gradient_clip_val
            
            # Log the expected clipped value
            if clip_val is not None and current_norm > clip_val:
                # Gradient will be clipped
                expected_clipped = clip_val
                self.log("grad_norm_clipped", expected_clipped, 
                         on_step=True, on_epoch=False, prog_bar=True)
                # Log a warning every 100 steps when large clipping occurs
                if self.global_step % 100 == 0 and current_norm > clip_val * 10:
                    logger.warning(f"Large gradient clipping at step {self.global_step}: {current_norm:.2f} -> {clip_val}")
            else:
                # No clipping needed
                self.log("grad_norm_clipped", current_norm, 
                         on_step=True, on_epoch=False, prog_bar=True)
    
    def optimizer_step(self, *args, **kwargs):
        """Override optimizer_step to monitor weight updates."""
        # Call parent optimizer step
        super().optimizer_step(*args, **kwargs)
        
        # After optimizer step, check if weights actually changed
        if hasattr(self, '_pre_step_weight_norms'):  # Changed to log every step
            with torch.no_grad():
                weight_changes = {}
                
                # Check classification head weights
                if hasattr(self.model, 'prediction_head'):
                    post_norm = self.model.prediction_head.fc.weight.norm().item()
                    pre_norm = self._pre_step_weight_norms.get('prediction_head.fc', post_norm)
                    weight_changes['fc_weight'] = abs(post_norm - pre_norm)
                    
                    if self.model.prediction_head.fc.bias is not None:
                        post_bias_norm = self.model.prediction_head.fc.bias.norm().item()
                        pre_bias_norm = self._pre_step_weight_norms.get('prediction_head.fc_bias', post_bias_norm)
                        weight_changes['fc_bias'] = abs(post_bias_norm - pre_bias_norm)
                    
                    # Check attention weights
                    post_attn_norm = self.model.prediction_head.attention_query.weight.norm().item()
                    pre_attn_norm = self._pre_step_weight_norms.get('prediction_head.attention_query', post_attn_norm)
                    weight_changes['attention_query'] = abs(post_attn_norm - pre_attn_norm)
                
                # Check transformer layer
                if hasattr(self.model, 'encoder_layers') and len(self.model.encoder_layers) > 0:
                    post_enc_norm = self.model.encoder_layers[0].self_attn.qkv_proj.weight.norm().item()
                    pre_enc_norm = self._pre_step_weight_norms.get('encoder_0.self_attn', post_enc_norm)
                    weight_changes['encoder_0_attn'] = abs(post_enc_norm - pre_enc_norm)
                
                # Log weight changes
                total_change = sum(weight_changes.values())
                
                # Warning if weights aren't changing
                if total_change < 1e-8:
                    if self.trainer.is_global_zero:  # Only log from rank 0
                        logger.warning(f"WARNING: Weights did not change at step {self.global_step}! "
                                     f"Changes: {weight_changes}")
                        logger.warning(f"Learning rate: {self.trainer.optimizers[0].param_groups[0]['lr']}")
                        logger.warning(f"Gradient norm: {self._pre_clip_grad_norm if hasattr(self, '_pre_clip_grad_norm') else 'N/A'}")
                else:
                    if self.trainer.is_global_zero:  # Only log from rank 0
                        logger.info(f"Step {self.global_step} - Weight changes: {weight_changes}, Total: {total_change:.6f}")
                
                # Log to tensorboard
                for name, change in weight_changes.items():
                    self.log(f'weight_change/{name}', change, on_step=True, on_epoch=False)
                self.log('weight_change/total', total_change, on_step=True, on_epoch=False)
    
    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        pass
    
    # Removed on_train_epoch_end - using step-based training only
        self.accumulated_loss = 0.0
        self.loss_count = 0
        self.accumulated_primary_loss = 0.0
        self.accumulated_distribution_loss = 0.0
        self.accumulated_soft_f1 = 0.0
        self.accumulated_soft_precision = 0.0
        self.accumulated_soft_recall = 0.0
        self._reset_train_metrics()
    
    def load_state_dict(self, state_dict, strict=True):
        """Override to allow loading checkpoints with mismatched shapes.
        
        This is useful when model architecture changes slightly but we still
        want to load most of the weights (e.g., when max_sequence_length changes).
        
        Special handling for positional encoding when sequence length changes.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce that the keys match
            
        Returns:
            When strict=False, returns (missing_keys, unexpected_keys) tuple
            When strict=True, returns None
        """
        # Initialize tracking for padded parameters
        self._padded_params = {}
        
        # Get checkpoint config settings
        checkpoint_config = self.checkpoint_config if hasattr(self, 'checkpoint_config') else None
        if checkpoint_config is None:
            checkpoint_config = {}
        
        allow_shape_mismatch = checkpoint_config.get('allow_shape_mismatch', False)
        adapt_positional = checkpoint_config.get('adapt_positional_encoding', True)
        
        # If strict=True and shape mismatch not allowed, use normal loading
        if strict and not allow_shape_mismatch:
            return super().load_state_dict(state_dict, strict=True)
        
        # Get current model state
        model_state = self.state_dict()
        
        # Filter out keys with shape mismatches
        filtered_state_dict = {}
        ignored_keys = []
        shape_mismatches = []
        missing_in_checkpoint = []
        adapted_keys = []
        
        # Check all keys in checkpoint
        for key, value in state_dict.items():
            if key in model_state:
                if value.shape == model_state[key].shape:
                    filtered_state_dict[key] = value
                else:
                    # Special handling for positional encoding if enabled
                    if adapt_positional and ('pos_encoder.pe' in key or 'positional_encoding' in key):
                        adapted_pe = self._adapt_positional_encoding(
                            key, value, model_state[key]
                        )
                        if adapted_pe is not None:
                            filtered_state_dict[key] = adapted_pe
                            adapted_keys.append(key)
                            logger.info(f"Adapted positional encoding {key}: {value.shape} -> {adapted_pe.shape}")
                        else:
                            ignored_keys.append(key)
                            shape_mismatches.append(
                                f"{key}: checkpoint shape {value.shape} vs model shape {model_state[key].shape} (adaptation failed)"
                            )
                    elif allow_shape_mismatch and 'embedding.input_projection.weight' in key:
                        # Special handling for input projection weight when adding new features
                        # If checkpoint has fewer input features, pad with initialized values
                        checkpoint_shape = value.shape
                        model_shape = model_state[key].shape
                        
                        if len(checkpoint_shape) == 2 and len(model_shape) == 2:
                            # Check if we're adding features (checkpoint has fewer input dims)
                            if checkpoint_shape[0] == model_shape[0] and checkpoint_shape[1] < model_shape[1]:
                                # Pad the weight matrix for new input features
                                import torch
                                import torch.nn.init as init
                                padded_weight = torch.zeros(model_shape, dtype=value.dtype, device=value.device)
                                # Copy existing weights
                                padded_weight[:, :checkpoint_shape[1]] = value
                                # Initialize new feature weights with small random values
                                init.xavier_uniform_(padded_weight[:, checkpoint_shape[1]:])
                                filtered_state_dict[key] = padded_weight
                                adapted_keys.append(key)
                                # Track this parameter was padded for optimizer state handling
                                self._padded_params[key] = {
                                    'old_shape': checkpoint_shape,
                                    'new_shape': model_shape
                                }
                                logger.info(f"Padded input projection weights for new features: {checkpoint_shape} -> {model_shape}")
                            else:
                                # Other shape mismatches we can't handle
                                ignored_keys.append(key)
                                shape_mismatches.append(
                                    f"{key}: checkpoint shape {value.shape} vs model shape {model_state[key].shape}"
                                )
                        else:
                            ignored_keys.append(key)
                            shape_mismatches.append(
                                f"{key}: checkpoint shape {value.shape} vs model shape {model_state[key].shape}"
                            )
                    else:
                        ignored_keys.append(key)
                        shape_mismatches.append(
                            f"{key}: checkpoint shape {value.shape} vs model shape {model_state[key].shape}"
                        )
            else:
                ignored_keys.append(key)
        
        # Check for keys in model but not in checkpoint
        for key in model_state:
            if key not in state_dict:
                missing_in_checkpoint.append(key)
        
        # Log what we're doing if we're modifying the state dict
        if ignored_keys or missing_in_checkpoint or adapted_keys:
            logger.warning(f"Loading checkpoint with modifications:")
            logger.warning(f"  - Total keys in checkpoint: {len(state_dict)}")
            logger.warning(f"  - Keys to be loaded: {len(filtered_state_dict)}")
            logger.warning(f"  - Keys ignored: {len(ignored_keys)}")
            logger.warning(f"  - Keys adapted: {len(adapted_keys)}")
            logger.warning(f"  - Keys missing in checkpoint: {len(missing_in_checkpoint)}")
        
        if adapted_keys:
            logger.info("Successfully adapted keys:")
            for key in adapted_keys:
                logger.info(f"  - {key}")
        
        # Only show detailed warnings if configured
        warn_on_missing = checkpoint_config.get('warn_on_missing_keys', True)
        warn_on_unexpected = checkpoint_config.get('warn_on_unexpected_keys', True)
        
        if shape_mismatches and warn_on_unexpected:
            logger.warning("Shape mismatches found:")
            for mismatch in shape_mismatches[:10]:  # Limit to first 10
                logger.warning(f"  - {mismatch}")
            if len(shape_mismatches) > 10:
                logger.warning(f"  ... and {len(shape_mismatches) - 10} more")
        
        if missing_in_checkpoint and warn_on_missing:
            logger.warning(f"Keys missing in checkpoint (will use random init): {missing_in_checkpoint[:10]}")
            if len(missing_in_checkpoint) > 10:
                logger.warning(f"  ... and {len(missing_in_checkpoint) - 10} more")
        
        # Load the filtered state dict with the requested strictness
        result = super().load_state_dict(filtered_state_dict, strict=strict)
        
        # Handle return value based on strict parameter
        if not strict:
            # When strict=False, PyTorch returns a NamedTuple with missing_keys and unexpected_keys
            # We need to augment it with our filtered keys
            if result:
                # Combine PyTorch's missing keys with ours
                all_missing = list(result.missing_keys) if hasattr(result, 'missing_keys') else []
                all_missing.extend(missing_in_checkpoint)
                
                # Combine PyTorch's unexpected keys with our ignored keys
                all_unexpected = list(result.unexpected_keys) if hasattr(result, 'unexpected_keys') else []
                all_unexpected.extend(ignored_keys)
                
                # Return as tuple (PyTorch convention for strict=False)
                return all_missing, all_unexpected
            else:
                # If no result from parent, return our keys
                return missing_in_checkpoint, ignored_keys
        
        # When strict=True, return None (PyTorch convention)
        logger.info("Successfully loaded checkpoint")
        return None
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Override learning rate from checkpoint with config value and optionally clear optimizer state.
        
        This ensures that when resuming training, we use the learning rate
        from config.py or config-dev.py instead of the one saved in checkpoint.
        Can also clear optimizer state completely if clear_optimizer_on_resume is True.
        """
        # Let the parent class handle the checkpoint loading first
        super().on_load_checkpoint(checkpoint)
        
        # Check if we should clear optimizer state
        if self.clear_optimizer_on_resume:
            if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
                logger.info("=" * 60)
                logger.info("CLEARING OPTIMIZER STATE FROM CHECKPOINT")
                logger.info("=" * 60)
                logger.info("This will reset:")
                logger.info("  - Adam momentum buffers (beta1)")
                logger.info("  - Adam second moment estimates (beta2)")
                logger.info("  - Any other optimizer internal states")
                logger.info("Model weights will be preserved from checkpoint")
                logger.info("=" * 60)
                
                # Clear the optimizer states
                checkpoint['optimizer_states'] = []
                
                logger.info("Optimizer states cleared - will start with fresh optimizer")
            else:
                logger.info("No optimizer states in checkpoint to clear")
        else:
            # Log the learning rate from checkpoint for comparison
            if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
                for opt_idx, opt_state in enumerate(checkpoint['optimizer_states']):
                    if 'param_groups' in opt_state:
                        for pg_idx, param_group in enumerate(opt_state['param_groups']):
                            if 'lr' in param_group:
                                logger.info(f"Checkpoint learning rate for optimizer {opt_idx}, param_group {pg_idx}: {param_group['lr']}")
        
        # Override with config learning rate
        logger.info(f"Overriding learning rate with config value: {self.learning_rate}")
        
        # The actual override will happen in configure_optimizers when the optimizer is created
        # We just need to ensure self.learning_rate has the correct value from config
        self.learning_rate = self.optimization_config.learning_rate
        
        # Log hparams to ensure they're updated
        if hasattr(self, 'hparams') and 'learning_rate' in self.hparams:
            self.hparams['learning_rate'] = self.learning_rate
            logger.info(f"Updated hparams learning_rate to: {self.learning_rate}")
    
    def _adapt_positional_encoding(self, key, checkpoint_pe, model_pe):
        """Adapt positional encoding when sequence length changes.
        
        Args:
            key: The parameter key
            checkpoint_pe: Positional encoding from checkpoint
            model_pe: Expected positional encoding shape in current model
        
        Returns:
            Adapted positional encoding tensor or None if adaptation fails
        """
        try:
            checkpoint_shape = checkpoint_pe.shape
            model_shape = model_pe.shape
            
            logger.info(f"Adapting positional encoding for {key}:")
            logger.info(f"  Checkpoint shape: {checkpoint_shape}")
            logger.info(f"  Model shape: {model_shape}")
            
            # Handle shape [max_len, 1, d_model] or [max_len, d_model]
            if len(checkpoint_shape) == 3 and len(model_shape) == 3:
                checkpoint_len, checkpoint_batch, checkpoint_dim = checkpoint_shape
                model_len, model_batch, model_dim = model_shape
                
                # Check if only sequence length changed
                if checkpoint_batch == model_batch and checkpoint_dim == model_dim:
                    if model_len <= checkpoint_len:
                        # Truncate: take first model_len positions
                        adapted_pe = checkpoint_pe[:model_len, :, :]
                        logger.info(f"  Truncated positional encoding from {checkpoint_len} to {model_len}")
                        return adapted_pe
                    else:
                        # Extend: need to generate new positions
                        # Get the original positional encoding parameters
                        device = checkpoint_pe.device
                        dtype = checkpoint_pe.dtype
                        
                        # Generate new positional encoding for the full length
                        adapted_pe = self._generate_positional_encoding(
                            model_len, model_dim, device, dtype
                        )
                        
                        # Copy existing encodings for positions that overlap
                        adapted_pe[:checkpoint_len, :, :] = checkpoint_pe
                        
                        logger.info(f"  Extended positional encoding from {checkpoint_len} to {model_len}")
                        return adapted_pe
            
            elif len(checkpoint_shape) == 2 and len(model_shape) == 2:
                checkpoint_len, checkpoint_dim = checkpoint_shape
                model_len, model_dim = model_shape
                
                if checkpoint_dim == model_dim:
                    if model_len <= checkpoint_len:
                        # Truncate
                        adapted_pe = checkpoint_pe[:model_len, :]
                        logger.info(f"  Truncated 2D positional encoding from {checkpoint_len} to {model_len}")
                        return adapted_pe
                    else:
                        # Extend
                        device = checkpoint_pe.device
                        dtype = checkpoint_pe.dtype
                        
                        adapted_pe = self._generate_positional_encoding_2d(
                            model_len, model_dim, device, dtype
                        )
                        adapted_pe[:checkpoint_len, :] = checkpoint_pe
                        
                        logger.info(f"  Extended 2D positional encoding from {checkpoint_len} to {model_len}")
                        return adapted_pe
            
            logger.warning(f"  Cannot adapt positional encoding: incompatible shapes")
            return None
            
        except Exception as e:
            logger.error(f"Error adapting positional encoding for {key}: {e}")
            return None
    
    def _generate_positional_encoding(self, max_len, d_model, device, dtype):
        """Generate sinusoidal positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            device: Device to create tensor on
            dtype: Data type
        
        Returns:
            Positional encoding tensor of shape [max_len, 1, d_model]
        """
        import math
        
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add batch dimension: [max_len, 1, d_model]
        
        return pe
    
    def _generate_positional_encoding_2d(self, max_len, d_model, device, dtype):
        """Generate 2D sinusoidal positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            device: Device to create tensor on
            dtype: Data type
        
        Returns:
            Positional encoding tensor of shape [max_len, d_model]
        """
        import math
        
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        optimizer = self._create_optimizer()
        
        # Create scheduler
        scheduler_config = self._create_scheduler(optimizer)
        
        if scheduler_config:
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return {"optimizer": optimizer}
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer directly."""
        optimizer_type = getattr(self.optimization_config, 'optimizer', 'adamw').lower()
        
        # Store base optimizer config for later adjustment
        self.base_lr = self.learning_rate
        self.base_weight_decay = self.weight_decay
        self.base_betas = (self.optimization_config.adam_beta1, self.optimization_config.adam_beta2)
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.optimization_config.adam_beta1, self.optimization_config.adam_beta2),
                eps=self.optimization_config.adam_eps
            )
            # Store initial lr for each param group
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
            return optimizer
        elif optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(self.optimization_config.adam_beta1, self.optimization_config.adam_beta2),
                eps=self.optimization_config.adam_eps
            )
        elif optimizer_type == 'lamb':
            from torch_optimizer import Lamb
            optimizer = Lamb(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(self.optimization_config.adam_beta1, self.optimization_config.adam_beta2),
                eps=self.optimization_config.adam_eps
                # Note: torch_optimizer.Lamb has bias_correction enabled by default
            )
            # Store initial lr for each param group
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
            return optimizer
        elif optimizer_type == 'sgd':
            # Get SGD-specific parameters with defaults
            momentum = getattr(self.optimization_config, 'sgd_momentum', 0.9)
            nesterov = getattr(self.optimization_config, 'sgd_nesterov', True)
            dampening = getattr(self.optimization_config, 'sgd_dampening', 0)
            
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=self.weight_decay,
                dampening=dampening,
                nesterov=nesterov
            )
            # Store initial lr for scheduler compatibility and ensure SGD params are in groups
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr']
                # Explicitly set SGD-specific parameters in param_groups
                # This prevents KeyError when optimizer.step() is called
                if 'momentum' not in group:
                    group['momentum'] = momentum
                if 'dampening' not in group:
                    group['dampening'] = dampening
                if 'nesterov' not in group:
                    group['nesterov'] = nesterov
            return optimizer
        else:
            # Default to AdamW
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Dict[str, Any]]:
        """Create learning rate scheduler directly."""
        scheduler_type = self.optimization_config.scheduler.lower()
        
        if scheduler_type == 'cosine':
            # Use max_steps for step-based training
            max_steps = self.config.training.max_steps if hasattr(self.config.training, 'max_steps') else 10000
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps,
                eta_min=self.config.optimization.cosine_eta_min
            )
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        elif scheduler_type == 'onecycle':
            # Use max_steps for step-based training
            max_steps = self.config.training.max_steps if hasattr(self.config.training, 'max_steps') else 10000
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=max_steps
            )
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        elif scheduler_type == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.config.optimization.exponential_gamma
            )
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        elif scheduler_type == 'none':
            return None
        else:
            # Default to cosine with step-based
            max_steps = self.config.training.max_steps if hasattr(self.config.training, 'max_steps') else 10000
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps
            )
            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
    
    def _get_portfolio_id_from_batch(self, batch: Dict[str, torch.Tensor]) -> str:
        """Extract portfolio ID from batch based on predict_start and predict_end values.
        
        Args:
            batch: Batch dictionary containing features
            
        Returns:
            Portfolio identifier string (e.g., "BTC-BN-3600-3620")
        """
        features = batch['features']
        
        # Assuming predict_start and predict_end are the last two features
        # Get the first sample in the batch for portfolio identification
        if features.dim() == 3:  # [batch, seq_len, features]
            sample_features = features[0, 0, :]  # First sample, first timestep
        else:
            sample_features = features[0, :]  # First sample
        
        # Extract predict_start and predict_end (last two features)
        predict_start = int(sample_features[-2].item())
        predict_end = int(sample_features[-1].item())
        
        # Try to find matching profile
        for profile in self.profiles:
            if hasattr(profile, 'predict_start') and hasattr(profile, 'predict_end'):
                if profile.predict_start == predict_start and profile.predict_end == predict_end:
                    # Return profile name if available
                    if hasattr(profile, 'name'):
                        return profile.name
        
        # Generate portfolio ID if no exact match found
        return f"portfolio-{predict_start}-{predict_end}"
    
    def _adjust_optimizer_for_portfolio(self, batch: Dict[str, torch.Tensor]):
        """Adjust optimizer parameters based on the current portfolio.
        
        Args:
            batch: Current batch to identify portfolio from
        """
        # Get portfolio ID from batch
        portfolio_id = self._get_portfolio_id_from_batch(batch)
        
        # Skip if already adjusted for this portfolio
        if portfolio_id == self.current_portfolio:
            return
        
        # Find matching profile configuration
        profile_config = None
        for profile in self.profiles:
            if hasattr(profile, 'name') and profile.name == portfolio_id:
                profile_config = profile
                break
        
        if profile_config is None:
            # Try to match by predict_start/predict_end
            features = batch['features']
            if features.dim() == 3:
                sample_features = features[0, 0, :]
            else:
                sample_features = features[0, :]
            predict_start = int(sample_features[-2].item())
            
            for profile in self.profiles:
                if hasattr(profile, 'predict_start') and profile.predict_start == predict_start:
                    profile_config = profile
                    break
        
        if profile_config and self.trainer and hasattr(self.trainer, 'optimizers'):
            optimizer = self.trainer.optimizers[0] if self.trainer.optimizers else None
            
            # Check for both AdamW and LAMB optimizers
            from torch_optimizer import Lamb
            if optimizer and (isinstance(optimizer, torch.optim.AdamW) or isinstance(optimizer, Lamb)):
                # Get portfolio-specific settings
                beta1 = getattr(profile_config, 'adam_beta1', self.base_betas[0])
                beta2 = getattr(profile_config, 'adam_beta2', self.base_betas[1])
                lr_multiplier = getattr(profile_config, 'learning_rate_multiplier', 1.0)
                weight_decay = getattr(profile_config, 'optimizer_weight_decay', self.base_weight_decay)
                
                # Update optimizer parameters
                for group in optimizer.param_groups:
                    # Update betas
                    group['betas'] = (beta1, beta2)
                    
                    # Adjust learning rate
                    if 'initial_lr' not in group:
                        group['initial_lr'] = self.base_lr
                    group['lr'] = group['initial_lr'] * lr_multiplier
                    
                    # Update weight decay
                    group['weight_decay'] = weight_decay
                
                # Log the change (only first few times to avoid spam)
                if not hasattr(self, '_portfolio_switch_count'):
                    self._portfolio_switch_count = {}
                if portfolio_id not in self._portfolio_switch_count:
                    self._portfolio_switch_count[portfolio_id] = 0
                self._portfolio_switch_count[portfolio_id] += 1
                
                if self._portfolio_switch_count[portfolio_id] <= 3:
                    logger.info(f"Switched to portfolio {portfolio_id}: "
                              f"β₁={beta1:.3f}, β₂={beta2:.3f}, "
                              f"lr_mult={lr_multiplier:.2f}, wd={weight_decay:.3f}")
                
                # Log metrics
                self.log(f'opt/portfolio', hash(portfolio_id) % 1000, on_step=True, on_epoch=False)
                self.log(f'opt/beta1', beta1, on_step=True, on_epoch=False)
                self.log(f'opt/beta2', beta2, on_step=True, on_epoch=False)
                self.log(f'opt/lr_multiplier', lr_multiplier, on_step=True, on_epoch=False)
        
        self.current_portfolio = portfolio_id
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Enhanced prediction step with FlashAttention optimization and streaming support.
        
        This method provides:
        - Optimized inference with FlashAttention (already in transformer)
        - Streaming prediction support for large datasets
        - Dynamic batch size optimization based on GPU memory
        - Comprehensive prediction post-processing
        
        Args:
            batch: Batch dictionary containing:
                - 'features': Input features [batch_size, seq_len, 307]
                - 'mask': Optional attention mask
                - 'timestamps': Optional timestamps for predictions
                - 'metadata': Optional metadata dictionary
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing:
                - 'predictions': Raw model predictions
                - 'confidence': Prediction confidence scores
                - 'trading_signals': Processed trading signals
                - 'metadata': Enhanced metadata with timing info
        """
        
        # Extract batch components
        features = batch['features']
        mask = batch.get('mask', None)
        timestamps = batch.get('timestamps', None)
        metadata = batch.get('metadata', {})
        
        # Batch size optimization is delegated to InferenceOptimizationCallback
        
        # Forward pass - Lightning handles mixed precision automatically
        predictions = self.forward(features, mask=mask)
        
        # Return raw predictions for simplicity
        results = {
            'predictions': predictions.get('logits', predictions.get('predictions')),
            'batch_idx': batch_idx
        }
        
        # Add targets if available (for evaluation during inference)
        if 'targets' in batch:
            results['targets'] = batch['targets']
        
        # Add metadata if available
        if 'timestamps' in batch:
            results['timestamps'] = batch['timestamps']
        if 'metadata' in batch:
            results['metadata'] = batch['metadata']
        
        return results
    
    
    # Batch size optimization is now handled by InferenceOptimizationCallback
    
    # Chunked prediction is now handled by InferenceOptimizationCallback and inference patterns
    
    # Chunk aggregation is now handled by inference patterns and callbacks
    
    # configure_prediction_optimization removed - handled by InferenceOptimizationCallback
    
    def on_train_start(self) -> None:
        """Called at the beginning of training."""
        # Initialize/reset batch tracking counters
        self.batch_count = 0
        self.last_display_batch = 0
        self.accumulated_samples = 0
        self.accumulated_loss = 0.0
        self.loss_count = 0
        self.accumulated_primary_loss = 0.0
        self.accumulated_distribution_loss = 0.0
        self.accumulated_soft_f1 = 0.0
        self.accumulated_soft_precision = 0.0
        self.accumulated_soft_recall = 0.0
        self.accumulated_fp_fn_balance_loss = 0.0
        self.accumulated_total_error_loss = 0.0
        self.accumulated_distribution_matching_loss = 0.0
        self.accumulated_fp_rate = 0.0
        self.accumulated_fn_rate = 0.0
        self.accumulated_variance_penalty = 0.0
        
        # Reset batch accumulators for ratios
        self.total_positives = 0.0
        self.total_predictions = 0.0
        self.total_samples = 0
        
        # Log hyperparameters to tensorboard/wandb
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
        
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Additional operations when saving checkpoint."""
        # Add custom metadata to checkpoint
        checkpoint['model_architecture'] = self.model.__class__.__name__
        checkpoint['training_epoch'] = self.current_epoch
        checkpoint['global_step'] = self.global_step
        
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Additional operations when loading checkpoint.
        
        Handles loading checkpoints that may be missing optimizer states for new parameters
        (e.g., category_scales added to OrderBookEmbedding).
        """
        # Handle optimizer state mismatch due to new parameters
        if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
            # Get the optimizer state dict
            optimizer_state = checkpoint['optimizer_states'][0]
            
            # Check if we have the correct number of parameter groups
            current_param_count = sum(1 for _ in self.parameters())
            
            # Log info about the checkpoint
            if 'state' in optimizer_state:
                checkpoint_param_count = len(optimizer_state['state'])
                logger.info(f"Checkpoint has optimizer state for {checkpoint_param_count} parameters")
                logger.info(f"Current model has {current_param_count} parameters")
                
                # If there's a mismatch, we need to handle it
                if checkpoint_param_count != current_param_count:
                    logger.warning(f"Parameter count mismatch: checkpoint has {checkpoint_param_count}, model has {current_param_count}")
                    logger.warning("This is likely due to new embedding parameters (category_scales) added to the model")
                    logger.warning("The optimizer will be reinitialized for the new parameters")
                    
                    # Clear the optimizer states to force reinitialization
                    # This allows PyTorch Lightning to create a fresh optimizer with the correct number of parameter groups
                    checkpoint['optimizer_states'] = []
                    logger.info("Cleared optimizer states from checkpoint - will start with fresh optimizer")
        
        # Handle model state dict loading with missing keys (e.g., new embedding parameters)
        if 'state_dict' in checkpoint:
            # Get checkpoint loading config
            checkpoint_config = self.checkpoint_config if hasattr(self, 'checkpoint_config') else None
            if checkpoint_config is None:
                checkpoint_config = {}
            
            strict_loading = checkpoint_config.get('strict_loading', True)
            allow_shape_mismatch = checkpoint_config.get('allow_shape_mismatch', False)
            
            try:
                # Try loading with configured strictness
                result = self.load_state_dict(checkpoint['state_dict'], strict=strict_loading)
                
                # If non-strict and result is returned, log the keys
                if not strict_loading and result:
                    missing_keys, unexpected_keys = result
                    if missing_keys:
                        logger.info(f"Missing keys (will be initialized): {missing_keys[:10]}")
                        if len(missing_keys) > 10:
                            logger.info(f"  ... and {len(missing_keys) - 10} more")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys (ignored): {unexpected_keys[:10]}")
                        if len(unexpected_keys) > 10:
                            logger.warning(f"  ... and {len(unexpected_keys) - 10} more")
                            
            except RuntimeError as e:
                # If strict loading failed, check if we should fall back
                if strict_loading and allow_shape_mismatch and ("Missing key(s)" in str(e) or "Unexpected key(s)" in str(e) or "size mismatch" in str(e)):
                    logger.warning(f"Strict loading failed: {e}")
                    logger.info("Falling back to non-strict loading as allow_shape_mismatch=True")
                    
                    # Try again with strict=False
                    result = self.load_state_dict(checkpoint['state_dict'], strict=False)
                    if result:
                        missing_keys, unexpected_keys = result
                        if missing_keys:
                            logger.info(f"Missing keys after fallback: {missing_keys[:10]}")
                        if unexpected_keys:
                            logger.warning(f"Unexpected keys after fallback: {unexpected_keys[:10]}")
                    
                    # Remove state_dict from checkpoint to prevent Lightning from trying to load it again
                    checkpoint.pop('state_dict', None)
                else:
                    # Re-raise if it's a different error or fallback not allowed
                    raise
        
        # Handle optimizer state padding for parameters that were padded during model loading
        if hasattr(self, '_padded_params') and self._padded_params and 'optimizer_states' in checkpoint:
            logger.info("Handling optimizer state for padded parameters...")
            
            for opt_idx, opt_state in enumerate(checkpoint['optimizer_states']):
                if 'state' in opt_state:
                    # Build a mapping from parameter to state index
                    param_to_state_idx = {}
                    idx = 0
                    for name, param in self.named_parameters():
                        param_to_state_idx[name] = idx
                        idx += 1
                    
                    # Check each padded parameter
                    for param_name, shape_info in self._padded_params.items():
                        if param_name in param_to_state_idx:
                            state_idx = param_to_state_idx[param_name]
                            
                            # Check if this state index exists in the optimizer state
                            if state_idx in opt_state['state']:
                                param_state = opt_state['state'][state_idx]
                                old_shape = shape_info['old_shape']
                                new_shape = shape_info['new_shape']
                                
                                # Pad exp_avg if it exists
                                if 'exp_avg' in param_state:
                                    if param_state['exp_avg'].shape == old_shape:
                                        import torch
                                        padded_avg = torch.zeros(new_shape, dtype=param_state['exp_avg'].dtype)
                                        padded_avg[:, :old_shape[1]] = param_state['exp_avg']
                                        param_state['exp_avg'] = padded_avg
                                        logger.info(f"  Padded exp_avg for {param_name}: {old_shape} -> {new_shape}")
                                
                                # Pad exp_avg_sq if it exists
                                if 'exp_avg_sq' in param_state:
                                    if param_state['exp_avg_sq'].shape == old_shape:
                                        import torch
                                        padded_avg_sq = torch.zeros(new_shape, dtype=param_state['exp_avg_sq'].dtype)
                                        padded_avg_sq[:, :old_shape[1]] = param_state['exp_avg_sq']
                                        param_state['exp_avg_sq'] = padded_avg_sq
                                        logger.info(f"  Padded exp_avg_sq for {param_name}: {old_shape} -> {new_shape}")
        
        # Log checkpoint info
        if 'training_epoch' in checkpoint:
            logger.info(f"Loaded checkpoint from epoch {checkpoint['training_epoch']}")
        if 'global_step' in checkpoint:
            logger.info(f"Resuming from global step {checkpoint['global_step']}")
        
        # Mark checkpoint resume for proper incremental scaler handling
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
            if hasattr(self.trainer.datamodule, 'train_dataset') and self.trainer.datamodule.train_dataset is not None:
                dataset = self.trainer.datamodule.train_dataset
                if hasattr(dataset, 'mark_checkpoint_resume'):
                    dataset.mark_checkpoint_resume()
                    logger.info("Marked checkpoint resume for incremental scaler")
    
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup model."""
        # Currently no setup needed - torch.compile removed as not implemented
        pass
    
    
    
    
    
    
    def configure_model(self) -> None:
        """Configure model optimizations.
        
        This method is called once at the beginning of fit/test/predict.
        It handles model-specific optimizations like memory formats.
        Note: Model compilation is handled in setup() to avoid conflicts.
        """
        # Enable channels last memory format if specified
        if getattr(self.model_config, 'use_channels_last', False):
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("Enabled channels_last memory format")
        
        # Enable gradient checkpointing if model supports it
        # Note: Since we don't have access to memory_optimization config here,
        # gradient checkpointing should be enabled through the model's own config
        # or via trainer callbacks
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            # Check if the model has gradient checkpointing enabled in its config
            if getattr(self.model, 'gradient_checkpointing', False):
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
    
    # Removed on_train_epoch_start - using step-based training only
    
    def on_fit_start(self) -> None:
        """Called at the beginning of fit."""
        # Configure model optimizations
        self.configure_model()
        
        # Log model summary
        if hasattr(self.logger, 'experiment'):
            # Calculate model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
            
            # Calculate model size in MB
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            logger.info(f"Model size: {model_size:.2f} MB")
    
    def on_train_start(self) -> None:
        """Called at the beginning of training, after optimizer is created.
        
        This is where we can override the learning rate if resuming from checkpoint.
        """
        # Override learning rate in all optimizers
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizers'):
            for opt_idx, optimizer in enumerate(self.trainer.optimizers):
                for pg_idx, param_group in enumerate(optimizer.param_groups):
                    old_lr = param_group.get('lr', 'N/A')
                    param_group['lr'] = self.learning_rate
                    logger.info(f"Updated optimizer {opt_idx}, param_group {pg_idx} learning rate: {old_lr} -> {self.learning_rate}")
        
        # Also update learning rate schedulers if they exist
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'lr_scheduler_configs'):
            for scheduler_config in self.trainer.lr_scheduler_configs:
                scheduler = scheduler_config.scheduler
                # Update base_lrs if it exists (most schedulers have this)
                if hasattr(scheduler, 'base_lrs'):
                    old_base_lrs = scheduler.base_lrs.copy() if hasattr(scheduler.base_lrs, 'copy') else list(scheduler.base_lrs)
                    scheduler.base_lrs = [self.learning_rate for _ in scheduler.base_lrs]
                    logger.info(f"Updated scheduler base_lrs: {old_base_lrs} -> {scheduler.base_lrs}")
                
                # For OneCycleLR, also update max_lrs
                if hasattr(scheduler, 'max_lrs'):
                    scheduler.max_lrs = [self.learning_rate for _ in scheduler.max_lrs]
                    logger.info(f"Updated OneCycleLR max_lrs to: {scheduler.max_lrs}")
    
    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Called at the beginning of training batch."""
        # Optional: Add custom batch preprocessing or logging
        pass
    
    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Called at the end of training batch."""
        # Store outputs for callbacks to access
        # This ensures callbacks can access the full outputs dict
        if isinstance(outputs, dict) and 'predictions' in outputs:
            self._last_batch_outputs = outputs
        else:
            self._last_batch_outputs = {'loss': outputs} if outputs is not None else None
    

    def get_progress_bar_dict(self) -> Dict[str, float]:
        """Get metrics for progress bar display."""
        items = super().get_progress_bar_dict()
        # Remove version number from progress bar
        items.pop("v_num", None)
        return items
    
    def _gather_global_metrics(self) -> Tuple[int, int, float, int, float, float, float, float, float, float, float]:
        """Gather metrics from all GPUs to compute global statistics.
        
        Returns:
            Tuple of (global_batch_count, global_samples, global_accumulated_loss, global_loss_count,
                     global_primary_loss, global_distribution_loss, global_fp_fn_balance_loss,
                     global_total_error_loss, global_distribution_matching_loss, global_fp_rate, global_fn_rate)
        """
        if torch.distributed.is_initialized() and self.trainer.world_size > 1:
            # Create tensors for all_reduce operations
            device = self.device
            local_batch_count = torch.tensor(self.batch_count, dtype=torch.int64, device=device)
            local_samples = torch.tensor(self.accumulated_samples, dtype=torch.int64, device=device)
            local_accumulated_loss = torch.tensor(self.accumulated_loss, dtype=torch.float32, device=device)
            local_loss_count = torch.tensor(self.loss_count, dtype=torch.int64, device=device)
            local_primary_loss = torch.tensor(self.accumulated_primary_loss, dtype=torch.float32, device=device)
            local_distribution_loss = torch.tensor(self.accumulated_distribution_loss, dtype=torch.float32, device=device)
            local_fp_fn_balance_loss = torch.tensor(self.accumulated_fp_fn_balance_loss, dtype=torch.float32, device=device)
            local_total_error_loss = torch.tensor(self.accumulated_total_error_loss, dtype=torch.float32, device=device)
            local_distribution_matching_loss = torch.tensor(self.accumulated_distribution_matching_loss, dtype=torch.float32, device=device)
            local_fp_rate = torch.tensor(self.accumulated_fp_rate, dtype=torch.float32, device=device)
            local_fn_rate = torch.tensor(self.accumulated_fn_rate, dtype=torch.float32, device=device)
            local_variance_penalty = torch.tensor(self.accumulated_variance_penalty, dtype=torch.float32, device=device)
            
            # Sum across all GPUs
            torch.distributed.all_reduce(local_batch_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_samples, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_accumulated_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_loss_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_primary_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_distribution_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_fp_fn_balance_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_total_error_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_distribution_matching_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_fp_rate, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_fn_rate, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_variance_penalty, op=torch.distributed.ReduceOp.SUM)
            
            return (
                local_batch_count.item(),
                local_samples.item(),
                local_accumulated_loss.item(),
                local_loss_count.item(),
                local_primary_loss.item(),
                local_distribution_loss.item(),
                local_fp_fn_balance_loss.item(),
                local_total_error_loss.item(),
                local_distribution_matching_loss.item(),
                local_fp_rate.item(),
                local_fn_rate.item(),
                local_variance_penalty.item()
            )
        else:
            # Single GPU or not distributed
            return (
                self.batch_count,
                self.accumulated_samples,
                self.accumulated_loss,
                self.loss_count,
                self.accumulated_primary_loss,
                self.accumulated_distribution_loss,
                self.accumulated_fp_fn_balance_loss,
                self.accumulated_total_error_loss,
                self.accumulated_distribution_matching_loss,
                self.accumulated_fp_rate,
                self.accumulated_fn_rate,
                self.accumulated_variance_penalty
            )
    
    def _display_accumulated_metrics(self, batch_range: tuple, samples: int, phase: str, 
                                   global_batch_count: int, global_samples: int,
                                   global_accumulated_loss: float, global_loss_count: int,
                                   global_primary_loss: float = 0.0,
                                   global_distribution_loss: float = 0.0,
                                   global_fp_fn_balance_loss: float = 0.0,
                                   global_total_error_loss: float = 0.0,
                                   global_distribution_matching_loss: float = 0.0,
                                   global_fp_rate: float = 0.0,
                                   global_fn_rate: float = 0.0,
                                   global_variance_penalty: float = 0.0) -> None:
        """Display accumulated metrics for a range of batches with global statistics.
        
        Args:
            batch_range: Tuple of (start_batch, end_batch) for rank 0
            samples: Number of samples processed on rank 0
            phase: 'train' or 'val'
            global_batch_count: Total batch count across all GPUs
            global_samples: Total samples across all GPUs
            global_accumulated_loss: Total loss sum across all GPUs
            global_loss_count: Total loss count across all GPUs
        """
        # Display information about the metrics period
        world_size = self.trainer.world_size if hasattr(self.trainer, 'world_size') else 1
        
        logger.info(f"\n{'='*60}")
        if world_size > 1:
            # For multi-GPU, show global statistics
            logger.info(f"Global Metrics Summary ({global_samples:,} samples across {world_size} GPUs)")
            logger.info(f"Total batches processed: {global_batch_count}")
        else:
            # For single GPU, show simple batch range
            logger.info(f"Metrics for batches {batch_range[0]}-{batch_range[1]} ({global_samples:,} samples)")
        logger.info(f"{'='*60}")
        
        # Display learning rate and scheduler info for training phase
        if phase == 'train' and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Display gradient norm if available
            if hasattr(self.trainer, 'logged_metrics') and 'grad_norm' in self.trainer.logged_metrics:
                grad_norm_value = self.trainer.logged_metrics['grad_norm']
                if torch.is_tensor(grad_norm_value):
                    grad_norm_value = grad_norm_value.item()
                logger.info(f"Gradient Norm (L2): {grad_norm_value:.4f}")
                
                # Display clipped gradient norm if available
                if 'grad_norm_clipped' in self.trainer.logged_metrics:
                    clipped_norm = self.trainer.logged_metrics['grad_norm_clipped']
                    if torch.is_tensor(clipped_norm):
                        clipped_norm = clipped_norm.item()
                    
                    # Check if clipping occurred
                    clip_val = self.trainer.gradient_clip_val
                    
                    # Debug: Always show the clip value
                    if clip_val is None:
                        logger.warning(f"WARNING: Gradient clipping is DISABLED (trainer.gradient_clip_val is None)")
                    else:
                        logger.info(f"Gradient clip value: {clip_val}")
                    
                    if clip_val and grad_norm_value > clip_val:
                        logger.info(f"Gradient Norm (Expected after clip): {clipped_norm:.4f} (will be clipped from {grad_norm_value:.4f})")
                    else:
                        logger.info(f"Gradient Norm (No clipping needed): {clipped_norm:.4f}")
            
            # Display scheduler type and progress
            if self.trainer.lr_scheduler_configs:
                scheduler_config = self.trainer.lr_scheduler_configs[0]
                scheduler = scheduler_config.scheduler
                scheduler_name = scheduler.__class__.__name__
                
                # Show scheduler-specific info
                if hasattr(scheduler, 'last_epoch'):
                    logger.info(f"Scheduler: {scheduler_name} (epoch {scheduler.last_epoch})")
                else:
                    logger.info(f"Scheduler: {scheduler_name}")
                    
                # For cosine annealing, show progress
                if scheduler_name == 'CosineAnnealingLR' and hasattr(scheduler, 'T_max'):
                    if scheduler.last_epoch == -1:
                        # Scheduler hasn't been stepped yet
                        logger.info(f"Cosine Progress: 0.0% (not started)")
                    else:
                        # Calculate actual progress
                        # Since last_epoch starts at -1 and increments, we use last_epoch directly
                        current_step = scheduler.last_epoch
                        progress = (current_step % scheduler.T_max) / scheduler.T_max * 100
                        logger.info(f"Cosine Progress: {progress:.1f}%")
                elif scheduler_name == 'OneCycleLR' and hasattr(scheduler, 'total_steps'):
                    if scheduler.last_epoch == -1:
                        logger.info(f"OneCycle Progress: 0.0% (not started)")
                    else:
                        # OneCycleLR uses last_epoch as step count
                        progress = scheduler.last_epoch / scheduler.total_steps * 100
                        logger.info(f"OneCycle Progress: {progress:.1f}%")
            else:
                logger.info(f"Scheduler: None (constant learning rate)")
        
        # Display average loss if available (using global values)
        if global_samples > 0:
            avg_loss = global_accumulated_loss / global_samples
            logger.info(f"Average Loss: {avg_loss:.4f}")
            
            # Display loss components if available
            if global_primary_loss > 0 or global_distribution_loss > 0:
                avg_primary = global_primary_loss / global_samples
                avg_distribution = global_distribution_loss / global_samples
                
                logger.info(f"Loss Components:")
                logger.info(f"  - Primary Loss (weighted): {avg_primary:.4f}")
                logger.info(f"  - Distribution Loss (weighted): {avg_distribution:.4f}")
                
                # Display FP/FN breakdown if available
                if global_fp_fn_balance_loss > 0 or global_total_error_loss > 0:
                    # Now these are accumulated with batch_size, so divide by samples like other losses
                    avg_fp_fn_balance = global_fp_fn_balance_loss / global_samples
                    avg_total_error = global_total_error_loss / global_samples
                    avg_fp_fn_diff = global_distribution_matching_loss / global_samples
                    avg_fp_rate = global_fp_rate / global_samples
                    avg_fn_rate = global_fn_rate / global_samples
                    
                    # Get actual weights from config
                    total_error_weight = getattr(self.binary_classification_config, 'total_error_weight', 1.0)
                    balance_weight = getattr(self.binary_classification_config, 'balance_weight', 0.5)
                    variance_penalty_weight = getattr(self.binary_classification_config, 'variance_penalty_weight', 0.1)
                    
                    logger.info(f"  - Regularization Loss: {avg_fp_fn_balance:.4f}")
                    logger.info(f"    - Total Error Loss: {avg_total_error:.4f} (weight: {total_error_weight})")
                    logger.info(f"    - Balance Loss (avg): {avg_fp_fn_diff:.4f} (weight: {balance_weight})")
                    
                    # Add confidence penalty if available
                    if global_variance_penalty > 0:
                        avg_variance_penalty = global_variance_penalty / global_samples
                        logger.info(f"    - Confidence Penalty: {avg_variance_penalty:.4f} (weight: {variance_penalty_weight})")
                    
                    logger.info(f"    - FP Expected per sample: {avg_fp_rate:.4f}")
                    logger.info(f"    - FN Expected per sample: {avg_fn_rate:.4f}")
                    
                    # Log distribution matching rates if available
                    if hasattr(self, 'last_loss_components') and self.last_loss_components:
                        if 'pred_rate' in self.last_loss_components:
                            logger.info(f"    - Model Positive Prediction Rate: {self.last_loss_components['pred_rate']:.4f}")
                        if 'target_rate' in self.last_loss_components:
                            logger.info(f"    - Target Positive Rate (actual): {self.last_loss_components['target_rate']:.4f}")
                            
                # Display component gradients if available
                if hasattr(self, '_last_component_gradients') and self._last_component_gradients:
                    logger.info(f"Gradient Contribution Analysis:")
                    
                    # Get the full gradient info (not just norms)
                    if hasattr(self, '_last_full_gradient_info'):
                        grad_info = self._last_full_gradient_info
                    else:
                        # Fallback to simple display
                        grad_info = {k: {'norm': v, 'contribution_pct': 0} for k, v in self._last_component_gradients.items()}
                    
                    # Display total gradient norm (should match the L2 norm above)
                    if 'total' in grad_info:
                        # Check if this is from the current step
                        is_current = (hasattr(self, '_gradient_info_step') and 
                                    self._gradient_info_step == self.global_step)
                        
                        if is_current or not hasattr(self, '_gradient_info_step'):
                            logger.info(f"  Total Gradient Norm: {grad_info['total']['norm']:.4f}")
                        else:
                            logger.info(f"  Total Gradient Norm: {grad_info['total']['norm']:.4f} (from step {self._gradient_info_step})")
                    
                    # Display main components (should add to 100%)
                    logger.info(f"  Main Components:")
                    if 'weighted_primary_loss' in grad_info:
                        info = grad_info['weighted_primary_loss']
                        logger.info(f"    - Primary Loss: {info['norm']:.4f} ({info['contribution_pct']:.1f}%)")
                    
                    if 'regularization_loss' in grad_info:
                        info = grad_info['regularization_loss']
                        logger.info(f"    - Regularization Loss: {info['norm']:.4f} ({info['contribution_pct']:.1f}%)")
                        
                        # Display regularization sub-components
                        reg_info = grad_info['regularization_loss']
                        reg_norm = reg_info['norm']
                        logger.info(f"      Regularization Breakdown (sum = {reg_info['contribution_pct']:.1f}%):")
                        
                        # Calculate percentages within regularization
                        reg_total_pct = 0.0
                        
                        if 'total_error_loss' in grad_info:
                            info = grad_info['total_error_loss']
                            # Show both: % of regularization and % of total
                            pct_of_reg = (info['norm'] / reg_norm * 100) if reg_norm > 0 else 0
                            reg_total_pct += info['contribution_pct']
                            logger.info(f"        - Total Error Loss: {info['norm']:.4f} ({info['contribution_pct']:.1f}% of total, {pct_of_reg:.1f}% of regularization)")
                        
                        if 'balance_loss' in grad_info:
                            info = grad_info['balance_loss']
                            pct_of_reg = (info['norm'] / reg_norm * 100) if reg_norm > 0 else 0
                            reg_total_pct += info['contribution_pct']
                            logger.info(f"        - Balance Loss: {info['norm']:.4f} ({info['contribution_pct']:.1f}% of total, {pct_of_reg:.1f}% of regularization)")
                        
                        if 'variance_penalty' in grad_info:
                            info = grad_info['variance_penalty']
                            pct_of_reg = (info['norm'] / reg_norm * 100) if reg_norm > 0 else 0
                            reg_total_pct += info['contribution_pct']
                            logger.info(f"        - Confidence Penalty: {info['norm']:.4f} ({info['contribution_pct']:.1f}% of total, {pct_of_reg:.1f}% of regularization)")
                        
                        # Verify the sum
                        logger.info(f"      (Regularization components sum: {reg_total_pct:.1f}% of total gradient)")
        
        # Display distribution matching info if enabled
        if phase == 'train' and self.binary_classification_config is not None:
            if getattr(self.binary_classification_config, 'match_batch_distribution', True):
                if hasattr(self.trainer, 'logged_metrics'):
                    # Display actual vs predicted distribution
                    if 'train/actual_positive_ratio' in self.trainer.logged_metrics:
                        actual_ratio = self.trainer.logged_metrics['train/actual_positive_ratio']
                        if torch.is_tensor(actual_ratio):
                            actual_ratio = actual_ratio.item()
                        logger.info(f"Actual Positive Ratio: {actual_ratio:.3f}")
                    
                    if 'train/predicted_positive_ratio' in self.trainer.logged_metrics:
                        pred_ratio = self.trainer.logged_metrics['train/predicted_positive_ratio']
                        if torch.is_tensor(pred_ratio):
                            pred_ratio = pred_ratio.item()
                        logger.info(f"Predicted Positive Ratio: {pred_ratio:.3f}")
                    
                    if 'train/distribution_error' in self.trainer.logged_metrics:
                        dist_error = self.trainer.logged_metrics['train/distribution_error']
                        if torch.is_tensor(dist_error):
                            dist_error = dist_error.item()
                        logger.info(f"Distribution Error: {dist_error:.4f}")
                    
        
        # Compute metrics for the specified phase
        metrics = self.metrics[phase]
        computed_metrics = metrics.compute()
        
        # Display metrics based on phase
        if phase == 'train':
            # Display core and cheap metrics only
            metric_order = ['accuracy', 'f1', 'auroc', 'precision', 'recall', 
                          'avg_precision', 'jaccard', 'mcc', 'fbeta', 'hamming']
        else:
            # Display all metrics for val/test
            metric_order = ['accuracy', 'f1', 'auroc', 'precision', 'recall', 
                          'avg_precision', 'jaccard', 'mcc', 'fbeta', 'hamming',
                          'cohen_kappa', 'calibration_error', 'hinge']
        
        # Display metrics in order
        for metric_name in metric_order:
            if metric_name in computed_metrics:
                value = computed_metrics[metric_name]
                # Handle tuple returns
                if isinstance(value, tuple):
                    value = value[0]
                    
                # Format metric name for display
                display_name = metric_name.replace('_', ' ').title()
                if metric_name == 'mcc':
                    display_name = 'Matthews Corr Coef'
                elif metric_name == 'fbeta':
                    display_name = 'F-Beta Score (β=2)'
                elif metric_name == 'auroc':
                    display_name = 'AUROC'
                elif metric_name == 'avg_precision':
                    display_name = 'Average Precision'
                    
                logger.info(f"{display_name}: {value:.4f}")
        
        # Handle precision/recall at fixed thresholds specially
        if phase != 'train':
            if 'precision_at_recall' in computed_metrics:
                value = computed_metrics['precision_at_recall']
                if isinstance(value, tuple):
                    value = value[0]
                logger.info(f"Precision at 80% Recall: {value:.4f}")
                
            if 'recall_at_precision' in computed_metrics:
                value = computed_metrics['recall_at_precision']
                if isinstance(value, tuple):
                    value = value[0]
                logger.info(f"Recall at 80% Precision: {value:.4f}")
        
        # Display confusion matrix
        if 'confusion_matrix' in computed_metrics:
            cm = computed_metrics['confusion_matrix']

            # Check if this is ternary (3x3) or binary (2x2) classification
            if cm.dim() == 2 and cm.shape == (3, 3):
                # Ternary classification - display full 3x3 matrix
                logger.info(f"\nConfusion Matrix (3x3):")
                logger.info(f"           Pred Hold  Pred Buy  Pred Sell")
                logger.info(f"True Hold:   {cm[0, 0]:6d}   {cm[0, 1]:6d}    {cm[0, 2]:6d}")
                logger.info(f"True Buy:    {cm[1, 0]:6d}   {cm[1, 1]:6d}    {cm[1, 2]:6d}")
                logger.info(f"True Sell:   {cm[2, 0]:6d}   {cm[2, 1]:6d}    {cm[2, 2]:6d}")

                # Calculate per-class accuracy
                total = cm.sum()
                if total > 0:
                    hold_acc = cm[0, 0] / (cm[0, :].sum() + 1e-10)
                    buy_acc = cm[1, 1] / (cm[1, :].sum() + 1e-10)
                    sell_acc = cm[2, 2] / (cm[2, :].sum() + 1e-10)

                    logger.info(f"\nPer-class Recall:")
                    logger.info(f"  Hold: {hold_acc:.2%}")
                    logger.info(f"  Buy:  {buy_acc:.2%}")
                    logger.info(f"  Sell: {sell_acc:.2%}")

                    # Class distribution
                    hold_rate = cm[0, :].sum() / total
                    buy_rate = cm[1, :].sum() / total
                    sell_rate = cm[2, :].sum() / total
                    logger.info(f"\nClass Distribution (Ground Truth):")
                    logger.info(f"  Hold: {hold_rate:.2%}")
                    logger.info(f"  Buy:  {buy_rate:.2%}")
                    logger.info(f"  Sell: {sell_rate:.2%}")

            elif cm.dim() == 2 and cm.shape == (2, 2):
                # Binary classification - original code
                tn, fp, fn, tp = cm.flatten()
                logger.info(f"\nConfusion Matrix (2x2):")
                logger.info(f"  True Positives: {tp}")
                logger.info(f"  True Negatives: {tn}")
                logger.info(f"  False Positives: {fp}")
                logger.info(f"  False Negatives: {fn}")

                # Calculate and log class distribution
                total = tn + fp + fn + tp
                if total > 0:
                    positive_rate = (fn + tp) / total
                    negative_rate = (tn + fp) / total
                    logger.info(f"\nClass Distribution:")
                    logger.info(f"  Positive class: {positive_rate:.2%}")
                    logger.info(f"  Negative class: {negative_rate:.2%}")
                    # Calculate additional derived metrics
                    if (tp + fp) > 0:
                        ppv = tp / (tp + fp)  # Positive Predictive Value
                        logger.info(f"\nDerived Metrics:")
                        logger.info(f"  Positive Predictive Value: {ppv:.4f}")
                    if (tn + fn) > 0:
                        npv = tn / (tn + fn)  # Negative Predictive Value
                        logger.info(f"  Negative Predictive Value: {npv:.4f}")
        
        logger.info(f"{'='*60}\n")
    
    def _reset_train_metrics(self) -> None:
        """Reset all training metrics for a new chunk."""
        self.train_metrics.reset()
    
    def _log_feature_information(self, features: torch.Tensor, batch_idx: int) -> None:
        """Log detailed feature information for the first training batch.
        
        Args:
            features: Feature tensor of shape [batch_size, seq_len, num_features]
            batch_idx: Index of the current batch
        """
        logger.info("="*80)
        logger.info("FEATURE INFORMATION - First Training Batch")
        logger.info("="*80)
        logger.info(f"Feature tensor shape: {features.shape}")
        logger.info(f"Number of features: {features.shape[-1]}")

        # Get feature names from datamodule
        feature_names = None
        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule:
            if hasattr(self.trainer.datamodule, 'train_dataset') and self.trainer.datamodule.train_dataset:
                dataset = self.trainer.datamodule.train_dataset
                if hasattr(dataset, 'feature_columns') and dataset.feature_columns:
                    feature_names = dataset.feature_columns
                elif hasattr(dataset, '_get_feature_names'):
                    feature_names = dataset._get_feature_names()

        # If we couldn't get feature names from dataset, infer from feature count
        if feature_names is None:
            # Standard features are all columns except timeMs and exchTimeMs
            # Plus predict_start and predict_end added during processing
            feature_names = [f"feature_{i}" for i in range(features.shape[-1]-2)]
            feature_names.extend(['predict_start', 'predict_end'])
            logger.info("Feature names not available from dataset, using generic names with predict_start/predict_end")

        logger.info(f"\nTotal number of features: {len(feature_names)}")

        # Print first sample from first sequence
        logger.info("\nFirst row of features from first sample:")
        logger.info("-"*80)

        first_sample = features[0, 0, :].cpu().numpy()  # First sample, most recent timestep (index 0)

        # Print feature names and values
        for i, (name, value) in enumerate(zip(feature_names, first_sample)):
            if i < 20:  # Print first 20 features
                logger.info(f"{i:3d}. {name:30s}: {value:15.6f}")
            elif i == 20:
                logger.info("... (showing first 20 features)")

        # Print last few features to show predict_start and predict_end
        if len(feature_names) > 20:
            logger.info("\nLast 20 features:")
            start_idx = max(len(feature_names) - 20, 0)
            for i in range(start_idx, len(feature_names)):
                logger.info(f"{i:3d}. {feature_names[i]:30s}: {first_sample[i]:15.6f}")

        # Specifically check for predict_start and predict_end
        if 'predict_start' in feature_names:
            idx = feature_names.index('predict_start')
            logger.info(f"\npredict_start found at index {idx}, value: {first_sample[idx]:.6f}")
            if abs(first_sample[idx] - 10000) < 1:
                logger.warning("predict_start appears to be unscaled (raw value ~10000)")
        if 'predict_end' in feature_names:
            idx = feature_names.index('predict_end')
            logger.info(f"predict_end found at index {idx}, value: {first_sample[idx]:.6f}")
            if abs(first_sample[idx] - 11000) < 1:
                logger.warning("predict_end appears to be unscaled (raw value ~11000)")

        # Print statistics
        logger.info("\nFeature statistics (first sample, most recent timestep):")
        logger.info(f"Min value: {first_sample.min():.6f}")
        logger.info(f"Max value: {first_sample.max():.6f}")
        logger.info(f"Mean value: {first_sample.mean():.6f}")
        logger.info(f"Std value: {first_sample.std():.6f}")
        logger.info(f"Non-zero features: {(first_sample != 0).sum()} / {len(first_sample)}")

        # Remove unnecessary barrier - Lightning handles epoch synchronization
        # This barrier could cause deadlock if any rank fails during training
        logger.info("="*80)

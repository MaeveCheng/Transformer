"""Factory for creating loss functions based on configuration."""

import logging
from typing import Optional

from .loss_registry import (
    create_loss, list_available_losses, get_loss_info, 
    validate_loss_config, _auto_register_losses
)
from .base_distribution_loss import BaseDistributionLoss

logger = logging.getLogger(__name__)

# Auto-register built-in losses
_auto_register_losses()


def create_loss_function(config, classification_mode='binary'):
    """
    Create loss function based on classification mode (binary or ternary).

    Args:
        config: Configuration object containing loss configuration
        classification_mode: 'binary' or 'ternary'

    Returns:
        Loss instance with all configured components

    Raises:
        ValueError: If configuration is invalid
    """
    if config is None:
        logger.warning(f"No classification config provided - using default loss for {classification_mode} mode")
        if classification_mode == 'ternary':
            from .ternary_loss import TernaryClassificationLoss
            return TernaryClassificationLoss()
        else:
            return create_loss('combined', bce_weight=1.0)

    # Handle ternary classification
    if classification_mode == 'ternary':
        from .ternary_loss import TernaryTradingLoss

        # Extract ternary-specific parameters
        ternary_params = {
            # Core loss parameters
            'loss_type': getattr(config, 'ternary_loss_type', 'cross_entropy'),
            'class_weights': getattr(config, 'class_weights', None),

            # Focal loss parameters
            'focal_alpha': getattr(config, 'ternary_focal_alpha', None),
            'focal_gamma': getattr(config, 'ternary_focal_gamma', 2.0),

            # Regularization
            'label_smoothing': getattr(config, 'label_smoothing', 0.0),
            'confidence_penalty_weight': getattr(config, 'confidence_penalty_weight', 0.0),
            'symmetric_penalty_weight': getattr(config, 'symmetric_penalty_weight', 0.0),

            # Distribution matching
            'target_distribution': getattr(config, 'target_distribution', None),
            'distribution_weight': getattr(config, 'distribution_weight', 0.0),

            # Trading decision thresholds (confidence-based, for inference/metrics)
            # Note: These are different from label_generation thresholds which are return-based
            'buy_threshold': getattr(config.decision_thresholds, 'buy_confidence', 0.6) if hasattr(config, 'decision_thresholds') else 0.6,
            'sell_threshold': getattr(config.decision_thresholds, 'sell_confidence', 0.6) if hasattr(config, 'decision_thresholds') else 0.6,
        }

        try:
            loss_fn = TernaryTradingLoss(**ternary_params)
        except Exception as e:
            logger.error(f"Failed to create ternary trading loss: {e}")
            logger.error(f"Parameters were: {ternary_params}")
            raise ValueError(f"Failed to create ternary trading loss: {e}")

        logger.info(f"Created ternary trading loss with parameters:")
        logger.info(f"  Loss type: {ternary_params['loss_type']}")
        if ternary_params['class_weights']:
            logger.info(f"  Class weights: {ternary_params['class_weights']}")
        return loss_fn

    # Original binary classification handling
    binary_config = config
    loss_weights = {
        'bce_weight': getattr(binary_config, 'bce_weight', 1.0),
        'focal_weight': getattr(binary_config, 'focal_weight', 0.0),
        'sigmoid_weight': getattr(binary_config, 'sigmoid_weight', 0.0),
    }
    
    # Common regularization parameters
    distribution_params = {
        'match_batch_distribution': binary_config.match_batch_distribution if hasattr(binary_config, 'match_batch_distribution') else True,
        'total_error_weight': binary_config.total_error_weight if hasattr(binary_config, 'total_error_weight') else 1.0,
        'balance_weight': binary_config.balance_weight if hasattr(binary_config, 'balance_weight') else 0.5,
        'variance_penalty_weight': binary_config.variance_penalty_weight if hasattr(binary_config, 'variance_penalty_weight') else 0.1,
        'target_prediction_ratio': binary_config.target_prediction_ratio if hasattr(binary_config, 'target_prediction_ratio') else 0.5,
        'balance_loss_type': binary_config.balance_loss_type if hasattr(binary_config, 'balance_loss_type') else 'squared',
        'balance_loss_k': binary_config.balance_loss_k if hasattr(binary_config, 'balance_loss_k') else 5.0,
        'balance_loss_gamma': binary_config.balance_loss_gamma if hasattr(binary_config, 'balance_loss_gamma') else 3.0,
        # Hard FP/FN parameters
        'use_hard_fp_fn': binary_config.use_hard_fp_fn if hasattr(binary_config, 'use_hard_fp_fn') else False,
        'fp_fn_temperature': binary_config.fp_fn_temperature if hasattr(binary_config, 'fp_fn_temperature') else 0.1,
        'fp_fn_threshold': binary_config.fp_fn_threshold if hasattr(binary_config, 'fp_fn_threshold') else 0.5,
        # Target rate parameters
        'target_fp_rate': binary_config.target_fp_rate if hasattr(binary_config, 'target_fp_rate') else None,
        'target_fn_rate': binary_config.target_fn_rate if hasattr(binary_config, 'target_fn_rate') else None,
        # Confidence penalty parameters
        'max_variance': binary_config.max_variance if hasattr(binary_config, 'max_variance') else 0.25,
    }
    
    # Primary loss-specific parameters
    loss_params = {
        # BCE parameters
        'bce_pos_weight': getattr(binary_config, 'bce_pos_weight', None),
        'label_smoothing': getattr(binary_config, 'label_smoothing', 0.0),
        # Focal parameters
        'focal_alpha': getattr(binary_config, 'focal_alpha', 0.25),
        'focal_gamma': getattr(binary_config, 'focal_gamma', 2.0),
        # Sigmoid parameters
        'sigmoid_k': getattr(binary_config, 'sigmoid_k', 5.0),
    }
    
    # Combine all parameters
    all_params = {**loss_weights, **distribution_params, **loss_params}
    
    # Use 'combined' loss type
    loss_type = 'combined'
    
    # Validate parameters
    validated_params = validate_loss_config(loss_type, all_params)
    
    try:
        # Create loss using registry
        loss_fn = create_loss(loss_type, **validated_params)
        
        # Log creation info
        logger.info(f"Created combined loss with weights:")
        logger.info(f"  BCE weight: {loss_weights['bce_weight']}")
        logger.info(f"  Focal weight: {loss_weights['focal_weight']}")
        logger.info(f"  Sigmoid weight: {loss_weights['sigmoid_weight']}")
        logger.info(f"  Distribution loss: {'enabled' if distribution_params['match_batch_distribution'] else 'disabled'}")
            
        # Log regularization configuration
        if distribution_params['match_batch_distribution']:
            logger.info(f"  Distribution Loss - enabled: True")
            logger.info(f"  Components:")
            logger.info(f"    - Total Error Weight: {distribution_params['total_error_weight']}")
            logger.info(f"    - Balance Weight: {distribution_params['balance_weight']}")
            logger.info(f"    - Balance Loss Type: {distribution_params['balance_loss_type']}")
            if distribution_params['balance_loss_type'] == 'exponential':
                logger.info(f"    - Balance Loss k: {distribution_params['balance_loss_k']}")
            elif distribution_params['balance_loss_type'] == 'focal':
                logger.info(f"    - Balance Loss gamma: {distribution_params['balance_loss_gamma']}")
            logger.info(f"    - Confidence Penalty Weight: {distribution_params['variance_penalty_weight']}")
            logger.info(f"    - Target Prediction Ratio: {distribution_params['target_prediction_ratio']}")
            
            # Log hard FP/FN settings
            if distribution_params['use_hard_fp_fn']:
                logger.info(f"    - Using Hard FP/FN: True")
                logger.info(f"    - Temperature: {distribution_params['fp_fn_temperature']}")
                logger.info(f"    - Threshold: {distribution_params['fp_fn_threshold']}")
            else:
                logger.info(f"    - Using Soft FP/FN: True")
            
            # Log target rate settings
            if distribution_params['target_fp_rate'] is not None or distribution_params['target_fn_rate'] is not None:
                logger.info(f"    - Target FP Rate: {distribution_params['target_fp_rate']}")
                logger.info(f"    - Target FN Rate: {distribution_params['target_fn_rate']}")
            else:
                logger.info(f"    - Target Rates: Using balanced formula based on class distribution")
        
        return loss_fn
        
    except ValueError as e:
        available = list_available_losses()
        raise ValueError(f"Failed to create combined loss. Available losses: {available}. Error: {e}")


def get_available_losses():
    """
    Get information about all available loss functions.
    
    Returns:
        Dictionary mapping loss names to their information
    """
    losses_info = {}
    for loss_name in list_available_losses():
        try:
            losses_info[loss_name] = get_loss_info(loss_name)
        except Exception as e:
            logger.warning(f"Failed to get info for loss '{loss_name}': {e}")
    
    return losses_info
"""Registry system for loss functions with automatic discovery and validation."""

import logging
from typing import Dict, Type, Any, Optional, List
from .base_distribution_loss import BaseDistributionLoss

logger = logging.getLogger(__name__)

# Global registry for loss functions
_LOSS_REGISTRY: Dict[str, Type[BaseDistributionLoss]] = {}


def register_loss(name: str):
    """
    Decorator to register a loss function.
    
    Usage:
        @register_loss('my_loss')
        class MyLoss(BaseDistributionLoss):
            ...
    
    Args:
        name: Name to register the loss under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Allow CombinedLoss which doesn't inherit from BaseDistributionLoss
        from torch import nn
        if not (issubclass(cls, (BaseDistributionLoss, nn.Module))):
            raise ValueError(f"Loss {cls.__name__} must inherit from BaseDistributionLoss or nn.Module")
        
        if name in _LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered to {_LOSS_REGISTRY[name].__name__}")
        
        _LOSS_REGISTRY[name] = cls
        logger.debug(f"Registered loss '{name}' -> {cls.__name__}")
        return cls
    
    return decorator


def get_loss_class(name: str) -> Type[BaseDistributionLoss]:
    """
    Get a loss class by name.
    
    Args:
        name: Name of the loss function
        
    Returns:
        Loss class
        
    Raises:
        ValueError: If loss name is not registered
    """
    if name not in _LOSS_REGISTRY:
        available = list(_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available losses: {available}")
    
    return _LOSS_REGISTRY[name]


def create_loss(name: str, **kwargs) -> BaseDistributionLoss:
    """
    Create a loss instance by name.
    
    Args:
        name: Name of the loss function
        **kwargs: Arguments to pass to the loss constructor
        
    Returns:
        Loss instance
    """
    loss_class = get_loss_class(name)
    return loss_class(**kwargs)


def list_available_losses() -> List[str]:
    """
    Get list of all registered loss names.
    
    Returns:
        List of loss names
    """
    return sorted(list(_LOSS_REGISTRY.keys()))


def get_loss_info(name: str) -> Dict[str, Any]:
    """
    Get information about a registered loss.
    
    Args:
        name: Name of the loss function
        
    Returns:
        Dictionary with loss information
    """
    loss_class = get_loss_class(name)
    
    # Extract init parameters from the loss class
    import inspect
    signature = inspect.signature(loss_class.__init__)
    params = {}
    
    for param_name, param in signature.parameters.items():
        if param_name in ['self', 'kwargs']:
            continue
        
        params[param_name] = {
            'default': param.default if param.default != inspect.Parameter.empty else None,
            'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
        }
    
    return {
        'name': name,
        'class': loss_class.__name__,
        'module': loss_class.__module__,
        'docstring': loss_class.__doc__,
        'parameters': params,
        'inherits_from': [base.__name__ for base in loss_class.__bases__]
    }


def validate_loss_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration for a loss function.
    
    Args:
        name: Name of the loss function
        config: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    loss_info = get_loss_info(name)
    valid_params = set(loss_info['parameters'].keys())
    
    # Add common BaseDistributionLoss parameters
    base_params = {
        'match_batch_distribution', 
        'primary_loss_weight',  # Keep for backward compatibility
        'total_error_weight', 
        'balance_weight',
        'variance_penalty_weight',
        'target_prediction_ratio',
        'balance_loss_type',
        'balance_loss_k',
        'balance_loss_gamma',
        'use_hard_fp_fn', 
        'fp_fn_temperature',
        'fp_fn_threshold', 
        'target_fp_rate', 
        'target_fn_rate',
        'max_variance',
        # Combined loss specific parameters
        'bce_weight',
        'focal_weight',
        'sigmoid_weight',
        'bce_pos_weight',
        'label_smoothing',
        'focal_alpha',
        'focal_gamma',
        'sigmoid_k'
    }
    valid_params.update(base_params)
    
    # Check for unknown parameters
    provided_params = set(config.keys())
    unknown_params = provided_params - valid_params
    
    if unknown_params:
        logger.warning(f"Unknown parameters for loss '{name}': {unknown_params}")
    
    # Filter to only valid parameters
    validated_config = {k: v for k, v in config.items() if k in valid_params}
    
    return validated_config


# Auto-register built-in losses when module is imported
def _auto_register_losses():
    """Auto-register built-in loss functions."""
    try:
        # Import loss modules to trigger registration
        from . import bce_distribution_loss
        from . import focal_distribution_loss
        from . import sigmoid_distribution_loss
        from . import combined_loss
    except ImportError as e:
        logger.warning(f"Failed to auto-register some losses: {e}")


# Note: We'll add the decorators to the loss classes in the next step
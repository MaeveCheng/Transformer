"""Logging utilities for PyTorch Lightning projects.

This module provides basic logging setup utilities.
For metrics logging, use PyTorch Lightning's built-in self.log() method.

MIGRATION GUIDE:
===============

The MetricLogger and ExperimentLogger classes have been removed in favor of
PyTorch Lightning's built-in logging features.

Migration patterns:

1. MetricLogger -> self.log()
   OLD:
   ```python
   self.metric_logger = MetricLogger(logger)
   self.metric_logger.log_metrics({'loss': loss}, step=step)
   ```
   NEW:
   ```python
   self.log('loss', loss, on_step=True, on_epoch=True)
   self.log_dict({'metric1': val1, 'metric2': val2})
   ```

2. ExperimentLogger -> Lightning Loggers + save_hyperparameters()
   OLD:
   ```python
   exp_logger = ExperimentLogger('experiment_name', log_dir)
   exp_logger.log_config(config)
   exp_logger.log_training_step(step, loss, metrics)
   ```
   NEW:
   ```python
   # In LightningModule.__init__:
   self.save_hyperparameters()
   
   # In training_step:
   self.log('train_loss', loss)
   self.log_dict(metrics)
   ```

3. Gradient norm logging is now handled by PyTorch Lightning:
   Set `track_grad_norm=2` in Trainer configuration.

4. For experiment tracking, use Lightning's built-in loggers:
   - TensorBoardLogger (default)
   - WandbLogger
   - MLFlowLogger
   - etc.

For more information, see:
https://lightning.ai/docs/pytorch/stable/extensions/logging.html
"""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Simple logging setup for scripts.
    
    This is a convenience function for scripts that need basic logging.
    For training scripts, use PyTorch Lightning's logging features.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
#!/usr/bin/env python3.10
"""
PyTorch Lightning-based evaluation utilities for order book transformer model.
Provides comprehensive evaluation using Lightning's Trainer.test() functionality.
"""

import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import logging
from pytorch_lightning.callbacks import Callback
from collections import defaultdict

# Import unified evaluation module
from utils.evaluation import (
    calculate_metrics,
    calculate_trading_metrics,
    generate_all_plots,
    print_evaluation_summary
)

# Import configurations
from config import Config

# Import data module
from data.unified_datamodule import UnifiedDataModule

# Import model and lightning module
from models.transformer import OrderBookTransformer
from lightning.module import OrderBookLightningModule

logger = logging.getLogger(__name__)


class EvaluationDataModule(UnifiedDataModule):
    """Extended DataModule for evaluation with additional features."""
    
    def __init__(
        self,
        config: Config,
        test_data_path: Optional[str] = None,
        batch_size_override: Optional[int] = None,
        num_workers_override: Optional[int] = None,
        **kwargs
    ):
        """Initialize evaluation data module.
        
        Args:
            config: Configuration object
            test_data_path: Optional path to test data (overrides config)
            batch_size_override: Optional batch size override for evaluation
            num_workers_override: Optional num workers override
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(config, **kwargs)
        
        # Override test data path if provided
        if test_data_path:
            self.test_data_path = test_data_path
        else:
            self.test_data_path = config.data.train_folder
            
        # Override batch size and num workers if provided
        if batch_size_override:
            self.batch_size = batch_size_override
        else:
            self.batch_size = config.data.batch_size
            
        if num_workers_override is not None:
            self.num_workers = num_workers_override
        else:
            self.num_workers = config.data.num_workers
    
    def test_dataloader(self):
        """Create test dataloader with evaluation-specific settings."""
        # Use parent's test_dataloader but with our overrides
        original_batch_size = self.config.data.batch_size
        original_num_workers = self.config.data.num_workers
        original_train_folder = self.config.data.train_folder
        
        try:
            # Temporarily override config values
            self.config.data.batch_size = self.batch_size
            self.config.data.num_workers = self.num_workers
            self.config.data.train_folder = self.test_data_path
            
            # Get dataloader from parent
            dataloader = super().test_dataloader()
            
            return dataloader
        finally:
            # Restore original values
            self.config.data.batch_size = original_batch_size
            self.config.data.num_workers = original_num_workers
            self.config.data.train_folder = original_train_folder


class PredictionCollectorCallback(Callback):
    """Callback to collect predictions during evaluation."""
    
    def __init__(self, store_features: bool = False):
        """Initialize prediction collector.
        
        Args:
            store_features: Whether to store input features along with predictions
        """
        super().__init__()
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.features = [] if store_features else None
        self.store_features = store_features
        self.batch_indices = []
        
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> None:
        """Collect predictions after each test batch."""
        # Extract predictions from outputs if available
        if outputs and isinstance(outputs, dict):
            # Check if outputs contain predictions from test step
            if 'predictions' in outputs:
                # Keep tensors on GPU until necessary
                self.predictions.extend(outputs['predictions'].cpu().numpy())
            if 'targets' in outputs:
                self.targets.extend(outputs['targets'].cpu().numpy())
            if 'probabilities' in outputs:
                self.probabilities.extend(outputs['probabilities'].cpu().numpy())
        else:
            # Compute predictions directly
            features = batch['features']
            targets = batch.get('targets')
            mask = batch.get('mask', None)
            
            with torch.no_grad():
                predictions = pl_module(features, mask=mask)
                
                # Binary classification metrics
                logits = predictions['logits']
                # Ensure logits are 1D
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                # Only transfer to CPU when necessary for storage
                self.predictions.extend(preds.cpu().numpy())
                self.probabilities.extend(probs.cpu().numpy())
                self.targets.extend(targets.cpu().numpy())
                
                if self.store_features:
                    self.features.append(features.cpu().numpy())
        
        self.batch_indices.append(batch_idx)
    
    def get_results(self) -> Dict[str, np.ndarray]:
        """Get collected results as numpy arrays."""
        results = {
            'predictions': np.array(self.predictions),
            'targets': np.array(self.targets),
            'probabilities': np.array(self.probabilities),
            'batch_indices': np.array(self.batch_indices)
        }
        
        if self.store_features and self.features:
            results['features'] = np.concatenate(self.features, axis=0)
            
        return results


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics using unified evaluation module."""
    
    @staticmethod
    def calculate_classification_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Calculate classification metrics using unified module.
        
        Args:
            predictions: Binary predictions
            targets: True labels
            probabilities: Prediction probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Use unified calculate_metrics function
        return calculate_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
            threshold=threshold
        )
    
    @staticmethod
    def calculate_trading_metrics(
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
        prices: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        min_confidence: float = 0.6
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics using unified module."""
        # prices parameter is kept for backward compatibility but not used
        return calculate_trading_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities,
            threshold=threshold,
            min_confidence=min_confidence
        )


class LightningEvaluator:
    """Main evaluator class using PyTorch Lightning."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
        enable_progress_bar: bool = True
    ):
        """Initialize Lightning evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Optional config override
            enable_progress_bar: Whether to show progress bar
        """
        self.checkpoint_path = checkpoint_path
        self.enable_progress_bar = enable_progress_bar
        
        # Load model and config
        self.model, self.config = self._load_model_and_config(checkpoint_path, config)
        
    def _load_model_and_config(
        self,
        checkpoint_path: str,
        config_override: Optional[Config] = None
    ) -> Tuple[OrderBookLightningModule, Config]:
        """Load model and configuration from checkpoint."""
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load the model directly to GPU
        model = OrderBookLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location='cuda'
        )
        
        # Get config
        if config_override:
            config = config_override
        else:
            # Try to load config from checkpoint directory
            config_path = Path(checkpoint_path).parent / "config.json"
            if config_path.exists():
                config = Config.from_json(config_path)
            else:
                # Extract from model hyperparameters
                checkpoint = torch.load(checkpoint_path, map_location='cuda')
                hparams = checkpoint.get('hyper_parameters', {})
                
                config = get_config()
                if 'model_config' in hparams:
                    config.model = hparams['model_config']
                if 'training_config' in hparams:
                    config.training = hparams['training_config']
        
        return model, config
    
    def evaluate(
        self,
        test_data_path: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        threshold: float = 0.5,
        min_confidence: float = 0.6,
        accelerator: str = "auto",
        devices: Union[int, List[int], str] = "auto",
        precision: str = "32-true",
        strategy: str = "auto"
    ) -> Dict[str, Any]:
        """Run evaluation using Lightning Trainer.test().
        
        Args:
            test_data_path: Optional path to test data
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            threshold: Classification threshold
            min_confidence: Minimum confidence for trading metrics
            accelerator: Lightning accelerator ("auto", "gpu", "cpu")
            devices: Device configuration (number of GPUs, list of GPU ids, or "auto")
            precision: Precision setting ("32", "16", "bf16", "16-true", "32-true", "64-true")
            strategy: Training strategy ("auto", "ddp", "dp", etc.)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create data module
        data_module = EvaluationDataModule(
            config=self.config,
            test_data_path=test_data_path,
            batch_size_override=batch_size,
            num_workers_override=num_workers,
            use_flash_optimizations=True,
            use_chunked=True
        )
        
        # Prepare data
        data_module.prepare_data()
        data_module.setup(stage='test')
        
        # Create prediction collector callback
        prediction_collector = PredictionCollectorCallback(store_features=False)
        
        # Create trainer for evaluation
        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
            enable_progress_bar=self.enable_progress_bar,
            enable_model_summary=False,
            logger=False,  # Disable logging during evaluation
            callbacks=[prediction_collector],
            enable_checkpointing=False
        )
        
        # Model caching is now handled by the prediction cache module
        # No need to manually enable or clear cache here
        
        # Run evaluation
        logger.info("Running model evaluation with Lightning Trainer.test()...")
        test_results = trainer.test(
            model=self.model,
            dataloaders=data_module.test_dataloader()
        )
        
        # Get predictions from callback
        results = prediction_collector.get_results()
        
        # Calculate metrics using unified evaluation module
        metrics = calculate_metrics(
            predictions=results['predictions'],
            targets=results['targets'],
            probabilities=results['probabilities'],
            threshold=threshold,
            include_trading_metrics=True,
            min_confidence=min_confidence
        )
        
        # Extract trading metrics for backward compatibility
        trading_metrics = metrics.pop('trading_metrics', {})
        
        # Combine results
        evaluation_results = {
            'checkpoint_path': self.checkpoint_path,
            'evaluation_metrics': metrics,
            'trading_metrics': trading_metrics,
            'lightning_test_results': test_results[0] if test_results else {},
            'predictions': results['predictions'],
            'targets': results['targets'],
            'probabilities': results['probabilities'],
            'threshold': threshold,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return evaluation_results
    
    def evaluate_multi_gpu(
        self,
        test_data_path: Optional[str] = None,
        batch_size: int = 128,
        num_workers: int = 8,
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run multi-GPU evaluation.
        
        Args:
            test_data_path: Optional path to test data
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers per GPU
            num_gpus: Number of GPUs to use (None = use all available)
            gpu_ids: Specific GPU IDs to use
            **kwargs: Additional arguments passed to evaluate()
            
        Returns:
            Dictionary containing evaluation results
        """
        # Determine device configuration
        if gpu_ids is not None:
            devices = gpu_ids
        elif num_gpus is not None:
            devices = num_gpus
        else:
            devices = -1  # Use all available GPUs
        
        # Use DDP strategy for multi-GPU
        strategy = "ddp" if isinstance(devices, list) or devices > 1 else "auto"
        
        # Run evaluation with multi-GPU settings
        return self.evaluate(
            test_data_path=test_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            accelerator="gpu",
            devices=devices,
            strategy=strategy,
            **kwargs
        )


def load_and_evaluate(
    checkpoint_path: str,
    test_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 128,
    num_workers: int = 8,
    threshold: float = 0.5,
    multi_gpu: bool = False,
    save_predictions: bool = False,
    plot_results: bool = True
) -> Dict[str, Any]:
    """Convenience function to load model and run evaluation.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Optional path to test data
        output_dir: Directory to save results (if None, don't save)
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        threshold: Classification threshold
        multi_gpu: Whether to use multi-GPU evaluation
        save_predictions: Whether to save predictions to file
        plot_results: Whether to generate plots
        
    Returns:
        Dictionary containing evaluation results
    """
    # Create evaluator
    evaluator = LightningEvaluator(
        checkpoint_path=checkpoint_path
    )
    
    # Run evaluation
    if multi_gpu:
        results = evaluator.evaluate_multi_gpu(
            test_data_path=test_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=threshold
        )
    else:
        results = evaluator.evaluate(
            test_data_path=test_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            threshold=threshold
        )
    
    # Save results if output directory is provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_path = output_path / 'evaluation_results.json'
        save_results = {
            'checkpoint_path': results['checkpoint_path'],
            'evaluation_metrics': results['evaluation_metrics'],
            'trading_metrics': results['trading_metrics'],
            'timestamp': results['timestamp']
        }
        
        # Remove non-serializable data
        if 'curves' in save_results['evaluation_metrics']:
            save_results['evaluation_metrics'].pop('curves')
        
        with open(metrics_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        logger.info(f"Evaluation results saved to: {metrics_path}")
        
        # Save predictions if requested
        if save_predictions:
            predictions_df = pd.DataFrame({
                'target': results['targets'],
                'prediction': results['predictions'],
                'probability': results['probabilities']
            })
            predictions_path = output_path / 'predictions.parquet'
            predictions_df.to_parquet(predictions_path)
            logger.info(f"Predictions saved to: {predictions_path}")
        
        # Generate plots if requested
        if plot_results and 'evaluation_metrics' in results:
            # Use unified evaluation module for plotting
            predictions_dict = {
                'probabilities': results['probabilities'],
                'targets': results['targets']
            }
            generate_all_plots(
                metrics=results['evaluation_metrics'],
                predictions=predictions_dict,
                output_dir=output_path
            )
            logger.info(f"Evaluation plots saved to: {output_path}")
    
    return results
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    ModelCheckpoint,
    LearningRateMonitor,
    DeviceStatsMonitor,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities import grad_norm
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as powerSGD
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
# Delayed import - matplotlib.pyplot imported when needed to avoid distributed training issues

# Import torchmetrics for replacing custom metrics
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryCalibrationError

logger = logging.getLogger(__name__)


class CustomTQDMProgressBar(TQDMProgressBar):
    #Custom TQDM Progress Bar for step-based training.
    #BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, total:{rate_noinv_fmt}{postfix}]"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = None
        self._last_update_time = None
        self._last_n = 0
        self._initial_step = None  # Track the starting step for this session

    def init_train_tqdm(self):
        """Override to properly initialize the progress bar with epoch info."""
        bar = super().init_train_tqdm()
        
        if hasattr(self, 'trainer') and self.trainer is not None:
            max_steps = self.trainer.max_steps if self.trainer.max_steps else 'inf'
            bar.set_description(f"Step {self.trainer.global_step}/{max_steps}")
        
        # Create a dynamic subclass with custom format_dict property
        trainer = self.trainer
        gpu_count = trainer.num_devices if hasattr(trainer, 'num_devices') else 1
        # Store reference to parent's methods
        parent_self = self
        
        class _CustomTqdm(bar.__class__):
            
            @property
            def format_dict(self):
                d = super().format_dict
                # 自己计算速率，因为Lightning不使用update()
                import time
                current_time = time.time()
                # 初始化时间和初始步数
                if parent_self._start_time is None:
                    parent_self._start_time = current_time
                    parent_self._last_update_time = current_time
                    parent_self._initial_step = d['n'] if 'n' in d and d['n'] is not None else 0
                    parent_self._last_n = parent_self._initial_step
                
                # 计算速率
                if 'n' in d and d['n'] is not None:
                    current_step = d['n']
                    # Calculate steps done in this session only
                    session_steps = current_step - parent_self._initial_step
                    elapsed = current_time - parent_self._start_time
                    
                    if elapsed > 0 and session_steps > 0:
                        # 计算平均速率（基于本次会话的步数）
                        avg_rate = session_steps / elapsed
                        
                        # 计算瞬时速率（最近的更新）
                        time_diff = current_time - parent_self._last_update_time
                        n_diff = current_step - parent_self._last_n
                        
                        if time_diff > 0.3 and n_diff > 0:  # 至少0.3秒且有进度
                            instant_rate = n_diff / time_diff
                            # 使用加权平均，偏向瞬时速率
                            rate = 0.4 * avg_rate + 0.6 * instant_rate
                            parent_self._last_update_time = current_time
                            parent_self._last_n = current_step
                        else:
                            rate = avg_rate
                        
                        # 计算总速率并设置到rate字段
                        d['rate'] = rate * trainer.num_devices
                    else:
                        # No progress yet in this session
                        d['rate'] = 0
                        #logger.info(f'****** {rate}')
                return d

        # Change the bar's class to our custom one
        bar.__class__ = _CustomTqdm
        bar.initial = 0
        
        return bar
    # Removed on_train_epoch_start - using step-based training

    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Step-based training - use global batch index
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)









class VisualizationCallback(Callback):
    """Generate automatic visualizations during evaluation."""
    
    def __init__(
        self,
        save_dir: str = "./evaluation_plots",
        plot_types: List[str] = ["confusion_matrix", "roc_curve", "precision_recall", "prediction_distribution"]
    ):
        """Initialize visualization callback.
        
        Args:
            save_dir: Directory to save plots
            plot_types: Types of plots to generate
        """
        self.save_dir = save_dir
        self.plot_types = plot_types
        os.makedirs(save_dir, exist_ok=True)
        
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def on_test_start(self, trainer, pl_module):
        """Reset collections at test start."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect data for visualization."""
        if outputs is not None:
            if 'predictions' in outputs:
                self.predictions.extend(outputs['predictions'].cpu().numpy())
            if 'targets' in outputs:
                self.targets.extend(outputs['targets'].cpu().numpy())
            if 'probabilities' in outputs:
                self.probabilities.extend(outputs['probabilities'].cpu().numpy())
                
    def on_test_end(self, trainer, pl_module):
        """Generate and save visualizations using unified evaluation module."""
        if not self.predictions:
            logger.warning("No predictions collected for visualization")
            return
            
        from utils.evaluation import (
            calculate_metrics,
            generate_all_plots
        )
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities) if self.probabilities else None
        
        # Calculate metrics first (needed for some plots)
        metrics = calculate_metrics(
            predictions=predictions,
            targets=targets,
            probabilities=probabilities
        )
        
        # Prepare predictions dict
        predictions_dict = {
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }
        
        # Generate all plots using unified module
        generate_all_plots(
            metrics=metrics,
            predictions=predictions_dict,
            output_dir=self.save_dir,
            plot_types=self.plot_types
        )
        
        logger.info(f"Saved evaluation plots to {self.save_dir}")
        
        
        
        


class ThresholdOptimizationCallback(Callback):
    """Optimize classification threshold for binary classification."""
    
    def __init__(
        self,
        save_dir: str = "./threshold_optimization",
        optimization_metric: str = "f1_score",
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        num_thresholds: int = 50
    ):
        """Initialize threshold optimization callback.
        
        Args:
            save_dir: Directory to save optimization results
            optimization_metric: Metric to optimize ('f1_score', 'accuracy', 'precision', 'recall')
            threshold_range: Range of thresholds to test
            num_thresholds: Number of thresholds to evaluate
        """
        self.save_dir = save_dir
        self.optimization_metric = optimization_metric
        self.threshold_range = threshold_range
        self.num_thresholds = num_thresholds
        os.makedirs(save_dir, exist_ok=True)
        
        self.probabilities = []
        self.targets = []
        
    def on_test_start(self, trainer, pl_module):
        """Reset collections at test start."""
        self.probabilities = []
        self.targets = []
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect probabilities for threshold optimization."""
        if outputs is not None and 'probabilities' in outputs:
            self.probabilities.extend(outputs['probabilities'].cpu().numpy())
            if 'targets' in outputs:
                self.targets.extend(outputs['targets'].cpu().numpy())
                
    def on_test_end(self, trainer, pl_module):
        """Optimize threshold and save results using unified evaluation module."""
        if not self.probabilities or not self.targets:
            logger.warning("No probabilities collected for threshold optimization")
            return
            
        from utils.evaluation import optimize_threshold
        
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        
        # Use unified optimization function
        optimization_results = optimize_threshold(
            targets=targets,
            probabilities=probabilities,
            optimization_metric=self.optimization_metric,
            threshold_range=self.threshold_range,
            num_thresholds=self.num_thresholds
        )
        
        # Extract results
        optimal_threshold = optimization_results['optimal_threshold']
        optimal_metrics = optimization_results['optimal_metrics']
        results = optimization_results['all_results']
        
        # Only rank 0 should save files and generate plots
        if trainer.is_global_zero:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = os.path.join(self.save_dir, f'threshold_optimization_{timestamp}.json')
            with open(results_file, 'w') as f:
                json.dump({
                    'optimal_threshold': optimal_threshold,
                    'optimal_metrics': optimal_metrics,
                    'optimization_metric': self.optimization_metric,
                    'all_results': results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
            # Plot threshold vs metrics
            results_df = pd.DataFrame(results)
            self._plot_threshold_metrics(results_df, optimal_threshold, timestamp)
            
            logger.info(f"Optimal threshold: {optimal_threshold:.3f} "
                       f"({self.optimization_metric}: {optimal_metrics[self.optimization_metric]:.4f}")
        else:
            logger.debug(f"Rank {trainer.global_rank}: Skipping file save and plot generation")
        
        # Ensure all ranks wait for rank 0 to complete file operations
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception as e:
                logger.warning(f"Barrier after threshold optimization failed: {e}")
        
    def _plot_threshold_metrics(self, results_df, optimal_threshold, timestamp):
        """Plot metrics vs threshold."""
        # Delayed import - matplotlib.pyplot imported when needed to avoid distributed training issues
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            plt.plot(results_df['threshold'], results_df[metric], label=metric.replace('_', ' ').title())
            
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal ({optimal_threshold:.3f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Classification Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Use non-blocking save
        save_path = os.path.join(self.save_dir, f'threshold_metrics_{timestamp}.png')
        plt.savefig(save_path)
        plt.close('all')  # Important: free memory immediately
        logger.debug(f"Saved threshold metrics plot to {save_path}")
        
        # Remove barrier to avoid potential deadlocks
        # Matplotlib file operations should be thread-safe


class InferenceOptimizationCallback(Callback):
    """Production-ready callback for optimizing inference performance.
    
    This callback provides:
    - Automatic torch.compile optimization for faster inference
    - Memory management and cleanup
    - Performance profiling
    - cuDNN optimization
    
    Note: For batch size optimization, use Lightning's auto_scale_batch_size feature
    in the Trainer or configure it in TrainingConfig.
    """
    
    def __init__(
        self,
        enable_torch_compile: bool = True,
        compile_mode: str = "reduce-overhead",
        memory_cleanup_interval: int = 100,
        enable_mixed_precision: bool = True,
        enable_cudnn_benchmark: bool = True,
        log_dir: str = "./inference_logs"
    ):
        """Initialize inference optimization callback.
        
        Args:
            enable_torch_compile: Whether to enable torch.compile optimization
            compile_mode: Compile mode ('default', 'reduce-overhead', 'max-autotune')
            memory_cleanup_interval: Interval for GPU memory cleanup
            enable_mixed_precision: Whether to use mixed precision inference
            enable_cudnn_benchmark: Whether to enable cuDNN auto-tuner
            log_dir: Directory to save inference logs
        """
        self.enable_torch_compile = enable_torch_compile
        self.compile_mode = compile_mode
        self.memory_cleanup_interval = memory_cleanup_interval
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_cudnn_benchmark = enable_cudnn_benchmark
        self.log_dir = log_dir
        
        # State tracking
        self.compiled_model = None
        self.inference_count = 0
        
        os.makedirs(log_dir, exist_ok=True)
        
    def on_predict_start(self, trainer, pl_module):
        """Optimize model for inference at prediction start."""
        
        # Enable cuDNN auto-tuner for optimal convolution algorithms
        if self.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cuDNN auto-tuner for inference optimization")
        
        # Compile model with torch.compile for faster inference
        if self.enable_torch_compile and hasattr(torch, 'compile'):
            logger.info(f"Compiling model with torch.compile (mode={self.compile_mode})")
            try:
                self.compiled_model = torch.compile(
                    pl_module.model,
                    mode=self.compile_mode,
                    fullgraph=True
                )
                # Replace the model with compiled version
                pl_module.model = self.compiled_model
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}. Using uncompiled model.")
        
        # Configure mixed precision
        if self.enable_mixed_precision:
            logger.info("Mixed precision inference enabled")
            
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Track batch processing."""
        # This method is now mainly for potential future extensions
        pass
                
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Track performance and manage memory after prediction."""
        self.inference_count += 1
        
            
        # Periodic memory cleanup
        if self.inference_count % self.memory_cleanup_interval == 0:
            torch.cuda.empty_cache()
            logger.debug(f"GPU memory cleanup at inference {self.inference_count}")
            
    def on_predict_end(self, trainer, pl_module):
        """Save performance statistics and cleanup at prediction end."""
            
        # Create comprehensive report
        report = {
            'inference_count': self.inference_count,
            'optimization_settings': {
                'torch_compile': self.enable_torch_compile,
                'compile_mode': self.compile_mode,
                'mixed_precision': self.enable_mixed_precision,
                'cudnn_benchmark': self.enable_cudnn_benchmark
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Only rank 0 should save report
        if trainer.is_global_zero:
            # Save report
            report_file = os.path.join(self.log_dir, f'inference_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
        # Reset state
        if self.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = False
            
        # Final cleanup
        torch.cuda.empty_cache()
            


class ModelWarmupCallback(Callback):
    """Warmup model before inference to ensure consistent latency.
    
    This callback performs model warmup by running dummy inferences to:
    - Initialize CUDA kernels and memory pools
    - Trigger torch.compile compilation if enabled
    - Stabilize memory allocation patterns
    - Ensure consistent inference latency from the first real request
    """
    
    def __init__(
        self,
        warmup_steps: int = 10,
        warmup_batch_sizes: List[int] = [1, 4, 8, 16, 32],
        measure_warmup_time: bool = True,
        warmup_with_real_data: bool = False,
        log_warmup_stats: bool = True
    ):
        """Initialize model warmup callback.
        
        Args:
            warmup_steps: Number of warmup iterations per batch size
            warmup_batch_sizes: List of batch sizes to warm up
            measure_warmup_time: Whether to measure and log warmup times
            warmup_with_real_data: Whether to use real data samples for warmup
            log_warmup_stats: Whether to log detailed warmup statistics
        """
        self.warmup_steps = warmup_steps
        self.warmup_batch_sizes = warmup_batch_sizes
        self.measure_warmup_time = measure_warmup_time
        self.warmup_with_real_data = warmup_with_real_data
        self.log_warmup_stats = log_warmup_stats
        
        self.warmup_completed = False
        self.warmup_stats = {}
        
    def on_predict_start(self, trainer, pl_module):
        """Perform model warmup before predictions start."""
        if self.warmup_completed:
            logger.info("Model warmup already completed, skipping...")
            return
            
        logger.info("Starting model warmup...")
        start_time = datetime.now()
        
        # Get model config for creating dummy inputs
        model_config = pl_module.model_config
        device = next(pl_module.parameters()).device
        
        # Warmup for each batch size
        for batch_size in self.warmup_batch_sizes:
            batch_stats = self._warmup_batch_size(
                pl_module, 
                batch_size, 
                model_config, 
                device
            )
            self.warmup_stats[f'batch_size_{batch_size}'] = batch_stats
            
        # Log warmup completion
        total_time = (datetime.now() - start_time).total_seconds()
        self.warmup_stats['total_warmup_time_seconds'] = total_time
        
        if self.log_warmup_stats:
            self._log_warmup_statistics()
            
        self.warmup_completed = True
        logger.info(f"Model warmup completed in {total_time:.2f} seconds")
        
    def _warmup_batch_size(
        self, 
        pl_module, 
        batch_size: int, 
        model_config, 
        device: torch.device
    ) -> Dict[str, Any]:
        """Perform warmup for a specific batch size.
        
        Args:
            pl_module: Lightning module
            batch_size: Batch size to warm up
            model_config: Model configuration
            device: Device to run warmup on
            
        Returns:
            Warmup statistics for this batch size
        """
        logger.debug(f"Warming up batch size {batch_size}...")
        
        # Create dummy batch
        if self.warmup_with_real_data and hasattr(pl_module, 'warmup_data'):
            # Use real data if available
            batch = self._get_real_warmup_batch(pl_module, batch_size)
        else:
            # Create synthetic dummy data
            batch = self._create_dummy_batch(batch_size, model_config, device)
            
        # Measure warmup times
        warmup_times = []
        
        # Set model to eval mode
        pl_module.eval()
        
        with torch.no_grad():
            for i in range(self.warmup_steps):
                if self.measure_warmup_time:
                    torch.cuda.synchronize()
                    start = datetime.now()
                    
                # Run prediction
                _ = pl_module.predict_step(batch, batch_idx=i)
                
                if self.measure_warmup_time:
                    torch.cuda.synchronize()
                    warmup_times.append((datetime.now() - start).total_seconds() * 1000)
                    
        # Compile statistics
        stats = {
            'batch_size': batch_size,
            'warmup_steps': self.warmup_steps,
            'times_ms': warmup_times if self.measure_warmup_time else [],
            'mean_time_ms': np.mean(warmup_times) if warmup_times else None,
            'std_time_ms': np.std(warmup_times) if warmup_times else None,
            'first_run_ms': warmup_times[0] if warmup_times else None,
            'last_run_ms': warmup_times[-1] if warmup_times else None
        }
        
        return stats
        
    def _create_dummy_batch(
        self, 
        batch_size: int, 
        model_config, 
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Create dummy batch for warmup.
        
        Args:
            batch_size: Batch size
            model_config: Model configuration
            device: Device to create tensors on
            
        Returns:
            Dummy batch dictionary
        """
        # Default sequence length and feature dimension
        seq_length = getattr(model_config, 'max_sequence_length', 1000)
        feature_dim = getattr(model_config, 'input_dim', 306)
        
        batch = {
            'features': torch.randn(batch_size, seq_length, feature_dim, device=device),
            'mask': torch.ones(batch_size, seq_length, dtype=torch.bool, device=device),
            'timestamps': torch.arange(batch_size, device=device),
            'metadata': {'warmup': True}
        }
        
        return batch
        
    def _get_real_warmup_batch(self, pl_module, batch_size: int) -> Dict[str, torch.Tensor]:
        """Get real data for warmup if available.
        
        Args:
            pl_module: Lightning module
            batch_size: Desired batch size
            
        Returns:
            Real data batch for warmup
        """
        # This would get real cached data if available
        # For now, fall back to dummy data
        return self._create_dummy_batch(
            batch_size,
            pl_module.model_config,
            next(pl_module.parameters()).device
        )
        
    def _log_warmup_statistics(self):
        """Log detailed warmup statistics."""
        logger.info("=== Model Warmup Statistics ===")
        
        for batch_key, stats in self.warmup_stats.items():
            if isinstance(stats, dict) and 'batch_size' in stats:
                logger.info(f"\nBatch size {stats['batch_size']}:")
                if stats['mean_time_ms'] is not None:
                    logger.info(f"  First run: {stats['first_run_ms']:.2f} ms")
                    logger.info(f"  Last run: {stats['last_run_ms']:.2f} ms")
                    logger.info(f"  Mean time: {stats['mean_time_ms']:.2f} ms")
                    logger.info(f"  Std dev: {stats['std_time_ms']:.2f} ms")
                    logger.info(f"  Speedup: {stats['first_run_ms']/stats['last_run_ms']:.2f}x")
                    
        logger.info(f"\nTotal warmup time: {self.warmup_stats.get('total_warmup_time_seconds', 0):.2f} seconds")


class PredictionCalibrationCallback(Callback):
    """Auto-calibrate balance loss weights during training.
    
    This callback dynamically adjusts loss component weights to achieve
    balanced buy/sell predictions (approximately 50/50).
    
    It uses torchmetrics for distributed-aware tracking of prediction statistics
    and adjusts weights based on deviation from target values.
    """
    
    def __init__(
        self,
        target_balance: float = 0.5,
        update_interval: int = 100,
        balance_adjustment_rate: float = 0.05,
        min_balance_weight: float = 0.5,
        max_balance_weight: float = 10.0,
        enable_logging: bool = True,
        log_interval: int = 100
    ):
        """Initialize prediction calibration callback.
        
        Args:
            target_balance: Target ratio of positive predictions (0.5 = 50/50)
            update_interval: How often to update weights (in batches)
            balance_adjustment_rate: Direct percentage adjustment rate for balance (0.05 = 5%)
            min_balance_weight: Absolute minimum value for balance weight
            max_balance_weight: Absolute maximum value for balance weight
            enable_logging: Whether to log statistics
            log_interval: How often to log performance statistics (in batches)
        """
        self.target_balance = target_balance
        self.update_interval = update_interval
        self.balance_adjustment_rate = balance_adjustment_rate
        self.min_balance_weight = min_balance_weight
        self.max_balance_weight = max_balance_weight
        self.enable_logging = enable_logging
        self.log_interval = log_interval
        
        # Initialize torchmetrics for tracking
        self.balance_metric = MeanMetric()
        self.calibration_error_metric = BinaryCalibrationError(n_bins=20)
        
        # Track batch count
        self.batch_count = 0
        
        # Track calibration count
        self.calibration_count = 0
        
        # Adaptive learning rate tracking (keep these as they are)
        self.best_loss = None
        self.loss_plateau_counter = 0
        self.last_lr_reduction_batch = 0
        
        # Store last computed metrics for weight updates
        self.last_balance = None
        self.last_calibration_error = None
        self.last_loss = None
        
        # Bimodal calibration parameters (will be updated from config)
        self.bimodal_target_separation = None  # Will be calculated from mode centers
        self.bimodal_adjustment_rate = 0.01  # Default adjustment rate
        self.bimodal_min_weight = 0.0
        self.bimodal_max_weight = 5.0
        self.last_mode_separation = None
        self.histogram_bins = 50  # For distribution analysis
        
    def on_train_start(self, trainer, pl_module):
        """Initialize tracking at training start."""
        
        # Check if module has the required attributes
        if not hasattr(pl_module, 'binary_classification_config'):
            logger.warning("PredictionCalibrationCallback: Module missing binary_classification_config")
            self.enabled = False
            return
            
        self.enabled = pl_module.binary_classification_config.enable_auto_calibration
        if not self.enabled:
            logger.info("PredictionCalibrationCallback: Auto-calibration disabled in config")
            return
            
        # Store module reference for device tracking
        self.pl_module = pl_module
        
        # Move metrics to the correct device
        self._update_metric_devices()
            
        # Initialize weights if not set
        if not hasattr(pl_module, 'balance_weight'):
            # Balance weight is for maintaining 50/50 buy/sell ratio, not used in bimodal calibration
            pl_module.balance_weight = 1.0  # Default neutral weight
            
        # Initialize bimodal calibration if enabled
        if pl_module.binary_classification_config.enable_bimodal_calibration:
            # Calculate target separation from mode centers
            mode_centers = pl_module.binary_classification_config.bimodal_mode_centers
            if mode_centers is None:
                mode_centers = [0.3, 0.7]  # Default values
            self.bimodal_target_separation = abs(mode_centers[1] - mode_centers[0])
            
            # Update other bimodal parameters from config
            self.bimodal_adjustment_rate = pl_module.binary_classification_config.bimodal_adjustment_rate
            self.bimodal_min_weight = pl_module.binary_classification_config.bimodal_min_weight
            self.bimodal_max_weight = pl_module.binary_classification_config.bimodal_max_weight
            
            # Initialize bimodal weight if not set
            if not hasattr(pl_module, 'bimodal_weight'):
                pl_module.bimodal_weight = pl_module.binary_classification_config.bimodal_mode_weight
            
        # Log initialization details (only rank 0 to avoid duplicates)
        if trainer.is_global_zero:
            logger.info("="*60)
            logger.info("PREDICTION CALIBRATION CALLBACK INITIALIZED")
            logger.info("="*60)
            logger.info(f"Auto-calibration: ENABLED")
            logger.info(f"Update interval: Every {self.update_interval} batches")
            logger.info(f"Using torchmetrics for distributed-aware tracking")
            
            # Log adaptive LR settings if enabled
            if pl_module.binary_classification_config.enable_adaptive_lr:
                logger.info("\nAdaptive Learning Rate:")
                logger.info(f"  Enabled: YES")
                logger.info(f"  Patience: {pl_module.binary_classification_config.lr_patience_batches} batches")
                logger.info(f"  Reduction factor: {pl_module.binary_classification_config.lr_reduction_factor}")
                logger.info(f"  Minimum LR: {pl_module.binary_classification_config.lr_min:.2e}")
                logger.info(f"  Cooldown: {pl_module.binary_classification_config.lr_cooldown_batches} batches")
            
            logger.info("\nInitial Loss Weights:")
            # Balance loss removed - focusing on bimodal calibration only
            
            # Log bimodal calibration configuration if enabled
            if pl_module.binary_classification_config.enable_bimodal_calibration:
                mode_centers = pl_module.binary_classification_config.bimodal_mode_centers
                if mode_centers is None:
                    mode_centers = [0.3, 0.7]
                logger.info("\nBimodal Calibration:")
                logger.info(f"  Enabled: YES")
                logger.info(f"  Mode centers: {mode_centers[0]:.2f} and {mode_centers[1]:.2f}")
                logger.info(f"  Target separation: {self.bimodal_target_separation:.3f} (calculated from centers)")
                logger.info(f"  Adjustment rate: {self.bimodal_adjustment_rate:.1%} per update")
                logger.info(f"  Weight range: [{self.bimodal_min_weight:.1f}, {self.bimodal_max_weight:.1f}]")
                logger.info(f"  Initial bimodal weight: {pl_module.bimodal_weight:.3f}")
            
            logger.info("="*60)
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update statistics and weights after each batch."""
        if batch_idx == 0:
            logger.info("PredictionCalibrationCallback: First batch end")
            
        if not hasattr(self, 'enabled') or not self.enabled:
            return
            
        # Skip if no outputs
        if outputs is None:
            return
            
        # Handle both dict and tensor outputs
        if isinstance(outputs, dict):
            loss = outputs.get('loss')
            predictions = outputs.get('predictions')
            targets = outputs.get('targets')
        else:
            # Backward compatibility: if outputs is just a tensor (loss)
            loss = outputs
            predictions = None
            targets = None
            
        if loss is None:
            return
            
        self.batch_count += 1
        
        # Ensure metrics are on the correct device (in case of device changes)
        self._update_metric_devices()
        
        # Extract predictions and ground truth (for calibration error)
        with torch.no_grad():
            # First try to get from outputs dict (preferred method)
            if predictions is not None and isinstance(predictions, dict) and 'logits' in predictions:
                logits = predictions['logits']
            # Check if module stored outputs for callbacks
            elif hasattr(pl_module, '_last_batch_outputs') and pl_module._last_batch_outputs is not None:
                if 'predictions' in pl_module._last_batch_outputs:
                    predictions = pl_module._last_batch_outputs['predictions']
                    if isinstance(predictions, dict) and 'logits' in predictions:
                        logits = predictions['logits']
                    else:
                        logits = None
                else:
                    logits = None
            # Fallback to model's last forward output
            elif hasattr(pl_module.model, 'last_forward_output') and pl_module.model.last_forward_output is not None:
                logits = pl_module.model.last_forward_output.get('logits')
            else:
                logger.warning("PredictionCalibrationCallback: Unable to get predictions")
                return
                
            if logits is None:
                return
                
            # Get probabilities and predictions
            # Ensure logits are 1D
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)  # Positive class probability
            preds = (probs > 0.5).float()
            
            # Get targets - prefer from outputs dict
            if targets is None:
                targets = batch.get('targets', None)
                if targets is None:
                    targets = batch.get('labels', None)
            
            # Prepare targets using the same method as the lightning module
            if targets is not None and hasattr(pl_module, '_prepare_targets'):
                targets = pl_module._prepare_targets(targets)
            
            # Update torchmetrics
            # 1. Balance metric (mean of predictions)
            self.balance_metric.update(preds)
            
            # 2. Calibration error (requires targets)
            if targets is not None:
                self.calibration_error_metric.update(probs, targets)
            
            # Get current loss value
            if isinstance(loss, torch.Tensor):
                current_loss = loss.item()
            elif isinstance(loss, (int, float)):
                current_loss = loss
            else:
                current_loss = 0
            self.last_loss = current_loss
            
            # Update weights every N batches
            if self.batch_count % self.update_interval == 0:
                self._compute_and_update_weights(pl_module, probs)
                
            # Log statistics
            if self.enable_logging and batch_idx % self.log_interval == 0:
                self._log_statistics(pl_module)
    
    def _compute_and_update_weights(self, pl_module, probs=None):
        """Compute metrics and update loss component weights."""
        # Compute all metrics (this automatically handles distributed sync)
        with torch.no_grad():
            balance = self.balance_metric.compute()
            self.last_balance = balance.item()
            
            # Calibration error might not be available if no targets
            try:
                calibration_error = self.calibration_error_metric.compute()
                self.last_calibration_error = calibration_error.item()
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Calibration error not available: {e}")
                self.last_calibration_error = None
        
        # Reset metrics for next interval
        self.balance_metric.reset()
        self.calibration_error_metric.reset()
        
        # Now update weights based on computed metrics (includes bimodal if enabled)
        self._update_weights_based_on_metrics(pl_module, probs)
                
    def _update_weights_based_on_metrics(self, pl_module, probs=None):
        """Update loss component weights based on torchmetrics values."""
        # Calculate errors from torchmetrics
        balance_error = abs(self.last_balance - self.target_balance)
        
        # Log calibration update header (only rank 0 to avoid duplicates)
        if hasattr(pl_module, 'trainer') and pl_module.trainer.is_global_zero:
            logger.info("\n" + "="*60)
            logger.info(f"AUTO CALIBRATION UPDATE (Batch {self.batch_count})")
            logger.info("="*60)
            logger.info(f"Update Interval: Every {self.update_interval} batches")
            logger.info(f"Calibration Count: {self.calibration_count + 1}")
            active_cals = []
            if pl_module.binary_classification_config.enable_bimodal_calibration:
                active_cals.append("Bimodal")
            logger.info(f"Active Calibrations: {', '.join(active_cals) if active_cals else 'None'}")
            
            # Log current statistics from torchmetrics
            logger.info("\nCurrent Statistics (Distributed-Aware):")
            logger.info(f"  Balance: {self.last_balance:.4f} (target: {self.target_balance}, error: {balance_error:.4f})")
            
            # Interpret balance value
            buy_pct = self.last_balance * 100
            sell_pct = (1 - self.last_balance) * 100
            
            # Define balance zones (using fixed tolerance since balance calibration is disabled)
            balance_threshold = 0.05  # Fixed 5% tolerance
            excellent_range = balance_threshold * 0.2  # Within 20% of tolerance
            good_range = balance_threshold * 0.5       # Within 50% of tolerance
            acceptable_range = balance_threshold       # Within the full tolerance
            
            # Determine balance status
            if balance_error <= excellent_range:
                balance_status = "EXCELLENT ✓✓✓"
                balance_desc = "perfectly balanced"
            elif balance_error <= good_range:
                balance_status = "GOOD ✓✓"
                balance_desc = "well balanced"
            elif balance_error <= acceptable_range:
                balance_status = "ACCEPTABLE ✓"
                balance_desc = "slightly imbalanced"
            elif balance_error <= 0.15:
                balance_status = "NEEDS ADJUSTMENT"
                balance_desc = "moderately imbalanced"
            else:
                balance_status = "POOR - NEEDS CORRECTION"
                balance_desc = "heavily imbalanced"
            
            # Determine direction
            if abs(self.last_balance - self.target_balance) < 0.001:
                direction_desc = ""
            elif self.last_balance > self.target_balance:
                direction_desc = " (too many buys)"
            else:
                direction_desc = " (too many sells)"
            
            logger.info(f"  Interpretation: {buy_pct:.1f}% buy, {sell_pct:.1f}% sell - {balance_status}")
            logger.info(f"  Balance Assessment: {balance_desc}{direction_desc}")
            if self.last_calibration_error is not None:
                logger.info(f"  Calibration Error: {self.last_calibration_error:.4f}")
            
        
        # Balance weight updates removed - focusing on bimodal calibration only
        # The balance weight remains at default 1.0
        
        # Update bimodal weight if enabled and probs available
        if pl_module.binary_classification_config.enable_bimodal_calibration and probs is not None:
            self._update_bimodal_weight(pl_module, probs)
        
        # Adaptive learning rate adjustment
        if pl_module.binary_classification_config.enable_adaptive_lr and hasattr(pl_module, 'optimizers'):
            # Check if loss has improved
            threshold = pl_module.binary_classification_config.lr_min * 0.01
            
            if self.best_loss is None or self.last_loss < self.best_loss - threshold:
                # Loss improved, reset counter
                self.best_loss = self.last_loss
                self.loss_plateau_counter = 0
            else:
                # No improvement, increment counter
                self.loss_plateau_counter += self.update_interval
            
            # Check if we should reduce learning rate
            cooldown_passed = (self.batch_count - self.last_lr_reduction_batch) >= pl_module.binary_classification_config.lr_cooldown_batches
            plateau_detected = self.loss_plateau_counter >= pl_module.binary_classification_config.lr_patience_batches
            
            if cooldown_passed and plateau_detected:
                # Get current learning rate
                optimizer = pl_module.optimizers()
                if isinstance(optimizer, list):
                    optimizer = optimizer[0]
                
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(
                    current_lr * pl_module.binary_classification_config.lr_reduction_factor,
                    pl_module.binary_classification_config.lr_min
                )
                
                # Only reduce if we're not already at minimum
                if new_lr < current_lr:
                    # Update learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    # Reset tracking
                    self.last_lr_reduction_batch = self.batch_count
                    self.loss_plateau_counter = 0
                    self.best_loss = self.last_loss
                    
                    # Log the change (only rank 0)
                    if hasattr(pl_module, 'trainer') and pl_module.trainer.is_global_zero:
                        logger.info(f"\nAdaptive Learning Rate Update:")
                        logger.info(f"  Loss plateau detected after {pl_module.binary_classification_config.lr_patience_batches} batches")
                        logger.info(f"  Current Loss: {self.last_loss:.6f} (best: {self.best_loss:.6f})")
                        logger.info(f"  Old LR: {current_lr:.2e}")
                        logger.info(f"  New LR: {new_lr:.2e}")
                    
                    # Log to tensorboard
                    pl_module.log('calibration/learning_rate', new_lr, on_step=True, on_epoch=False)
        
        # Log closing separator (only rank 0)
        if hasattr(pl_module, 'trainer') and pl_module.trainer.is_global_zero:
            logger.info("="*60 + "\n")
        
        # Increment calibration count
        self.calibration_count += 1
    
    def _update_metric_devices(self):
        """Update metric devices to match the model device."""
        if hasattr(self, 'pl_module') and self.pl_module is not None:
            device = next(self.pl_module.parameters()).device
            if self.balance_metric.device != device:
                self.balance_metric = self.balance_metric.to(device)
            if self.calibration_error_metric.device != device:
                self.calibration_error_metric = self.calibration_error_metric.to(device)
    
    def _update_bimodal_weight(self, pl_module, probs):
        """Update bimodal weight based on mode separation only."""
        import numpy as np
        from scipy.signal import find_peaks
        
        # Convert probabilities to numpy for analysis
        # Handle BFloat16 by converting to float32 first
        if probs.dtype == torch.bfloat16:
            probs_np = probs.float().cpu().numpy()
        else:
            probs_np = probs.cpu().numpy()
        
        # Analyze distribution using histogram
        hist, bin_edges = np.histogram(probs_np, bins=self.histogram_bins, range=(0, 1))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks (modes) using scipy
        # Lower threshold for early training - at least 1% of samples
        min_height = max(len(probs_np) * 0.01, 10)  # At least 10 samples
        peaks, properties = find_peaks(hist, height=min_height, distance=5)
        
        if len(peaks) >= 2:
            # Get two highest peaks
            peak_heights = properties['peak_heights']
            sorted_indices = np.argsort(peak_heights)[-2:]
            mode1_idx = peaks[sorted_indices[0]]
            mode2_idx = peaks[sorted_indices[1]]
            
            mode1 = bin_centers[mode1_idx]
            mode2 = bin_centers[mode2_idx]
            
            # Calculate separation
            separation = abs(mode2 - mode1)
            
            # Log mode positions
            logger.debug(f"Detected modes at {mode1:.3f} and {mode2:.3f}, separation: {separation:.3f}")
        else:
            # If we can't find clear modes, estimate from distribution spread
            # Use percentiles to estimate where modes might form
            p25 = np.percentile(probs_np, 25)
            p75 = np.percentile(probs_np, 75)
            
            # Estimate separation based on spread
            separation = p75 - p25
            
            logger.debug(f"No clear modes detected. Using percentile spread: {separation:.3f} (P25={p25:.3f}, P75={p75:.3f})")
        
        self.last_mode_separation = separation
        
        # Determine adjustment based on new separation logic
        good_threshold = 0.2
        target_separation = self.bimodal_target_separation  # 0.4
        # Calculate the upper bound of good range: target - (target - 0.2)/2
        good_upper_bound = target_separation - (target_separation - good_threshold) / 2  # 0.3 for target=0.4
        
        # Get current weight
        old_weight = pl_module.bimodal_weight if hasattr(pl_module, 'bimodal_weight') else 0.0
        
        if separation < good_threshold:
            # Below good threshold - need to increase weight
            # The farther below 0.2, the more we increase
            increase_factor = (good_threshold - separation) / good_threshold
            multiplier = 1.0 + (self.bimodal_adjustment_rate * (1 + increase_factor))
            strategy = f"INCREASE - Below good threshold ({separation:.3f} < {good_threshold})"
        elif separation <= good_upper_bound:
            # Between 0.2 and 0.3 - good range, no change needed
            multiplier = 1.0
            strategy = f"STABLE - In good range ({good_threshold} ≤ {separation:.3f} ≤ {good_upper_bound:.3f})"
        else:
            # Above good upper bound (0.3) - decrease weight
            # The farther above, the more we decrease
            decrease_factor = min((separation - good_upper_bound) / 0.2, 1.0)
            multiplier = 1.0 - (self.bimodal_adjustment_rate * decrease_factor)
            strategy = f"DECREASE - Above good range ({separation:.3f} > {good_upper_bound:.3f})"
        
        # Update weight
        new_weight = old_weight * multiplier
        new_weight = max(self.bimodal_min_weight, min(new_weight, self.bimodal_max_weight))
        
        pl_module.bimodal_weight = new_weight
        
        # Update loss function if it has the update method
        if hasattr(pl_module, 'loss_fn') and hasattr(pl_module.loss_fn, 'update_mode_weight'):
            pl_module.loss_fn.update_mode_weight(new_weight)
        
        # Log update (only rank 0)
        if hasattr(pl_module, 'trainer') and pl_module.trainer.is_global_zero:
            logger.info(f"\nBimodal Weight Update:")
            
            # Show distribution analysis
            if len(peaks) >= 2:
                logger.info(f"  Distribution: Bimodal detected ({len(peaks)} peaks)")
            else:
                logger.info(f"  Distribution: Not clearly bimodal (using percentile spread)")
            
            good_upper = target_separation - (target_separation - 0.2) / 2
            logger.info(f"  Mode Separation: {separation:.3f} (good: 0.2-{good_upper:.3f}, target: {self.bimodal_target_separation:.3f})")
            
            # Show separation status
            if separation < 0.2:
                sep_status = "❌ Below threshold"
            elif separation <= good_upper:
                sep_status = "✅ Good range"
            else:
                sep_status = "⬇️ Above range (decreasing weight)"
            logger.info(f"  Status: {sep_status}")
            logger.info(f"  Strategy: {strategy}")
            logger.info(f"  Weight: {old_weight:.4f} → {new_weight:.4f} ({(multiplier - 1.0) * 100:+.2f}%)")
            
            # Interpret the weight effect
            if new_weight <= 0.1:
                logger.info(f"  Effect: Minimal bimodal enforcement")
            elif new_weight < 0.5:
                logger.info(f"  Effect: Weak bimodal enforcement")
            elif new_weight < 1.0:
                logger.info(f"  Effect: Moderate bimodal enforcement")
            else:
                logger.info(f"  Effect: Strong bimodal enforcement")
            
    def _log_statistics(self, pl_module):
        """Log current statistics and weights."""
        # Log torchmetrics values if available
        if self.last_balance is not None:
            pl_module.log('calibration/balance', self.last_balance, on_step=True, on_epoch=False)
        if self.last_calibration_error is not None:
            pl_module.log('calibration/calibration_error', self.last_calibration_error, on_step=True, on_epoch=False)
        if self.last_loss is not None:
            pl_module.log('calibration/loss', self.last_loss, on_step=True, on_epoch=False)
        
        # Log current learning rate
        if pl_module.binary_classification_config.enable_adaptive_lr and hasattr(pl_module, 'optimizers'):
            optimizer = pl_module.optimizers()
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            current_lr = optimizer.param_groups[0]['lr']
            pl_module.log('calibration/current_lr', current_lr, on_step=True, on_epoch=False)
        
        # Log weights
        pl_module.log('calibration/balance_weight', pl_module.balance_weight, on_step=True, on_epoch=False)
        
        # Log bimodal calibration metrics if enabled
        if pl_module.binary_classification_config.enable_bimodal_calibration:
            if hasattr(pl_module, 'bimodal_weight'):
                pl_module.log('calibration/bimodal_weight', pl_module.bimodal_weight, on_step=True, on_epoch=False)
            if self.last_mode_separation is not None:
                pl_module.log('calibration/mode_separation', self.last_mode_separation, on_step=True, on_epoch=False)
        
        
        # Distribution logging removed - now using JS divergence metric
            
            
    # Removed on_train_epoch_end - using step-based training
    # Consider adding periodic step-based summaries if needed
    def _disabled_on_train_epoch_end(self, trainer, pl_module):
        """Log summary statistics - disabled for step-based training."""
        if not hasattr(self, 'enabled') or not self.enabled:
            return
            
        # Log epoch summary (only rank 0 to avoid duplicates)
        if trainer.is_global_zero:
            logger.info("\n" + "="*60)
            logger.info(f"EPOCH {trainer.global_step} CALIBRATION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total calibrations performed: {self.calibration_count}")
            logger.info(f"Total batches processed: {self.batch_count}")
            logger.info(f"\nFinal Statistics (from last update):")
            if self.last_balance is not None:
                final_error = abs(self.last_balance - self.target_balance)
                buy_pct = self.last_balance * 100
                sell_pct = (1 - self.last_balance) * 100
                
                # Determine final status
                if final_error <= 0.02:
                    final_status = "EXCELLENT ✓✓✓"
                elif final_error <= 0.05:
                    final_status = "GOOD ✓✓"
                elif final_error <= 0.10:
                    final_status = "ACCEPTABLE ✓"
                else:
                    final_status = "NEEDS IMPROVEMENT ✗"
                
                logger.info(f"  Balance: {self.last_balance:.3f} (target: {self.target_balance}) - {final_status}")
                logger.info(f"  Final Split: {buy_pct:.1f}% buy, {sell_pct:.1f}% sell")
            if self.last_calibration_error is not None:
                logger.info(f"  Calibration Error: {self.last_calibration_error:.3f}")
            logger.info(f"\nFinal Loss Weights:")
            logger.info(f"  Balance: {pl_module.balance_weight:.3f}")
            
            # Log bimodal calibration summary if enabled
            if pl_module.binary_classification_config.enable_bimodal_calibration and hasattr(pl_module, 'bimodal_weight'):
                logger.info(f"  Bimodal: {pl_module.bimodal_weight:.3f}")
                if self.last_mode_separation is not None:
                    good_upper = self.bimodal_target_separation - (self.bimodal_target_separation - 0.2) / 2
                    if self.last_mode_separation < 0.2:
                        separation_status = "❌ BELOW THRESHOLD"
                    elif self.last_mode_separation <= good_upper:
                        separation_status = "✅ GOOD"
                    else:
                        separation_status = "⬇️ ABOVE RANGE"
                    logger.info(f"  Mode Separation: {self.last_mode_separation:.3f} (good: 0.2-{good_upper:.3f}) - {separation_status}")
            
            logger.info("="*60 + "\n")


class TrainingVisualizationCallback(Callback):
    """Generate prediction distribution plots during training.
    
    This callback creates visualizations of prediction probability distributions
    at regular intervals during training, showing the development of the
    prediction distribution over time.
    """
    
    def __init__(
        self,
        plot_interval: int = 1000,
        save_plots_to_disk: bool = True,
        plot_dir: str = "./training_plots",
        histogram_bins: int = 50,
        log_to_tensorboard: bool = True,
        max_samples_per_plot: int = 50000
    ):
        """Initialize training visualization callback.
        
        Args:
            plot_interval: Number of batches between plots
            save_plots_to_disk: Whether to save plots to disk
            plot_dir: Directory to save plots
            histogram_bins: Number of bins for histogram
            log_to_tensorboard: Whether to log to TensorBoard
            max_samples_per_plot: Maximum samples to use for plotting
        """
        self.plot_interval = plot_interval
        self.save_plots_to_disk = save_plots_to_disk
        self.plot_dir = Path(plot_dir)
        self.histogram_bins = histogram_bins
        self.log_to_tensorboard = log_to_tensorboard
        self.max_samples_per_plot = max_samples_per_plot
        
        # Create plot directory
        if self.save_plots_to_disk:
            self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.batch_count = 0
        self.accumulated_probs = []
        self.plot_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect predictions and generate plots at intervals."""
        self.batch_count += 1
        
        # Extract predictions
        with torch.no_grad():
            # Get logits from the model's last forward pass
            if hasattr(pl_module.model, 'last_forward_output') and pl_module.model.last_forward_output is not None:
                logits = pl_module.model.last_forward_output.get('logits')
            else:
                # Try to get from batch forward pass
                features = batch['features']
                mask = batch.get('mask', None)
                predictions = pl_module(features, mask=mask)
                logits = predictions['logits']
                
            if logits is None:
                return
                
            # Get probabilities based on classification mode
            if hasattr(pl_module, 'classification_mode') and pl_module.classification_mode == 'ternary':
                # Ternary classification - accumulate class probabilities
                # logits shape: [batch_size, 3]
                probs = torch.softmax(logits, dim=-1)  # Convert to probabilities

                # Initialize separate lists for each class if needed
                if not hasattr(self, 'accumulated_probs_ternary'):
                    self.accumulated_probs_ternary = {'hold': [], 'buy': [], 'sell': []}
                if not hasattr(self, 'accumulated_predictions_ternary'):
                    self.accumulated_predictions_ternary = {'hold': 0, 'buy': 0, 'sell': 0}

                # Accumulate probabilities for each class
                probs_np = probs.cpu().float().numpy()
                self.accumulated_probs_ternary['hold'].extend(probs_np[:, 0].tolist())
                self.accumulated_probs_ternary['buy'].extend(probs_np[:, 1].tolist())
                self.accumulated_probs_ternary['sell'].extend(probs_np[:, 2].tolist())

                # Count actual predictions (argmax)
                predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                for pred in predictions:
                    if pred == 0:
                        self.accumulated_predictions_ternary['hold'] += 1
                    elif pred == 1:
                        self.accumulated_predictions_ternary['buy'] += 1
                    elif pred == 2:
                        self.accumulated_predictions_ternary['sell'] += 1

                # Limit accumulated samples
                if len(self.accumulated_probs_ternary['hold']) > self.max_samples_per_plot:
                    for key in self.accumulated_probs_ternary:
                        self.accumulated_probs_ternary[key] = self.accumulated_probs_ternary[key][-self.max_samples_per_plot:]
            else:
                # Binary classification
                # Ensure logits are 1D
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)  # Positive class probability

                # Accumulate probabilities
                self.accumulated_probs.extend(probs.cpu().float().numpy().tolist())
            
            # Limit accumulated samples
            if len(self.accumulated_probs) > self.max_samples_per_plot:
                # Keep only the most recent samples
                self.accumulated_probs = self.accumulated_probs[-self.max_samples_per_plot:]
        
        # Generate plot at intervals (only on rank 0)
        if trainer.is_global_zero and self.batch_count % self.plot_interval == 0:
            # Check if we have data to plot
            if hasattr(pl_module, 'classification_mode') and pl_module.classification_mode == 'ternary':
                if hasattr(self, 'accumulated_probs_ternary') and len(self.accumulated_probs_ternary['hold']) > 0:
                    self._generate_ternary_distribution_plot(trainer, pl_module)
            elif len(self.accumulated_probs) > 0:
                self._generate_distribution_plot(trainer, pl_module)
            
    def _generate_ternary_distribution_plot(self, trainer, pl_module):
        """Generate and save prediction distribution plots for ternary classification."""
        import matplotlib.pyplot as plt
        from scipy import stats

        # Create single figure
        fig, ax = plt.subplots(figsize=(12, 7))

        class_names = ['Hold', 'Buy', 'Sell']
        class_keys = ['hold', 'buy', 'sell']
        colors = ['blue', 'green', 'red']

        # Store statistics for legend
        legend_elements = []
        stats_info = []

        # Plot each class distribution
        for class_name, class_key, color in zip(class_names, class_keys, colors):
            probs = np.array(self.accumulated_probs_ternary[class_key])

            # Create histogram with more transparency for overlapping
            counts, bins, patches = ax.hist(
                probs,
                bins=self.histogram_bins,
                density=True,
                alpha=0.3,
                color=color,
                edgecolor=color,
                linewidth=1,
                label=f'{class_name} (histogram)'
            )

            # Add KDE (kernel density estimation) for smooth curve
            if len(probs) > 1:
                try:
                    kde = stats.gaussian_kde(probs)
                    x_range = np.linspace(0, 1, 200)
                    density = kde(x_range)
                    ax.plot(x_range, density, color=color, linewidth=2,
                           label=f'{class_name} (μ={probs.mean():.3f}, σ={probs.std():.3f})')
                except:
                    # Fallback if KDE fails
                    pass

            # Add mean line
            mean_prob = probs.mean()
            ax.axvline(mean_prob, color=color, linestyle='--', linewidth=1.5, alpha=0.7)

            # Store statistics (only for internal use, not display)
            stats_info.append({
                'name': class_name,
                'mean': probs.mean(),
                'std': probs.std()
            })

        # Add grid and labels
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Ternary Classification Probability Distributions - Step {trainer.global_step}',
                    fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3)

        # Create legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # Add simplified info box with only prediction rates
        if hasattr(self, 'accumulated_predictions_ternary'):
            total_predictions = sum(self.accumulated_predictions_ternary.values())
            if total_predictions > 0:
                info_text = f"Actual Predictions (n={total_predictions}):\n"
                info_text += "-"*25 + "\n"

                hold_rate = self.accumulated_predictions_ternary['hold'] / total_predictions * 100
                buy_rate = self.accumulated_predictions_ternary['buy'] / total_predictions * 100
                sell_rate = self.accumulated_predictions_ternary['sell'] / total_predictions * 100

                info_text += f"Hold: {hold_rate:5.1f}% ({self.accumulated_predictions_ternary['hold']:5d})\n"
                info_text += f"Buy:  {buy_rate:5.1f}% ({self.accumulated_predictions_ternary['buy']:5d})\n"
                info_text += f"Sell: {sell_rate:5.1f}% ({self.accumulated_predictions_ternary['sell']:5d})\n"
                info_text += "-"*25 + "\n"
                info_text += f"Target: [80%, 10%, 10%]"

                ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
                       family='monospace')

        # Add reference lines
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='0.5 reference')
        ax.axhline(0, color='black', linewidth=0.5)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plot_path = os.path.join(self.plot_dir, f'ternary_distribution_step_{trainer.global_step}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')

        # Log to tensorboard if available (before closing)
        if trainer.logger:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if hasattr(trainer.logger, 'experiment') and isinstance(trainer.logger.experiment, SummaryWriter):
                    trainer.logger.experiment.add_figure(
                        'ternary_distribution', fig, global_step=trainer.global_step
                    )
            except Exception as e:
                logger.debug(f"Could not log figure to tensorboard: {e}")

        plt.close()

        # Reset prediction counters after plotting to track new predictions
        if hasattr(self, 'accumulated_predictions_ternary'):
            self.accumulated_predictions_ternary = {'hold': 0, 'buy': 0, 'sell': 0}

        logger.info(f"Saved ternary distribution plot to {plot_path}")

    def _generate_distribution_plot(self, trainer, pl_module):
        """Generate and save prediction distribution plot."""
        import matplotlib.pyplot as plt
        
        probs = np.array(self.accumulated_probs)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        counts, bins, patches = plt.hist(
            probs, 
            bins=self.histogram_bins, 
            density=True, 
            alpha=0.7, 
            color='orange',
            edgecolor='black',
            linewidth=0.5
        )
        
        # Calculate statistics for bimodal distribution
        # Find modes using kernel density estimation
        from scipy import stats
        from scipy.signal import find_peaks
        
        # Calculate KDE for smoother mode detection with error handling
        try:
            # Check if the data has sufficient variance
            probs_std = probs.std()
            if probs_std < 1e-6:
                # Data is too uniform for KDE, use simple statistics
                logger.warning(f"Probabilities have very low variance (std={probs_std:.2e}), skipping KDE")
                mode1 = probs.min()
                mode2 = probs.max()
                # If they're still too close, create artificial separation
                if abs(mode2 - mode1) < 1e-6:
                    mean_val = probs.mean()
                    mode1 = max(0.0, mean_val - 0.01)
                    mode2 = min(1.0, mean_val + 0.01)
            else:
                kde = stats.gaussian_kde(probs)
                x_range = np.linspace(0, 1, 200)
                kde_values = kde(x_range)
                
                # Find peaks (modes)
                peaks, properties = find_peaks(kde_values, height=0.1, distance=20)
                
                if len(peaks) >= 2:
                    # Get the two highest peaks
                    peak_heights = properties['peak_heights']
                    sorted_indices = np.argsort(peak_heights)[-2:]
                    mode1_idx = peaks[sorted_indices[0]]
                    mode2_idx = peaks[sorted_indices[1]]
                    
                    mode1 = x_range[mode1_idx]
                    mode2 = x_range[mode2_idx]
                    
                    # Ensure mode1 < mode2
                    if mode1 > mode2:
                        mode1, mode2 = mode2, mode1
                else:
                    # Fallback if we can't find two clear modes
                    mode1 = np.percentile(probs, 25)
                    mode2 = np.percentile(probs, 75)
        except Exception as e:
            logger.warning(f"Error in KDE calculation: {e}. Using simple statistics.")
            mode1 = np.percentile(probs, 25)
            mode2 = np.percentile(probs, 75)
            if abs(mode2 - mode1) < 1e-6:
                mean_val = probs.mean()
                mode1 = max(0.0, mean_val - 0.01)
                mode2 = min(1.0, mean_val + 0.01)
        
        # Use actual prediction threshold from module
        threshold = pl_module.prediction_threshold
        
        # Add vertical lines
        plt.axvline(x=mode1, color='blue', linestyle='--', linewidth=2, 
                   label=f'Mode 1: {mode1:.3f}')
        plt.axvline(x=mode2, color='orange', linestyle='--', linewidth=2, 
                   label=f'Mode 2: {mode2:.3f}')
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.3f}')
        plt.axvline(x=0.5, color='gray', linestyle=':', linewidth=2, 
                   label='0.5')
        
        # Labels and title
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title(f'Prediction Distribution - Epoch {trainer.global_step}, Batch {self.batch_count}')
        plt.legend(loc='upper right')
        plt.xlim(0, 1)
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        if self.save_plots_to_disk:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.plot_dir / f'distribution_epoch{trainer.global_step}_batch{self.batch_count}_{timestamp}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            # Remove barrier to avoid potential deadlocks
            # Matplotlib file operations should be thread-safe
            logger.info(f"Saved distribution plot to {filename}")
        
        # Log to TensorBoard
        if self.log_to_tensorboard and hasattr(pl_module.logger, 'experiment'):
            if hasattr(pl_module.logger.experiment, 'add_figure'):
                pl_module.logger.experiment.add_figure(
                    'training/prediction_distribution',
                    plt.gcf(),
                    global_step=pl_module.global_step
                )
        
        plt.close()
        
        # Log statistics
        mean_prob = probs.mean()
        std_prob = probs.std()
        logger.info(f"Distribution stats - Mean: {mean_prob:.3f}, Std: {std_prob:.3f}, "
                   f"Mode1: {mode1:.3f}, Mode2: {mode2:.3f}, Threshold: {threshold:.3f}")
        
        # Clear accumulated probabilities after plotting
        self.accumulated_probs = []
        self.plot_count += 1
        
    # Removed on_train_epoch_end - using step-based training  
    def _disabled_on_train_epoch_end(self, trainer, pl_module):
        """Generate final plot - disabled for step-based training."""
        # Only rank 0 should generate plots to avoid DDP synchronization issues
        if trainer.is_global_zero and len(self.accumulated_probs) > 0:
            logger.info(f"Rank 0: Generating end-of-epoch distribution plot with {len(self.accumulated_probs)} samples")
            self._generate_distribution_plot(trainer, pl_module)
        elif not trainer.is_global_zero:
            # Other ranks should clear their accumulated data
            logger.debug(f"Rank {trainer.global_rank}: Clearing {len(self.accumulated_probs)} accumulated probabilities")
            self.accumulated_probs = []


from collections import defaultdict


# BatchMetricsCSVCallback removed - CSV logging disabled
# OrderBookPredictionWriter removed - not used
"""Callback for verifying global distribution synchronization in multi-GPU training."""

import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Any
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)


class DistributionVerificationCallback(Callback):
    """Callback to monitor and verify distribution statistics across GPUs.
    
    This callback helps verify that global distribution synchronization is working
    correctly by tracking and comparing statistics across all GPUs.
    """
    
    def __init__(
        self,
        log_interval: int = 50,
        variance_warning_threshold: float = 0.01,
        log_to_tensorboard: bool = True,
        verbose: bool = True
    ):
        """Initialize the distribution verification callback.
        
        Args:
            log_interval: How often to log verification stats (in steps)
            variance_warning_threshold: Threshold for warning about high variance
            log_to_tensorboard: Whether to log metrics to TensorBoard
            verbose: Whether to print detailed logs
        """
        super().__init__()
        self.log_interval = log_interval
        self.variance_warning_threshold = variance_warning_threshold
        self.log_to_tensorboard = log_to_tensorboard
        self.verbose = verbose
        
        # Storage for statistics
        self.step_count = 0
        self.stats_history: List[Dict[str, Any]] = []
        
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log initial setup information."""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            
            # Check if global distribution matching is enabled
            is_global = (hasattr(pl_module, 'binary_classification_config') and 
                        pl_module.binary_classification_config is not None and
                        getattr(pl_module.binary_classification_config, 'global_distribution_matching', False))
            
            if rank == 0:
                logger.info("="*80)
                logger.info("Distribution Verification Callback Active")
                logger.info(f"World size: {world_size}")
                logger.info(f"Global distribution matching: {'ENABLED' if is_global else 'DISABLED'}")
                logger.info(f"Verification interval: every {self.log_interval} steps")
                logger.info("="*80)
        else:
            logger.info("Distribution Verification: Single GPU mode - no verification needed")
    
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """Monitor distribution statistics after each batch."""
        self.step_count += 1
        
        # Only verify at specified intervals
        if self.step_count % self.log_interval != 0:
            return
            
        # Skip if not in distributed mode
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        
        # Get local statistics from the module
        local_stats = pl_module.get_distribution_stats()
        if not local_stats:
            return
            
        # Perform distributed verification
        self._verify_distributions(trainer, pl_module, local_stats)
    
    def _verify_distributions(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        local_stats: Dict[str, Any]
    ) -> None:
        """Verify distribution statistics across all GPUs."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Key metrics to verify
        local_positive_ratio = torch.tensor(local_stats['actual_positive_ratio'], device=pl_module.device)
        local_predicted_ratio = torch.tensor(local_stats['predicted_positive_ratio'], device=pl_module.device)
        local_batch_size = torch.tensor(local_stats['batch_size'], device=pl_module.device)
        
        # Gather from all GPUs
        all_positive_ratios = [torch.zeros_like(local_positive_ratio) for _ in range(world_size)]
        all_predicted_ratios = [torch.zeros_like(local_predicted_ratio) for _ in range(world_size)]
        all_batch_sizes = [torch.zeros_like(local_batch_size) for _ in range(world_size)]
        
        dist.all_gather(all_positive_ratios, local_positive_ratio)
        dist.all_gather(all_predicted_ratios, local_predicted_ratio)
        dist.all_gather(all_batch_sizes, local_batch_size)
        
        # Convert to numpy for analysis
        positive_ratios = np.array([r.cpu().item() for r in all_positive_ratios])
        predicted_ratios = np.array([r.cpu().item() for r in all_predicted_ratios])
        batch_sizes = np.array([b.cpu().item() for b in all_batch_sizes])
        
        # Calculate statistics
        stats = {
            'step': self.step_count,
            'positive_ratio_mean': positive_ratios.mean(),
            'positive_ratio_std': positive_ratios.std(),
            'positive_ratio_variance': positive_ratios.var(),
            'predicted_ratio_mean': predicted_ratios.mean(),
            'predicted_ratio_std': predicted_ratios.std(),
            'predicted_ratio_variance': predicted_ratios.var(),
            'batch_size_mean': batch_sizes.mean(),
            'batch_size_std': batch_sizes.std(),
        }
        
        # Calculate per-GPU deviations
        stats['max_positive_ratio_deviation'] = np.abs(positive_ratios - positive_ratios.mean()).max()
        stats['max_predicted_ratio_deviation'] = np.abs(predicted_ratios - predicted_ratios.mean()).max()
        
        # Store history
        self.stats_history.append(stats)
        
        # Log to TensorBoard if enabled
        if self.log_to_tensorboard:
            pl_module.log('verify/positive_ratio_variance', stats['positive_ratio_variance'])
            pl_module.log('verify/predicted_ratio_variance', stats['predicted_ratio_variance'])
            pl_module.log('verify/max_positive_deviation', stats['max_positive_ratio_deviation'])
            pl_module.log('verify/max_predicted_deviation', stats['max_predicted_ratio_deviation'])
        
        # Log detailed information on rank 0
        if rank == 0 and self.verbose:
            self._log_verification_results(stats, positive_ratios, predicted_ratios, world_size)
    
    def _log_verification_results(
        self,
        stats: Dict[str, Any],
        positive_ratios: np.ndarray,
        predicted_ratios: np.ndarray,
        world_size: int
    ) -> None:
        """Log detailed verification results."""
        logger.info("\n" + "="*60)
        logger.info(f"Distribution Verification Report - Step {self.step_count}")
        logger.info("="*60)
        
        # Per-GPU statistics
        logger.info("Per-GPU Actual Positive Ratios:")
        for i, ratio in enumerate(positive_ratios):
            logger.info(f"  GPU {i}: {ratio:.4f}")
        
        logger.info(f"\nGlobal Statistics:")
        logger.info(f"  Mean positive ratio: {stats['positive_ratio_mean']:.4f}")
        logger.info(f"  Std positive ratio: {stats['positive_ratio_std']:.4f}")
        logger.info(f"  Variance: {stats['positive_ratio_variance']:.6f}")
        logger.info(f"  Max deviation: {stats['max_positive_ratio_deviation']:.4f}")
        
        # Check for high variance
        if stats['positive_ratio_variance'] > self.variance_warning_threshold:
            logger.warning(f"⚠️  HIGH VARIANCE DETECTED: {stats['positive_ratio_variance']:.6f} > {self.variance_warning_threshold}")
            logger.warning("   This suggests GPUs are seeing different distributions!")
        else:
            logger.info(f"✓ Variance within acceptable range: {stats['positive_ratio_variance']:.6f}")
        
        # Check if using global distribution matching
        logger.info(f"\nPredicted Positive Ratios:")
        for i, ratio in enumerate(predicted_ratios):
            logger.info(f"  GPU {i}: {ratio:.4f}")
        
        logger.info(f"\nPredicted Statistics:")
        logger.info(f"  Mean: {stats['predicted_ratio_mean']:.4f}")
        logger.info(f"  Std: {stats['predicted_ratio_std']:.4f}")
        logger.info(f"  Max deviation: {stats['max_predicted_ratio_deviation']:.4f}")
        
        logger.info("="*60 + "\n")
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log final summary statistics."""
        if not self.stats_history or dist.get_rank() != 0:
            return
            
        logger.info("\n" + "="*80)
        logger.info("Distribution Verification Final Summary")
        logger.info("="*80)
        
        # Calculate summary statistics over entire training
        positive_variances = [s['positive_ratio_variance'] for s in self.stats_history]
        predicted_variances = [s['predicted_ratio_variance'] for s in self.stats_history]
        
        logger.info(f"Training Statistics:")
        logger.info(f"  Total verification checks: {len(self.stats_history)}")
        logger.info(f"  Average positive ratio variance: {np.mean(positive_variances):.6f}")
        logger.info(f"  Max positive ratio variance: {np.max(positive_variances):.6f}")
        logger.info(f"  Average predicted ratio variance: {np.mean(predicted_variances):.6f}")
        logger.info(f"  Max predicted ratio variance: {np.max(predicted_variances):.6f}")
        
        # Count warnings
        high_variance_count = sum(1 for v in positive_variances if v > self.variance_warning_threshold)
        if high_variance_count > 0:
            logger.warning(f"\n⚠️  High variance detected in {high_variance_count}/{len(self.stats_history)} checks")
        else:
            logger.info(f"\n✓ All variance checks passed!")
        


class PowerSGDCallback(Callback):
    """Callback to configure and monitor PowerSGD gradient compression.
    
    This callback:
    1. Configures PowerSGD after model is distributed
    2. Monitors compression statistics
    3. Handles warm-up period before enabling compression
    4. Tracks communication time savings
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.powersgd_state = None
        self.compression_stats = {
            'compression_ratios': [],
            'communication_times': [],
            'bytes_communicated': [],
            'iteration': 0
        }
        self.hook_registered = False
        self.start_time = None
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Configure PowerSGD after model is distributed."""
        if not self.config.training.use_powersgd:
            return
            
        # Check if we're in distributed mode
        if not trainer.world_size > 1:
            logger.warning("PowerSGD requested but not in distributed mode. Skipping.")
            return
            
        # Get the DDP model
        if hasattr(trainer.strategy, 'model') and hasattr(trainer.strategy.model, 'register_comm_hook'):
            model = trainer.strategy.model
            
            # Create PowerSGD state
            self.powersgd_state = powerSGD.PowerSGDState(
                process_group=None,  # Use default process group
                matrix_approximation_rank=self.config.training.powersgd_rank,
                start_powerSGD_iter=self.config.training.powersgd_start_iter,
                min_compression_rate=self.config.training.powersgd_min_compression_rate,
                use_error_feedback=self.config.training.powersgd_use_error_feedback,
                warm_start=self.config.training.powersgd_warm_start,
                orthogonalization_epsilon=self.config.training.powersgd_orthogonalization_epsilon,
                # Note: grad_buffer_size_mb is not a valid parameter in PyTorch
                batch_tensors_with_same_shape=True,  # Group tensors by shape for efficiency
            )
            
            # Register the communication hook
            model.register_comm_hook(self.powersgd_state, powerSGD.powerSGD_hook)
            self.hook_registered = True
            
            logger.info(f"✓ PowerSGD hook registered successfully")
            logger.info(f"  - Rank: {self.config.training.powersgd_rank}")
            logger.info(f"  - Start iteration: {self.config.training.powersgd_start_iter}")
            logger.info(f"  - Min compression rate: {self.config.training.powersgd_min_compression_rate}")
            logger.info(f"  - Error feedback: {self.config.training.powersgd_use_error_feedback}")
            
            # Count model parameters
            total_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            total_mb = total_params * 4 / (1024 * 1024)  # Assuming FP32
            logger.info(f"  - Model size: {total_params:,} parameters ({total_mb:.1f} MB)")
            logger.info(f"  - Expected compression: {total_mb:.1f} MB → ~{total_mb/10:.1f} MB per iteration")
        else:
            logger.warning("Could not register PowerSGD hook. Model may not be properly distributed.")
    
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int):
        """Track communication time."""
        if self.hook_registered:
            self.start_time = time.time()
            
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        """Log compression statistics."""
        if not self.hook_registered:
            return
            
        # Update iteration counter
        self.compression_stats['iteration'] += 1
        
        # Track communication time
        if self.start_time is not None:
            comm_time = time.time() - self.start_time
            self.compression_stats['communication_times'].append(comm_time)
            
        # Log statistics periodically
        if self.compression_stats['iteration'] % 100 == 0:
            if self.compression_stats['iteration'] >= self.config.training.powersgd_start_iter:
                # PowerSGD is active
                avg_time = np.mean(self.compression_stats['communication_times'][-100:])
                pl_module.log('powersgd/communication_time', avg_time, on_step=True, on_epoch=False)
                pl_module.log('powersgd/active', 1.0, on_step=True, on_epoch=False)
                
                # Estimate compression ratio based on rank
                estimated_compression = self.config.training.powersgd_rank * 2 / 100  # Rough estimate
                pl_module.log('powersgd/compression_ratio', 1.0 / estimated_compression, on_step=True, on_epoch=False)
            else:
                # Still in warm-up
                remaining = self.config.training.powersgd_start_iter - self.compression_stats['iteration']
                pl_module.log('powersgd/warmup_remaining', remaining, on_step=True, on_epoch=False)
                pl_module.log('powersgd/active', 0.0, on_step=True, on_epoch=False)
                
    # Removed on_train_epoch_end - using step-based training
    def _disabled_on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log compression statistics - disabled for step-based training."""
        if not self.hook_registered:
            return
            
        if len(self.compression_stats['communication_times']) > 0:
            avg_comm_time = np.mean(self.compression_stats['communication_times'])
            total_comm_time = np.sum(self.compression_stats['communication_times'])
            
            logger.info(f"\nPowerSGD Statistics - Epoch {trainer.global_step}:")
            logger.info(f"  Total iterations: {self.compression_stats['iteration']}")
            logger.info(f"  Average communication time: {avg_comm_time:.3f}s")
            logger.info(f"  Total communication time: {total_comm_time:.1f}s")
            
            if self.compression_stats['iteration'] >= self.config.training.powersgd_start_iter:
                logger.info(f"  PowerSGD Status: ACTIVE (rank={self.config.training.powersgd_rank})")
            else:
                remaining = self.config.training.powersgd_start_iter - self.compression_stats['iteration']
                logger.info(f"  PowerSGD Status: WARMING UP ({remaining} iterations remaining)")
                
    def state_dict(self) -> Dict[str, Any]:
        """Save callback state."""
        return {
            'compression_stats': self.compression_stats,
            'hook_registered': self.hook_registered
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load callback state."""
        self.compression_stats = state_dict.get('compression_stats', self.compression_stats)
        self.hook_registered = state_dict.get('hook_registered', False)


class GradientLoggerCallback(Callback):
    """
    Callback for comprehensive gradient logging and monitoring.
    
    Features:
    - Per-loss-component gradient tracking
    - Layer-wise gradient statistics
    - Gradient flow visualization
    - Gradient anomaly detection
    - Detailed gradient histograms (W&B only)
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_component_gradients: bool = True,
        log_layer_gradients: bool = True,
        log_gradient_histograms: bool = False,
        detect_anomalies: bool = True,
        anomaly_threshold: float = 50.0,
        track_gradient_flow: bool = True
    ):
        """
        Initialize gradient logger callback.
        
        Args:
            log_every_n_steps: Frequency of gradient logging
            log_component_gradients: Log gradients for individual loss components
            log_layer_gradients: Log gradients for each model layer
            log_gradient_histograms: Log gradient histograms (requires W&B)
            detect_anomalies: Detect gradient anomalies (explosions, vanishing)
            anomaly_threshold: Threshold for gradient anomaly detection
            track_gradient_flow: Track gradient flow through the network
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_component_gradients = log_component_gradients
        self.log_layer_gradients = log_layer_gradients
        self.log_gradient_histograms = log_gradient_histograms
        self.detect_anomalies = detect_anomalies
        self.anomaly_threshold = anomaly_threshold
        self.track_gradient_flow = track_gradient_flow
        
        # Track gradient statistics over time
        self.gradient_history = defaultdict(list)
        self.component_gradient_history = defaultdict(list)
        
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Called before optimizer step to log gradients."""
        
        # Only log at specified intervals
        if trainer.global_step % self.log_every_n_steps != 0:
            return
            
        # Get overall gradient norm
        norms = grad_norm(pl_module, norm_type=2)
        if norms and "grad_2.0_norm_total" in norms:
            total_norm = norms["grad_2.0_norm_total"]
            
            # Detect gradient anomalies
            if self.detect_anomalies:
                self._detect_gradient_anomalies(total_norm, trainer.global_step)
            
            # Track gradient flow
            if self.track_gradient_flow:
                self._track_gradient_flow(pl_module, trainer)
            
            # Log layer-wise gradients
            if self.log_layer_gradients:
                self._log_layer_gradients(pl_module, trainer)
                
            # Log component gradients if the module supports it
            if self.log_component_gradients and hasattr(pl_module, '_last_loss_dict'):
                self._log_component_gradients(pl_module, trainer)
                
            # Log gradient histograms for W&B
            if self.log_gradient_histograms and trainer.logger:
                try:
                    import wandb
                    if isinstance(trainer.logger.experiment, wandb.sdk.wandb_run.Run):
                        self._log_gradient_histograms(pl_module, trainer)
                except ImportError:
                    pass
    
    def _detect_gradient_anomalies(self, grad_norm: float, global_step: int) -> None:
        """Detect gradient explosions or vanishing gradients."""
        
        # Check for gradient explosion
        if grad_norm > self.anomaly_threshold:
            logger.warning(
                f"⚠️  Gradient explosion detected at step {global_step}: "
                f"norm={grad_norm:.2f} (threshold={self.anomaly_threshold})"
            )
            
        # Check for vanishing gradients
        elif grad_norm < 1e-7:
            logger.warning(
                f"⚠️  Vanishing gradient detected at step {global_step}: "
                f"norm={grad_norm:.2e}"
            )
    
    def _track_gradient_flow(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Track gradient flow through the network."""
        
        # Get gradients from first and last layers
        first_layer_grad = None
        last_layer_grad = None
        
        # Find first and last layers with gradients
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                
                if first_layer_grad is None:
                    first_layer_grad = (name, grad_norm)
                last_layer_grad = (name, grad_norm)
        
        if first_layer_grad and last_layer_grad:
            # Calculate gradient flow ratio
            if first_layer_grad[1] > 0:
                flow_ratio = last_layer_grad[1] / first_layer_grad[1]
                trainer.logger.log_metrics({
                    "gradient_flow/ratio": flow_ratio,
                    "gradient_flow/first_layer_norm": first_layer_grad[1],
                    "gradient_flow/last_layer_norm": last_layer_grad[1]
                }, step=trainer.global_step)
                
                # Warn if gradient flow is poor
                if flow_ratio < 0.01:
                    logger.warning(
                        f"⚠️  Poor gradient flow detected: "
                        f"last/first ratio = {flow_ratio:.4f}"
                    )
    
    def _log_layer_gradients(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Log gradient statistics for each layer."""
        
        layer_stats = {}
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Compute statistics
                grad_norm = grad.norm(2).item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                grad_max = grad.abs().max().item()
                
                # Store in dict
                layer_stats[f"grad_norm/layers/{name}"] = grad_norm
                layer_stats[f"grad_mean/layers/{name}"] = grad_mean
                layer_stats[f"grad_std/layers/{name}"] = grad_std
                layer_stats[f"grad_max/layers/{name}"] = grad_max
                
                # Track history
                self.gradient_history[name].append(grad_norm)
        
        # Log all layer stats
        if layer_stats:
            trainer.logger.log_metrics(layer_stats, step=trainer.global_step)
    
    def _log_component_gradients(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Log gradients for individual loss components."""
        
        # Check if module has component gradient computation
        criterion = getattr(pl_module, 'criterion', None)
        if criterion is None or not hasattr(criterion, 'compute_component_gradients'):
            return
            
        # Get last loss dict
        loss_dict = getattr(pl_module, '_last_loss_dict', None)
        if loss_dict is None:
            return
            
        try:
            # Compute component gradients
            component_grads = criterion.compute_component_gradients(
                loss_dict,
                pl_module.parameters()
            )
            
            # Log component gradients
            component_metrics = {}
            for component, grad_norm in component_grads.items():
                metric_name = f"grad_norm/components/{component}"
                component_metrics[metric_name] = grad_norm
                
                # Track history
                self.component_gradient_history[component].append(grad_norm)
                
            if component_metrics:
                trainer.logger.log_metrics(component_metrics, step=trainer.global_step)
                
                # Log relative contributions
                total_grad = component_grads.get('total', 1.0)
                if total_grad > 0:
                    relative_metrics = {}
                    for component, grad_norm in component_grads.items():
                        if component != 'total':
                            relative_metrics[f"grad_contribution/{component}"] = grad_norm / total_grad
                    
                    if relative_metrics:
                        trainer.logger.log_metrics(relative_metrics, step=trainer.global_step)
                        
        except Exception as e:
            logger.warning(f"Failed to compute component gradients: {e}")
    
    def _log_gradient_histograms(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Log gradient histograms to W&B."""
        
        import wandb
        
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                # Log gradient histogram
                trainer.logger.experiment.log({
                    f"gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())
                }, step=trainer.global_step)
    
    # Removed on_train_epoch_end - using step-based training
    def _disabled_on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Summarize gradient statistics - disabled for step-based training."""
        
        if not self.gradient_history:
            return
            
        logger.info("\n" + "="*60)
        logger.info("Gradient Statistics Summary for Epoch")
        logger.info("="*60)
        
        # Summarize layer gradients
        if self.gradient_history:
            logger.info("\nLayer Gradient Norms (average over epoch):")
            for layer_name, grad_norms in sorted(self.gradient_history.items()):
                if grad_norms:
                    avg_norm = sum(grad_norms) / len(grad_norms)
                    max_norm = max(grad_norms)
                    min_norm = min(grad_norms)
                    logger.info(
                        f"  {layer_name}: avg={avg_norm:.4f}, "
                        f"max={max_norm:.4f}, min={min_norm:.4f}"
                    )
        
        # Summarize component gradients
        if self.component_gradient_history:
            logger.info("\nLoss Component Gradient Norms (average over epoch):")
            for component, grad_norms in sorted(self.component_gradient_history.items()):
                if grad_norms:
                    avg_norm = sum(grad_norms) / len(grad_norms)
                    logger.info(f"  {component}: {avg_norm:.4f}")
        
        # Clear history for next epoch
        self.gradient_history.clear()
        self.component_gradient_history.clear()

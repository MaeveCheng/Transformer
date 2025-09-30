#!/usr/bin/env python3.10
"""
Unified Training Script for Order Book Transformer

Zero-config usage:
    python3.10 scripts/train.py

CLI usage:
    python3.10 scripts/train.py fit --max-epochs 50 --learning-rate 0.001
    python3.10 scripts/train.py test --checkpoint best.ckpt
    python3.10 scripts/train.py tune  # Auto-tune LR and batch size

All config values can be overridden via CLI:
    python3.10 scripts/train.py --batch-size 64 --precision 32 --devices 2
"""

import sys
import os
import argparse
import datetime
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import atexit
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
    LearningRateMonitor,
    DeviceStatsMonitor
)
from pytorch_lightning.plugins.environments import ClusterEnvironment, LightningEnvironment
from lightning_fabric.utilities.distributed import _InfiniteBarrier
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.strategies import DDPStrategy
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as default,
    powerSGD_hook as powerSGD,
)

# Set up multiprocessing for CUDA
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# DDP debugging imports removed

from config import Config
from data.randomized_datamodule import RandomizedDataModule
from models.transformer import OrderBookTransformer
from lightning.module import OrderBookLightningModule
from lightning.callbacks import (
    PowerSGDCallback,
    CustomTQDMProgressBar,
    TrainingVisualizationCallback,
    DistributionVerificationCallback,
    GradientLoggerCallback
)
from utils.memory_optimizer import MemoryManagementCallback

# Load configuration early to get logging level
config = Config()

# Apply PyTorch backend settings from config
if hasattr(config, 'pytorch_backend'):
    torch.backends.cuda.matmul.allow_tf32 = config.pytorch_backend.get('allow_tf32', True)
    torch.backends.cudnn.allow_tf32 = config.pytorch_backend.get('cudnn_allow_tf32', True)
    if 'cudnn_benchmark' in config.pytorch_backend:
        torch.backends.cudnn.benchmark = config.pytorch_backend.cudnn_benchmark
else:
    # Fallback to defaults for backward compatibility
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Map string level to logging constant
level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}
log_level = level_map.get(config.logging.level.upper(), logging.INFO)

# Setup logging with level from config
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables for better NCCL cleanup and stability
os.environ['NCCL_DEBUG'] = 'WARN'  # Only show NCCL warnings and errors
os.environ['NCCL_IB_DISABLE'] = '0'

# Additional settings for multi-node stability
if 'WORLD_SIZE' in os.environ and 'LOCAL_WORLD_SIZE' in os.environ:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    num_nodes_env = world_size // local_world_size if local_world_size > 0 else 1
    
    if num_nodes_env >= 3:
        # For 3+ nodes, use more conservative NCCL settings
        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Disable tree algorithm
        os.environ['NCCL_IB_TIMEOUT'] = '22'  # Increase IB timeout
        os.environ['NCCL_P2P_LEVEL'] = 'LOC'  # Limit P2P to local node
        os.environ['NCCL_CROSS_NIC'] = '0'  # Disable cross-NIC communication
        logger.info(f"Applied conservative NCCL settings for {num_nodes_env}-node setup")


# Detect if we're using isolated GPUs (simulating multi-node on single machine)
if 'CUDA_VISIBLE_DEVICES' in os.environ and 'RANK' in os.environ:
    # When using CUDA_VISIBLE_DEVICES to isolate GPUs, disable P2P
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
    # Note: We don't disable SHM as it can still be useful for same-machine communication
    print("Detected isolated GPU setup - disabling NCCL P2P and IB")
else:
    os.environ['NCCL_P2P_DISABLE'] = '0'  # Enable P2P for better performance with multiple GPUs


def check_checkpoint_exists(checkpoint_path: str) -> None:
    """Check that checkpoint file exists."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def get_best_checkpoint_path(trainer: pl.Trainer) -> Optional[str]:
    """Get best model checkpoint path from trainer callbacks."""
    if trainer.checkpoint_callbacks:
        for callback in trainer.checkpoint_callbacks:
            if hasattr(callback, 'best_model_path') and callback.best_model_path:
                return callback.best_model_path
    return None


def get_log_directory(trainer: pl.Trainer, config: Config) -> Path:
    """Get log directory from trainer logger or config."""
    # Try to get from logger
    if trainer.logger:
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        for lg in loggers:
            if hasattr(lg, 'log_dir') and lg.log_dir:
                return Path(lg.log_dir)
    
    # Fallback to config
    return Path(config.experiment.log_dir) / config.experiment.experiment_name


def parse_devices(devices_str: str):
    """Parse devices string into appropriate format."""
    if devices_str in ['auto', '-1']:
        return 'auto'
    
    # Try parsing as single integer
    try:
        return int(devices_str)
    except ValueError:
        # Try parsing as comma-separated list
        try:
            return [int(d.strip()) for d in devices_str.split(',')]
        except ValueError:
            raise ValueError(f"Invalid devices format: {devices_str}. Use 'auto', '-1', single int, or comma-separated ints.")


def parse_args():
    """Parse command line arguments with support for all config overrides."""
    parser = argparse.ArgumentParser(
        description="Order Book Transformer Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main action
    parser.add_argument(
        'action',
        nargs='?',
        default='fit',
        choices=['fit', 'test', 'tune', 'lr_find'],
        help='Action to perform'
    )
    
    # Core training arguments
    parser.add_argument('--experiment-name', type=str, help='Experiment name for logging')
    # Removed --max-epochs argument, using step-based training only
    parser.add_argument('--checkpoint', '--ckpt-path', type=str, help='Checkpoint path for resume/test')  # Keep both for backward compatibility
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # Model arguments
    parser.add_argument('--learning-rate', '--lr', type=float, help='Learning rate')
    parser.add_argument('--model-dim', type=int, help='Model dimension')
    parser.add_argument('--n-heads', type=int, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, help='Data directory path')
    parser.add_argument('--num-workers', type=int, help='Number of data workers')
    
    # Training arguments
    parser.add_argument('--precision', type=str, choices=['16', '32', '64', 'bf16', '16-true', '32-true', '64-true'], 
                       help='Training precision (PyTorch Lightning 2.0+)')
    parser.add_argument('--devices', type=str, help='Number of devices or device IDs (e.g., "2" or "0,1")')
    parser.add_argument('--num_nodes', type=int, help='Number of machines')
    parser.add_argument('--accumulate-grad-batches', type=int, help='Gradient accumulation steps')
    parser.add_argument('--gradient-clip-val', type=float, help='Gradient clipping value')
    parser.add_argument('--strategy', type=str, help='Training strategy (ddp, ddp_spawn, etc.)')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw', 'sgd', 'lamb'], 
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, 
                       choices=['cosine', 'onecycle', 'step', 'exponential', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--weight-decay', type=float, help='Weight decay')
    
    # Callback arguments
    parser.add_argument('--save-top-k', type=int, help='Number of best checkpoints to save')
    parser.add_argument('--swa', action='store_true', help='Enable Stochastic Weight Averaging')
    parser.add_argument('--auto-lr-find', action='store_true', 
                       help='Automatically find optimal learning rate')
    
    # Logging arguments
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, help='W&B project name')
    parser.add_argument('--log-every-n-steps', type=int, help='Logging frequency')
    parser.add_argument('--profiler', choices=['simple', 'advanced', 'pytorch'], 
                       help='Enable profiling')
    
    # Step-based training arguments
    parser.add_argument('--max-steps', type=int, 
                       help='Maximum training steps (default: use config value)')
    
    # Development/debugging
    parser.add_argument('--fast-dev-run', type=int, help='Run N batches for debugging')
    parser.add_argument('--overfit-batches', type=float, help='Overfit on subset of batches')
    parser.add_argument('--limit-train-batches', type=float, help='Limit training data percentage')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    parser.add_argument('--benchmark', action='store_true', help='Enable cudNN benchmarking')
    
    # Additional options
    parser.add_argument('--notes', type=str, help='Notes about this run')
    parser.add_argument('--tags', nargs='+', help='Tags for this run')
    parser.add_argument('--test-after-training', action='store_true', 
                       help='Run test after training completes')
    parser.add_argument('--clear-optimizer-state', action='store_true',
                       help='Clear optimizer state when resuming from checkpoint (fresh optimizer, keep model weights)')
    
    # DDP debugging options
    parser.add_argument('--debug-ddp', action='store_true',
                       help='Enable DDP debugging mode')
    
    # CPU preprocessing options
    parser.add_argument('--cpu-preprocessing', action='store_true',
                       help='Use CPU preprocessing pipeline (recommended for multi-node)')
    parser.add_argument('--cpu-workers', type=int, 
                       help='Number of CPU workers for preprocessing')
    parser.add_argument('--enable-distributed', action='store_true',
                       help='Enable distributed training optimizations')
    parser.add_argument('--distributed-cache', type=str,
                       help='Path for distributed cache (e.g., shared NFS mount)')
    
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for conflicts."""
    # Check for conflicting debugging/production settings
    if args.fast_dev_run and args.max_steps:
        logger.warning("Both --fast-dev-run and --max-steps specified. --fast-dev-run will take precedence.")
    
    # Check for action-specific requirements
    if args.action == 'test' and not args.checkpoint:
        raise ValueError("--checkpoint is required for test action")
    
    # Check for conflicting auto-tune settings with explicit values
    if args.auto_lr_find and args.learning_rate:
        logger.warning("Both --auto-lr-find and --learning-rate specified. Auto LR find will override the manual setting.")


def update_config(config: Config, args: argparse.Namespace) -> Config:
    """Update config with command line arguments."""
    # Only update if argument was explicitly provided
    if args.experiment_name is not None:
        config.experiment.experiment_name = args.experiment_name
    if args.seed is not None:
        config.seed = args.seed
        
    # Model arguments
    if args.learning_rate is not None:
        config.optimization.learning_rate = args.learning_rate
    if args.model_dim is not None:
        config.model.d_model = args.model_dim
    if args.n_heads is not None:
        config.model.n_heads = args.n_heads
    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
        
    # Data arguments
    if args.data_dir is not None:
        config.data.train_folder = args.data_dir
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
        
    # Optimization arguments
    if args.optimizer is not None:
        config.optimization.optimizer = args.optimizer
    if args.scheduler is not None:
        config.optimization.scheduler = args.scheduler
    if args.weight_decay is not None:
        config.optimization.weight_decay = args.weight_decay
    if args.gradient_clip_val is not None:
        config.training.gradient_clip_val = args.gradient_clip_val
        
    # Experiment/logging
    if args.use_wandb:
        config.experiment.use_wandb = True
    if args.wandb_project is not None:
        config.experiment.wandb_project = args.wandb_project
    if args.notes is not None:
        config.experiment.notes = args.notes
    if args.tags is not None:
        config.experiment.tags = args.tags
    
    # Step-based training settings
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    
    # CPU preprocessing settings
    if args.cpu_workers is not None:
        config.data.cpu_preprocessing_workers = args.cpu_workers
        
    if args.distributed_cache is not None:
        config.data.distributed_cache_path = args.distributed_cache
    
    # Distributed training optimizations
    if args.enable_distributed:
        # Auto-detect number of nodes from environment
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        num_nodes = world_size // local_world_size if local_world_size > 0 else 1
        
        config.enable_distributed_training(num_nodes)
        logger.info(f"Enabled distributed training optimizations for {num_nodes} nodes")
            
    return config


def get_callbacks(config: Config, args: argparse.Namespace) -> list:
    """Get all callbacks based on configuration and CLI arguments."""
    callbacks = []
    
    # 1. Progress bar
    if config.training.enable_progress_bar:
        callbacks.append(CustomTQDMProgressBar())
    
    # 2. Model checkpointing
    if config.training.enable_checkpointing:
        # Use checkpoint settings from config
        checkpoint_dir_name = config.training.get('checkpoint_dir_name', 'checkpoints')
        checkpoint_dir = os.path.join(
            config.experiment.log_dir, 
            config.experiment.experiment_name,
            checkpoint_dir_name
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get checkpoint settings from config with defaults
        save_top_k = config.training.get('checkpoint_save_top_k', 10000)
        save_interval = config.training.get('checkpoint_save_interval', 500)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='step_{step:06d}',
            save_top_k=save_top_k,
            monitor='step',
            mode='max',
            save_last=True,
            verbose=True,
            save_on_train_epoch_end=True,
            every_n_train_steps=save_interval,
            enable_version_counter=False
        )
        
        # Override save_top_k if specified via CLI
        if args.save_top_k is not None:
            checkpoint_callback.save_top_k = args.save_top_k
        
        callbacks.append(checkpoint_callback)
    
    # 3. Learning rate monitoring (always enabled)
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # 4. Device stats monitoring (always enabled)
    callbacks.append(DeviceStatsMonitor())
    
    # 5. Stochastic Weight Averaging - disabled for step-based training
    # SWA is inherently epoch-based in PyTorch Lightning and not compatible with step-only training
    if False:  # Disabled: was config.training.use_swa or args.swa
        pass  # SWA requires epoch boundaries which we've removed
    
    # 6. Memory management callback
    if config.memory_optimization.auto_batch_size_on_oom or config.memory_optimization.gradient_checkpointing:
        callbacks.append(MemoryManagementCallback(
            enable_gradient_checkpointing=config.memory_optimization.gradient_checkpointing,
            profile_memory=False,
            auto_batch_size=config.memory_optimization.auto_batch_size_on_oom,
            min_batch_size=config.memory_optimization.min_batch_size,
            memory_fraction=config.memory_optimization.memory_fraction
        ))
    
    # 7. Training visualization callback
    if config.visualization.enable_training_plots:
        callbacks.append(TrainingVisualizationCallback(
            plot_interval=config.visualization.plot_interval,
            save_plots_to_disk=config.visualization.save_plots_to_disk,
            plot_dir=config.visualization.plot_dir,
            histogram_bins=config.visualization.plot_histogram_bins,
            log_to_tensorboard=config.visualization.log_to_tensorboard,
            max_samples_per_plot=config.visualization.max_samples_per_plot
        ))
    
    # 8. Distribution verification callback
    if config.distribution_verification.enable_verification:
        callbacks.append(DistributionVerificationCallback(
            log_interval=config.distribution_verification.verification_interval,
            variance_warning_threshold=config.distribution_verification.variance_warning_threshold,
            log_to_tensorboard=config.distribution_verification.log_to_tensorboard,
            verbose=config.distribution_verification.verbose
        ))
    
    # 9. Gradient logger callback
    if config.gradient_tracking.use_gradient_logger_callback:
        callbacks.append(GradientLoggerCallback(
            log_every_n_steps=config.gradient_tracking.gradient_logger_interval,
            log_component_gradients=config.gradient_tracking.enable_component_tracking,
            log_layer_gradients=config.gradient_tracking.log_layer_gradients,
            log_gradient_histograms=config.gradient_tracking.log_gradient_histograms,
            detect_anomalies=config.gradient_tracking.detect_gradient_anomalies,
            anomaly_threshold=config.gradient_tracking.gradient_anomaly_threshold,
            track_gradient_flow=config.gradient_tracking.track_gradient_flow
        ))
    
    # Add PowerSGD callback if enabled
    if config.training.use_powersgd:
        callbacks.append(PowerSGDCallback(config))
        logger.info("Added PowerSGD callback for gradient compression")
    
    
    # Deadlock detection removed
    
    return callbacks


def count_model_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters and return statistics."""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
        else:
            frozen_params += num_params
            
    return {
        'total': total_params,
        'trainable': trainable_params, 
        'frozen': frozen_params,
        'total_mb': total_params * 4 / (1024 * 1024),  # Assuming FP32
        'trainable_mb': trainable_params * 4 / (1024 * 1024)
    }


def get_logger(config: Config, args: argparse.Namespace):
    """Get logger(s) based on configuration."""
    loggers = []
    
    # Ensure log directory is absolute
    log_dir = Path(config.experiment.log_dir).absolute()
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=config.experiment.experiment_name,
        log_graph=False,
        default_hp_metric=True
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger
    if config.experiment.use_wandb:
        wandb_logger = WandbLogger(
            project=config.experiment.wandb_project,
            name=config.experiment.experiment_name,
            save_dir=str(log_dir),
            log_model=True,
            notes=config.experiment.notes,
            tags=config.experiment.tags,
        )
        loggers.append(wandb_logger)
    
    # Return single logger or list
    return loggers[0] if len(loggers) == 1 else loggers


def get_profiler(profiler_name: Optional[str]):
    """Get profiler based on name."""
    if profiler_name == 'simple':
        return SimpleProfiler()
    elif profiler_name == 'advanced':
        return AdvancedProfiler()
    elif profiler_name == 'pytorch':
        return 'pytorch'
    return None


def find_latest_checkpoint(log_dir: str, experiment_name: str) -> Optional[str]:
    """Find the latest checkpoint to resume from.
    
    Args:
        log_dir: Base log directory
        experiment_name: Experiment name
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(log_dir) / experiment_name / 'checkpoints'
    if not checkpoint_dir.exists():
        return None
        
    # Look for last.ckpt first
    last_ckpt = checkpoint_dir / 'last.ckpt'
    if last_ckpt.exists():
        return str(last_ckpt)
        
    # Find latest checkpoint by modification time
    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
        
    return None


def main():
    """Main training function with CLI support."""
    global config
    
    args = parse_args()
    
    # Validate arguments
    check_args(args)
    
    # Setup DDP debugging if requested
    if args.debug_ddp:
        logger.info("Enabling DDP debugging...")
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Early distributed environment check for multi-node setups
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        node_rank = int(os.environ.get('NODE_RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        num_nodes = world_size // local_world_size if local_world_size > 0 else 1
        
        logger.info(f"Distributed environment detected:")
        logger.info(f"  Global rank: {rank}/{world_size}")
        logger.info(f"  Node rank: {node_rank}/{num_nodes}")
        logger.info(f"  Local rank: {local_rank}/{local_world_size}")
        logger.info(f"  Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")
        
        # For 3+ node setups, add pre-initialization delay to avoid race conditions
        if num_nodes >= 3 and node_rank > 0:
            import time
            startup_delay = node_rank * 5  # 5 seconds per node rank
            logger.info(f"Node {node_rank}: Waiting {startup_delay}s before initialization (3+ node setup)")
            time.sleep(startup_delay)
    
    # Detect isolated GPU setup for multi-node training
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_torchrun and 'CUDA_VISIBLE_DEVICES' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        num_nodes_early = world_size // local_world_size if local_world_size > 0 else 1
        
        if num_nodes_early > 1 and torch.cuda.device_count() == 1:
            # Multi-node with isolated GPUs - disable NCCL P2P and IB
            logger.info("Detected isolated GPU setup - disabling NCCL P2P and IB")
            os.environ['NCCL_P2P_DISABLE'] = '1'
            os.environ['NCCL_IB_DISABLE'] = '1'
            
            # Check if we can reach the master address
            master_addr = os.environ.get('MASTER_ADDR', '')
            master_port = os.environ.get('MASTER_PORT', '')
            if master_addr and master_port:
                logger.info(f"Multi-node setup with master at {master_addr}:{master_port}")
                
                # Test connectivity to master
                import socket
                try:
                    # Test hostname resolution first
                    resolved_addr = socket.gethostbyname(master_addr)
                    logger.info(f"Master hostname {master_addr} resolves to {resolved_addr}")
                    
                    # Test actual connectivity
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    sock.connect((resolved_addr, int(master_port)))
                    sock.close()
                    logger.info(f"Successfully connected to master at {master_addr}:{master_port}")
                except socket.gaierror as e:
                    logger.error(f"Failed to resolve hostname {master_addr}: {e}")
                    logger.error("Check /etc/hosts or DNS configuration on all nodes")
                except socket.timeout:
                    logger.error(f"Connection to {master_addr}:{master_port} timed out")
                    logger.error("Check firewall rules and network connectivity")
                except Exception as e:
                    logger.warning(f"Network test failed: {e}")
            
            # Let NCCL auto-detect interfaces unless user has specified
            if 'NCCL_SOCKET_IFNAME' in os.environ:
                logger.info(f"Using user-specified NCCL_SOCKET_IFNAME: {os.environ['NCCL_SOCKET_IFNAME']}")
            else:
                logger.info("Letting NCCL auto-detect network interfaces")
            
    config = update_config(config, args)
    
    # Set CUDA device early if using distributed training
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        # When using CUDA_VISIBLE_DEVICES, PyTorch renumbers the devices
        # So we always use local_rank directly
        if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            logger.info(f"Set CUDA device to {local_rank} for distributed training")
    
    # Set seed for reproducibility
    if config.seed is not None:
        pl.seed_everything(config.seed)
    
    # Log key settings
    logger.info(f"Action: {args.action}")
    logger.info(f"Experiment: {config.experiment.experiment_name}")
    
    # Log profile information
    logger.info(f"Profiles: {len(config.profiles)} configured")
    #for profile in config.profiles:
    #    logger.info(f"  - {profile.name}: weight={profile.weight:.2f}, folder={profile.train_folder}")
    
    # Validate data directories for all profiles
    for profile in config.profiles:
        if not Path(profile.train_folder).exists():
            raise FileNotFoundError(f"Data directory not found for profile '{profile.name}': {profile.train_folder}")

    # Create directories
    Path(config.experiment.log_dir).mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # Create data module
    logger.info(f"Setting up data module... rank: {local_rank}")
    
    # Use randomized data loading system
    logger.info("Using randomized data loading system")
    data_module = RandomizedDataModule(
        data_config=config.data,
        profiles=config.profiles,  # Pass profiles for multi-profile training
        training_config=config.training,  # Pass training config for date range parameters
    )
    
    # Get feature dimensions without calling setup
    # (Lightning will call setup automatically at the right time)
    feature_dim = data_module.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")
    # Log profile information
    # for profile in config.profiles:
        # logger.info(f"Profile '{profile.name}': batch_size={profile.batch_size}, seq_length={profile.seq_length}, weight={profile.weight:.2f}")
    
    # Create model
    # n_features is calculated from feature_columns, not a config field
    config.model.n_features = feature_dim
    # Use the maximum seq_length across all profiles to ensure model can handle any profile
    # max_sequence_length is calculated from profiles, not a config field
    max_seq_length = max(profile.seq_length for profile in config.profiles)
    config.model.max_sequence_length = max_seq_length
    logger.info(f"Model max_sequence_length set to {max_seq_length} (max across all profiles)")
    
    
    # Select model architecture based on configuration
    if getattr(config.model, 'use_hybrid_architecture', False):
        logger.info("Using Hybrid LSTM-Transformer architecture")
        from models.hybrid_lstm_transformer import HybridLSTMTransformer
        model = HybridLSTMTransformer(config.model, binary_classification_config=config.binary_classification)
    else:
        logger.info("Using standard Transformer architecture")
        model = OrderBookTransformer(config.model, binary_classification_config=config.binary_classification)
    
    # Create Lightning module
    # Note: class_weights will be None at this point since setup() hasn't been called
    # This is fine - the module can handle None class_weights
    lightning_module = OrderBookLightningModule(
        model=model,
        optimization_config=config.optimization,
        model_config=config.model,
        binary_classification_config=config.binary_classification,
        gradient_tracking_config=config.gradient_tracking,
        data_config=config.data,
        checkpoint_config=config.checkpoint,  # Pass checkpoint config for proper loading
        class_weights=None,  # Will be set later by Lightning hooks if needed
        metrics_display_interval=config.training.performance_log_interval,
        clear_optimizer_on_resume=args.clear_optimizer_state,  # Pass CLI flag for clearing optimizer
    )
    
    # Count and log model parameters
    param_stats = count_model_parameters(model)
    logger.info("Model created successfully")
    logger.info(f"Model Parameters:")
    logger.info(f"  Total: {param_stats['total']:,} ({param_stats['total_mb']:.2f} MB)")
    logger.info(f"  Trainable: {param_stats['trainable']:,} ({param_stats['trainable_mb']:.2f} MB)")
    if param_stats['frozen'] > 0:
        logger.info(f"  Frozen: {param_stats['frozen']:,}")
    
    # Note: Model memory pinning removed (handled automatically by PyTorch Lightning)
    # Auto-resume logic - always try to resume
    resume_checkpoint = args.checkpoint  # User-provided checkpoint
    
    if not resume_checkpoint and args.action == 'fit':
        # Try to find latest checkpoint
        resume_checkpoint = find_latest_checkpoint(
            config.experiment.log_dir, 
            config.experiment.experiment_name
        )
        if resume_checkpoint:
            logger.info(f"Auto-resuming from checkpoint: {resume_checkpoint}")
        else:
            logger.info("No checkpoint found, starting fresh training")
    
    # Get callbacks, logger, and profiler
    callbacks = get_callbacks(config, args)
    loggers = get_logger(config, args)
    profiler = get_profiler(args.profiler if args.profiler is not None else config.training.profiler)
    
    # Log TensorBoard directory information
    tb_log_dir = Path(config.experiment.log_dir).absolute() / config.experiment.experiment_name
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    logger.info(f"To view TensorBoard, run: tensorboard --logdir {tb_log_dir}")
    
    # Determine precision from CLI or config
    precision = args.precision if args.precision is not None else config.training.precision
    
    # Backward compatibility mapping for PyTorch Lightning 2.0+
    precision_mapping = {
        'bf16-mixed': 'bf16',
        '16-mixed': '16',
        '32-true': '32',
        '64-true': '64'
    }
    if precision in precision_mapping:
        logger.info(f"Converting legacy precision '{precision}' to '{precision_mapping[precision]}' for PyTorch Lightning 2.0+")
        precision = precision_mapping[precision]
    
    # Check if running under torchrun
    is_torchrun = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    # Determine num_nodes for torchrun
    if is_torchrun:
        # When using torchrun with --nnodes, we need to get the actual number of nodes
        # Your command: --nnodes=2
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        num_nodes = world_size // local_world_size if local_world_size > 0 else 1
        logger.info(f"Torchrun: Calculated num_nodes={num_nodes} from WORLD_SIZE={world_size} / LOCAL_WORLD_SIZE={local_world_size}")
    else:
        num_nodes = args.num_nodes if args.num_nodes is not None else 1
    
    # When using torchrun, we need to properly handle multi-node simulation
    # The key insight: with torchrun, each process gets its own rank
    # For multi-node simulation on a single machine, we use WORLD_SIZE processes
    if is_torchrun:
        logger.info(f"Torchrun environment detected:")
        logger.info(f"  RANK={os.environ.get('RANK')}")
        logger.info(f"  LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        logger.info(f"  WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        logger.info(f"  Visible GPUs: {torch.cuda.device_count()}")
    
    if args.devices is not None:
        devices = parse_devices(args.devices)
    elif is_torchrun:
        # For torchrun multi-node simulation:
        # Each node runs nproc_per_node processes
        # In your case: 2 nodes × 1 process/node = 2 total processes
        # Lightning expects devices to be the number of devices PER NODE
        devices = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        logger.info(f"Torchrun multi-node mode: devices={devices} (per node)")
    else:
        devices = config.training.devices
    
    # Determine strategy - use DDP with find_unused_parameters when using multiple GPUs
    if args.strategy is not None:
        strategy = args.strategy
    else:
        strategy = config.training.strategy

    # Check if using multiple GPUs
    is_multi_gpu = isinstance(devices, int) and devices > 1
    
    # If using multiple GPUs and strategy is 'auto' or 'ddp', use DDP with find_unused_parameters
    if is_multi_gpu or is_torchrun:
        if strategy in ['auto', 'ddp']:
            # Check if we're in a multi-node setup
            if num_nodes > 1:
                # Force NCCL for multi-node setups, even with single GPU per node
                backend = 'nccl'
                logger.info(f"Using NCCL backend for multi-node setup (num_nodes={num_nodes})")
                # Ensure master addr/port are set for multi-node
                if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
                    logger.info(f"Master address: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
                else:
                    logger.warning("MASTER_ADDR and/or MASTER_PORT not set in environment!")
            elif 'CUDA_VISIBLE_DEVICES' in os.environ and torch.cuda.device_count() == 1:
                # Use gloo backend only for single-node isolated GPU setup
                backend = 'nccl'
                logger.info("Using nccl backend for single-node isolated GPU simulation")
            else:
                # Use NCCL for normal multi-GPU setup
                backend = 'nccl'
                
            # Use custom cluster environment for torchrun multi-node
            
            # Adjust timeout based on number of nodes
            if num_nodes >= 3:
                ddp_timeout = datetime.timedelta(minutes=60)  # 1 hour for 3+ nodes
                logger.info(f"Using extended DDP timeout of {ddp_timeout} for {num_nodes}-node setup")
            else:
                ddp_timeout = datetime.timedelta(minutes=30)
                
            # Get static_graph setting with fallback for backward compatibility
            static_graph = getattr(config.training, 'ddp_static_graph', False)
            
            # Configure communication hooks based on settings
            if config.training.use_powersgd:
                logger.info(f"PowerSGD will be enabled after {config.training.powersgd_start_iter} iterations")
                logger.info(f"PowerSGD rank: {config.training.powersgd_rank}, error feedback: {config.training.powersgd_use_error_feedback}")
                
                # Create PowerSGD state
                from functools import partial
                
                # We'll configure the hook after model initialization
                ddp_comm_hook = None  # Will be set up later
                ddp_comm_state = None  
            elif config.training.use_fp16_compression:
                logger.info("Using FP16 gradient compression")
                ddp_comm_hook = default.fp16_compress_hook
                ddp_comm_state = None
            else:
                ddp_comm_hook = None
                ddp_comm_state = None
            
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,  # More efficient gradient synchronization
                static_graph=static_graph,  # Use config value for static graph optimization
                ddp_comm_state=False,  # Disable communication state logging to reduce overhead
                ddp_comm_hook=ddp_comm_hook,  # Communication hook for gradient compression
                process_group_backend=backend,
                # Add timeout for debugging
                timeout=ddp_timeout,
                # cluster_environment=cluster_env,
            )
            logger.info(f"Using DDP strategy with {backend} backend (static_graph={static_graph})")
            if is_torchrun:
                logger.info(f"Detected torchrun environment: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
    
    # Create trainer with Lightning parameters from CLI and config
    logger.info("Creating PyTorch Lightning Trainer...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"PyTorch Lightning version: {pl.__version__}")
    
    # Determine accelerator
    accelerator = config.training.accelerator
    if accelerator == 'gpu':
        if torch.cuda.is_available():
            accelerator = 'cuda'  # Use 'cuda' instead of 'gpu' for newer PyTorch Lightning versions
        else:
            logger.error("GPU requested but CUDA is not available!")
            accelerator = 'cpu'
    
    logger.info(f"Using accelerator: {accelerator}, devices: {devices}, strategy: {type(strategy).__name__}")

    trainer = pl.Trainer(
        # Basic settings from CLI or config
        max_epochs=999999,  # Single infinite epoch for step-based training
        max_steps=args.max_steps if args.max_steps is not None else config.training.max_steps,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        
        # Gradient settings
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val is not None else config.training.gradient_clip_val,
        gradient_clip_algorithm=config.training.gradient_clip_algorithm,
        accumulate_grad_batches=args.accumulate_grad_batches if args.accumulate_grad_batches is not None else config.training.accumulate_grad_batches,
        
        # Logging
        logger=loggers,
        log_every_n_steps=args.log_every_n_steps if args.log_every_n_steps is not None else config.training.log_every_n_steps,
        profiler=profiler,
        
        # Callbacks
        callbacks=callbacks,
        
        # Progress and summary
        enable_progress_bar=config.training.enable_progress_bar,
        enable_model_summary=config.training.enable_model_summary,
        
        # Performance optimizations
        benchmark=args.benchmark if args.benchmark is not None else config.training.benchmark,
        
        # Checkpointing (now enabled in DDP mode - Lightning handles it properly)
        enable_checkpointing=config.training.enable_checkpointing,
        
        # Development/debugging features
        fast_dev_run=args.fast_dev_run if args.fast_dev_run is not None else config.training.fast_dev_run,
        overfit_batches=args.overfit_batches if args.overfit_batches is not None else config.training.overfit_batches,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else config.training.limit_train_batches,
        
        # Reproducibility
        deterministic=args.deterministic if args.deterministic is not None else config.training.deterministic,
        
        # Other settings from config
        detect_anomaly=config.training.detect_anomaly,
    )
    
    logger.info("Trainer created successfully")
    
    # Save configuration
    logger.info("Getting log directory...")
    log_dir = get_log_directory(trainer, config)
    logger.info(f"Log directory: {log_dir}")
    
    logger.info("Creating config path...")
    config_path = log_dir / "config.json"
    
    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving configuration...")
    # Save config dict to JSON
    import json
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # NOTE: Do NOT call data_module.setup() here!
    # Lightning will call it automatically when the distributed environment is ready.
    # Calling it early can cause collective operation mismatches between ranks.
    
    # Execute action
    try:
        if args.action == 'tune':
            logger.info("Running hyperparameter tuning...")
            trainer.tune(lightning_module, datamodule=data_module)
            
        elif args.action == 'lr_find':
            logger.info("Running learning rate finder...")
            # In PyTorch Lightning 2.x, lr_find is a method of Tuner class
            from pytorch_lightning.tuner import Tuner
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(lightning_module, datamodule=data_module)
            suggested_lr = lr_finder.suggestion()
            logger.info(f"Suggested learning rate: {suggested_lr}")
            
        elif args.action == 'fit':
            # Auto-tune features
            # Use config default, but allow CLI to override
            tune_lr = args.auto_lr_find if args.auto_lr_find is not None else config.optimization.auto_lr_find
            
            # Auto learning rate finder
            if tune_lr:
                logger.info("Running automatic learning rate finder...")
                from pytorch_lightning.tuner import Tuner
                tuner = Tuner(trainer)
                lr_finder = tuner.lr_find(
                    lightning_module,
                    datamodule=data_module,
                    min_lr=1e-8,
                    max_lr=1,
                    num_training=100
                )
                # Get suggested learning rate
                suggested_lr = lr_finder.suggestion()
                if suggested_lr is not None:
                    # Update config before optimizer is created
                    config.optimization.learning_rate = suggested_lr
                    lightning_module.optimization_config.learning_rate = suggested_lr
                    lightning_module.hparams['learning_rate'] = suggested_lr
                    logger.info(f"Learning rate tuned to: {suggested_lr}")
                else:
                    logger.warning("Could not find optimal learning rate, using default")
            
            logger.info("Starting training...")
            # Validate checkpoint if resuming
            if resume_checkpoint:
                check_checkpoint_exists(resume_checkpoint)
                logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            
            # Add synchronization barrier for distributed training
            # Critical for ensuring all ranks are ready before training starts
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                node_rank = int(os.environ.get('NODE_RANK', 0))
                
                logger.info(f"Rank {rank}/{world_size} (Node {node_rank}, Local {local_rank}): Entering pre-training barrier...")
                logger.info(f"Rank {rank}: Backend={torch.distributed.get_backend()}")
                
                # Add debugging info
                logger.info(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR')}")
                logger.info(f"Rank {rank}: MASTER_PORT={os.environ.get('MASTER_PORT')}")
                logger.info(f"Rank {rank}: WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
                
                # For 3+ node setups, add extra timeout and retries
                num_nodes = world_size // int(os.environ.get('LOCAL_WORLD_SIZE', 1))
                
                # Longer timeout for more nodes
                if num_nodes >= 3:
                    barrier_timeout = datetime.timedelta(minutes=10)
                    logger.info(f"Rank {rank}: Using extended timeout for {num_nodes}-node setup")
                    
                    # Add small delay for higher-ranked nodes to avoid thundering herd
                    if node_rank > 0:
                        import time
                        delay = node_rank * 2  # 2 seconds per node rank
                        logger.info(f"Rank {rank}: Node {node_rank} waiting {delay}s before barrier")
                        time.sleep(delay)
                else:
                    barrier_timeout = datetime.timedelta(minutes=5)
                
                try:
                    # Set environment for better debugging
                    os.environ['NCCL_BLOCKING_WAIT'] = '1'
                    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
                    
                    # For 3+ nodes, ensure NCCL uses TCP sockets
                    if num_nodes >= 3:
                        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand for reliability
                        os.environ['NCCL_TREE_THRESHOLD'] = '0'  # Disable tree algorithm
                    
                    start_time = datetime.datetime.now()
                    torch.distributed.barrier()
                    end_time = datetime.datetime.now()
                    
                    logger.info(f"Rank {rank}: Barrier completed in {(end_time - start_time).total_seconds():.2f} seconds")
                except Exception as e:
                    logger.error(f"Rank {rank}: Pre-training barrier failed: {e}")
                    logger.error(f"Rank {rank}: This often indicates that not all {world_size} processes have started")
                    logger.error(f"Rank {rank}: Check that all {num_nodes} nodes are running with correct parameters")
                    # Don't continue - this is a critical error
                    raise
            
            logger.info("About to start trainer.fit()...")

            # DO NOT call setup on datamodule explicitly - let Lightning handle it after distributed init
            # This was causing the ranks to be incorrect during setup
            
            # Add debugging for distributed environment
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                logger.info(f"Rank {rank}: Before trainer.fit() - distributed initialized")
            else:
                logger.info("Before trainer.fit() - distributed NOT yet initialized (Lightning will handle it)")
            
            try:
                trainer.fit(
                    lightning_module,
                    datamodule=data_module,
                    ckpt_path=resume_checkpoint  # Use auto-resume checkpoint or user-provided
                )
            except Exception as e:
                logger.error(f"Error in trainer.fit(): {type(e).__name__}: {str(e)}")
                # Clean up distributed training on error
                if torch.distributed.is_initialized():
                    try:
                        torch.distributed.destroy_process_group()
                    except:
                        pass
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            logger.info("Training completed successfully!")

            # Note: Distributed cleanup is handled by cleanup_handler at exit
            # Removing duplicate destroy_process_group call to prevent errors

            
            # Optional: Run test after training
            if args.test_after_training:
                logger.info("Running test evaluation...")
                # Use best checkpoint if available
                best_path = get_best_checkpoint_path(trainer)
                if best_path:
                    logger.info(f"Testing with best checkpoint: {best_path}")
                else:
                    logger.info("Testing with current model weights (no best checkpoint found)")
                # Reload model from checkpoint before testing
                test_results = trainer.test(datamodule=data_module, ckpt_path=best_path)
                
        elif args.action == 'test':
            check_checkpoint_exists(args.checkpoint)
            logger.info(f"Testing model from: {args.checkpoint}")
            trainer.test(
                lightning_module,
                datamodule=data_module,
                ckpt_path=args.checkpoint
            )
            
            
    except KeyboardInterrupt:
        logger.info(f"{args.action} interrupted by user")
        # Cleanup will be handled by cleanup_handler
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid value: {str(e)}")
        raise
    except RuntimeError as e:
        error_msg = str(e)
        # Handle common distributed training errors gracefully
        if any(err in error_msg for err in ["ProcessGroupNCCL", "TCPStore", "Broken pipe", "Connection reset"]):
            logger.warning(f"Distributed training error during {args.action} (this is often harmless during shutdown): {error_msg}")
            # For distributed errors, ensure we still log the best checkpoint
            if args.action == 'fit' and 'trainer' in locals():
                best_path = get_best_checkpoint_path(trainer)
                if best_path:
                    logger.info(f"Best model checkpoint: {best_path}")
        else:
            logger.error(f"Runtime error during {args.action}: {error_msg}")
            raise
    
    # Log best checkpoint path
    if args.action == 'fit':
        best_path = get_best_checkpoint_path(trainer)
        if best_path:
            logger.info(f"Best model checkpoint: {best_path}")


if __name__ == "__main__":
    
    # Simplified cleanup logic - let PyTorch Lightning handle most cleanup
    def signal_handler(signum, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        
        # For DDP, let Lightning handle cleanup
        # Just exit normally and let Python cleanup
        sys.exit(0)
    
    # Enable signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        main()
    except SystemExit:
        # SystemExit is normal - cleanup handled by atexit
        raise
    except Exception as e:
        # Let PyTorch Lightning handle cleanup
        logger.error(f"Training failed with error: {e}")
        raise

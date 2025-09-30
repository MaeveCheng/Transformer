"""
Randomized DataModule for fully randomized data loading.

Replaces DistributedDataModule with complete randomization support.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import logging
import socket
import multiprocessing as mp
import random
import pyarrow.parquet as pq

# Import DictConfig from config module
from config import DictConfig
from .simplified_random_dataset import SimplifiedRandomDataset
from .cpu_window_creator import CPUWindowCreator
from .cpu_preprocessor import CPUPreprocessorCompat
from .distributed_collate import DistributedCollateFunction
from .pre_normalization_filter import PreNormalizationFilter

logger = logging.getLogger(__name__)


class RandomizedDataModule(pl.LightningDataModule):
    """
    Randomized data module with full randomization support.
    
    Features:
    - Complete randomization at every level
    - CPU-based preprocessing pipeline
    - Cross-file reading support
    - Distributed training support
    """
    
    def __init__(
        self,
        data_config,
        profiles: List,  # Required: Profile configurations for multi-profile support
        training_config=None,  # Optional: Training configuration for date range parameters
        # Randomization settings (can be overridden by data_config)
        randomize_files: bool = True,
        randomize_chunks: bool = True,
        **kwargs
    ):
        """
        Initialize randomized data module.
        
        Args:
            data_config: Data configuration
            profiles: List of profile configurations for multi-profile support
            training_config: Optional training configuration containing date range parameters
            randomize_files: Whether to randomize file selection
            randomize_chunks: Whether to randomize chunk start positions
        """
        super().__init__()
        self.data_config = data_config
        # chunking_config removed - each profile has its own chunk_size_lines
        self.training_config = training_config
        
        # Profiles are always provided by Config class (it handles legacy conversion)
        self.profiles = profiles
        
        # Use values from data_config if available, otherwise use defaults (for logging)
        self.randomize_files = getattr(data_config, 'randomize_files', randomize_files)
        self.randomize_chunks = getattr(data_config, 'randomize_chunks', randomize_chunks)
        
        # Distributed training info
        self.rank = None
        self.world_size = None
        self.local_world_size = None
        
        # Profile-specific storage
        self.profile_files = {}  # {profile_name: [files]}
        self.profile_configs = {}  # {profile_name: merged_config}
        
        # For backward compatibility
        self.all_files = []
        self.train_files = []
        
        logger.info(f"RandomizedDataModule initialized with: "
                   f"randomize_files={self.randomize_files}, "
                   f"randomize_chunks={self.randomize_chunks}")
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension from data configuration."""
        if self.data_config.feature_columns:
            return len(self.data_config.feature_columns)
        else:
            raise ValueError("feature_columns must be configured in the data section of config.json5")
    
    def setup(self, stage: Optional[str] = None):
        """Setup data module"""
        # Get distributed training info
        self._setup_distributed_info()
        
        # Discover data files
        self._discover_files()
        
        # Split files for train/val
        self._split_files()
        
        # Initialize preprocessing components
        self._setup_preprocessing()
        
        logger.info(f"Setup complete for rank {self.rank}/{self.world_size}")
    
    def _setup_distributed_info(self):
        """Get distributed training information"""
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        else:
            # Single process training
            self.rank = 0
            self.world_size = 1
            self.local_world_size = 1
    
    def _discover_files(self):
        """Discover files for all profiles"""
        total_files = 0
        
        for profile in self.profiles:
            pattern = os.path.join(profile.train_folder, profile.file_pattern)
            files = sorted(glob.glob(pattern))
            
            if not files:
                raise ValueError(f"No files found for profile {profile.name}: {pattern}")
            
            self.profile_files[profile.name] = files
            total_files += len(files)
            #logger.info(f"Profile '{profile.name}': Found {len(files)} files")
        
        # For backward compatibility - combine all files
        self.all_files = []
        for files in self.profile_files.values():
            self.all_files.extend(files)
        
        logger.info(f"Total files across all profiles: {total_files}")
    
    def _split_files(self):
        """Split files into train/val sets"""
        # For multi-profile, we keep profile-specific file lists
        # Train files and val files are now managed per profile
        
        # For backward compatibility
        self.train_files = self.all_files
        
        logger.info(f"Using all files for both training and validation")
    
    def _setup_preprocessing(self):
        """Prepare configurations for each profile (components will be initialized in dataset)"""
        # Create profile configurations
        for profile in self.profiles:
            # Merge data_config with profile (profile settings override)
            profile_config = DictConfig({**self.data_config, **profile})
            self.profile_configs[profile.name] = profile_config
        
        logger.info("Profile configurations prepared (components will be initialized in dataset)")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader with multi-profile support"""
        # Get date range parameters from training config if available
        training_config = getattr(self, 'training_config', None)
        train_date_start = None
        train_date_end = None
        date_group_size = 10
        date_group_lookback = 5
        iter_every_group = 100
        date_group_start_date = None
        
        if training_config:
            train_date_start = getattr(training_config, 'train_date_start', None)
            train_date_end = getattr(training_config, 'train_date_end', None)
            date_group_size = getattr(training_config, 'date_group_size', 10)
            date_group_lookback = getattr(training_config, 'date_group_lookback', 5)
            iter_every_group = getattr(training_config, 'iter_every_group', 100)
            date_group_start_date = getattr(training_config, 'date_group_start_date', None)
        
        dataset = SimplifiedRandomDataset(
            file_list=self.train_files,
            data_config=self.data_config,
            profiles=self.profiles,
            profile_files=self.profile_files,
            profile_configs=self.profile_configs,
            # Components will be initialized in dataset
            rank=self.rank,
            world_size=self.world_size,
            seed=None,  # True randomness for training
            train_date_start=train_date_start,
            train_date_end=train_date_end,
            date_group_size=date_group_size,
            date_group_lookback=date_group_lookback,
            iter_every_group=iter_every_group,
            date_group_start_date=date_group_start_date
        )
        
        # Create collate function
        # Key insight: Don't prefetch to GPU or use pinned memory when using workers
        # to avoid CUDA fork issues
        collate_fn = DistributedCollateFunction(
            use_pinned_memory=torch.cuda.is_available() and self.data_config.num_workers == 0,
            prefetch_to_gpu=torch.cuda.is_available() and self.data_config.num_workers == 0,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Create dataloader
        dataloader_kwargs = {
            'batch_size': None,  # Dataset returns batches
            'num_workers': self.data_config.num_workers,
            'pin_memory': False,  # Handled by collate function
            'persistent_workers': (
                self.data_config.persistent_workers 
                if self.data_config.num_workers > 0 else False
            ),
            'collate_fn': collate_fn,
        }
        
        # Add prefetch_factor if using multiprocessing
        if self.data_config.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = self.data_config.prefetch_factor
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up resources"""
        # Components are now managed in dataset, no cleanup needed here
        logger.info(f"Teardown complete for rank {self.rank}")

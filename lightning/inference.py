"""
Lightning-based inference library for Order Book Transformer.

This module provides inference utilities for use by scripts/inference.py:
- LightningInferenceDataset: Dataset for single-file inference
- InferenceDataModule: DataModule for batch processing
- InferenceOptimizationCallback: Callback for inference optimization
- LightningInference: Main class that handles all inference operations

This is a library module - use scripts/inference.py as the CLI entry point.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from types import SimpleNamespace
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import gc
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import pyarrow.parquet as pq

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from data.cpu_preprocessor import CPUPreprocessorCompat
from data.simplified_random_dataset import SimplifiedRandomDataset
from data.sequential_inference_dataset import SequentialInferenceDataset
from lightning.module import OrderBookLightningModule

logger = logging.getLogger(__name__)


class LightningInferenceDataset(Dataset):
    """Lightning-compatible dataset for inference on order book data.
    
    Features:
    - Sliding window support
    - Optional preprocessing
    - Metadata tracking
    - Memory-efficient data handling
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray, List[str]],
        window_size: int,
        stride: int,
        preprocessor: Optional[CPUPreprocessorCompat] = None,
        feature_columns: Optional[List[str]] = None,
        include_metadata: bool = True
    ):
        """
        Initialize Lightning inference dataset.
        
        Args:
            data: Order book data (DataFrame, numpy array, or list of file paths)
            window_size: Size of sliding window
            stride: Stride for sliding window
            preprocessor: CPUPreprocessorCompat for data normalization
            feature_columns: Specific feature columns to use
            include_metadata: Whether to include metadata in batch
        """
        self.window_size = window_size
        self.stride = stride
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns
        self.include_metadata = include_metadata
        
        # Handle different data types
        if isinstance(data, list):
            # List of file paths - lazy loading
            self.file_paths = data
            self.data_array = None
            self._setup_lazy_loading()
        else:
            # Direct data loading
            self.file_paths = None
            if isinstance(data, pd.DataFrame):
                self.data_array = data.values
                self.columns = data.columns.tolist()
            else:
                self.data_array = data
                self.columns = None
            
            # Calculate number of windows
            self.num_samples = (len(self.data_array) - window_size) // stride + 1
    
    def _setup_lazy_loading(self):
        """Setup for lazy loading from files."""
        # Count total samples across all files
        self.file_info = []
        total_rows = 0
        
        for file_path in self.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            num_rows = parquet_file.metadata.num_rows
            self.file_info.append({
                'path': file_path,
                'start_row': total_rows,
                'end_row': total_rows + num_rows,
                'num_rows': num_rows
            })
            total_rows += num_rows
        
        self.total_rows = total_rows
        self.num_samples = (total_rows - self.window_size) // self.stride + 1
        
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # Get window data
        if self.file_paths:
            window_data = self._load_window_from_files(start_idx, end_idx)
        else:
            window_data = self.data_array[start_idx:end_idx]
        
        # Apply preprocessing if available
        if self.preprocessor and hasattr(self.preprocessor, 'transform'):
            if isinstance(window_data, np.ndarray) and self.columns:
                window_df = pd.DataFrame(window_data, columns=self.columns)
            else:
                window_df = window_data
            window_data = self.preprocessor.transform(window_df)
            if isinstance(window_data, pd.DataFrame):
                window_data = window_data.values
        
        # Create batch dictionary
        batch = {
            'features': torch.tensor(window_data, dtype=torch.float32),
        }
        
        # Add metadata if requested
        if self.include_metadata:
            batch['metadata'] = {
                'window_start': start_idx,
                'window_end': end_idx,
                'window_idx': idx
            }
        
        return batch
    
    def _load_window_from_files(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Load a window of data spanning multiple files."""
        window_data = []
        
        for file_info in self.file_info:
            # Skip files that don't overlap with our window
            if start_idx >= file_info['end_row'] or end_idx <= file_info['start_row']:
                continue
                
            # Calculate file-relative indices
            file_start = max(0, start_idx - file_info['start_row'])
            file_end = min(file_info['num_rows'], end_idx - file_info['start_row'])
            
            # Load relevant portion
            df = pd.read_parquet(
                file_info['path'],
                columns=self.feature_columns
            ).iloc[file_start:file_end]
            
            window_data.append(df.values)
        
        return np.vstack(window_data) if window_data else np.array([])


class InferenceDataModule(pl.LightningDataModule):
    """Lightning DataModule for inference.
    
    Handles:
    - Data loading and preprocessing
    - Batch creation for inference
    - Memory-efficient processing
    - Multi-file support
    """
    
    def __init__(
        self,
        data_source: Union[str, pd.DataFrame, List[str]],
        config: Config,
        preprocessor: Optional[CPUPreprocessorCompat] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        use_chunked: bool = True,
        sequential_mode: bool = False,
        stride: Optional[int] = None,
        overlap_ratio: float = 0.0
    ):
        """
        Initialize inference DataModule.
        
        Args:
            data_source: Input data (file path, DataFrame, or list of files)
            config: Configuration object
            preprocessor: Data preprocessor
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            use_chunked: Whether to use chunked loading for large files
            sequential_mode: Whether to use sequential processing
            stride: Stride for sequential mode
            overlap_ratio: Overlap ratio for sequential chunks
        """
        super().__init__()
        self.data_source = data_source
        self.config = config
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_chunked = use_chunked
        self.sequential_mode = sequential_mode
        self.stride = stride
        self.overlap_ratio = overlap_ratio
        
        # Ensure we have feature_columns available
        if not hasattr(self.config.data, 'feature_columns'):
            # If feature_columns not present but feature_categories is, generate them
            if hasattr(self.config.data, 'feature_categories'):
                self._generate_feature_columns_from_categories()
            else:
                logger.warning("No feature_columns or feature_categories in config.data")
                # Try to get from config if it has a method
                if hasattr(self.config, '_generate_feature_columns'):
                    self.config._generate_feature_columns()
                else:
                    raise ValueError("Cannot find or generate feature_columns")
    
    def _generate_feature_columns_from_categories(self):
        """Generate feature_columns from feature_categories following the same logic as Config."""
        if not hasattr(self.config.data, 'feature_categories'):
            raise ValueError("No feature_categories to generate columns from")
        
        # Define category order (same as in config.py)
        CATEGORY_ORDER = ['time', 'price', 'volume', 'count', 'spread', 'volume_imbalance', 'meta', 'window', 'other']
        
        feature_columns = []
        seen_columns = set()
        
        # Process categories in order
        for category in CATEGORY_ORDER:
            if category in self.config.data.feature_categories:
                for col in self.config.data.feature_categories[category]:
                    if col in seen_columns:
                        logger.warning(f"Duplicate column '{col}' found in category '{category}' - skipping")
                        continue
                    feature_columns.append(col)
                    seen_columns.add(col)
        
        # Handle any new categories not in CATEGORY_ORDER
        for category in self.config.data.feature_categories:
            if category not in CATEGORY_ORDER:
                logger.warning(f"Unknown category '{category}' found - adding at end")
                for col in self.config.data.feature_categories[category]:
                    if col not in seen_columns:
                        feature_columns.append(col)
                        seen_columns.add(col)
        
        # Set the generated columns
        self.config.data.feature_columns = feature_columns
        logger.info(f"Generated {len(feature_columns)} feature columns from categories")
        
    def _create_dataset(self) -> Dataset:
        """Create appropriate dataset based on data source type."""
        # Handle parquet files with chunked loading
        if self.use_chunked and self._is_parquet_source():
            parquet_files = [self.data_source] if isinstance(self.data_source, str) else self.data_source
            
            # For inference, we need to ensure data_config has the necessary attributes
            # Use the first profile's settings if available
            if hasattr(self.config, 'profiles') and self.config.profiles:
                first_profile = self.config.profiles[0]
                # Add profile attributes to data_config for legacy mode compatibility
                self.config.data.seq_length = first_profile.seq_length
                self.config.data.predict_start = first_profile.predict_start
                self.config.data.predict_end = first_profile.predict_end
                # min_stride and max_stride removed - use command line --stride parameter
                self.config.data.batch_size = first_profile.batch_size
                self.config.data.exponential_lookback = first_profile.exponential_lookback
                logger.info(f"Using profile '{first_profile.name}' settings for inference")
                
                # Create window creator for the dataset
                from data.cpu_window_creator import CPUWindowCreator
                import numpy as np
                
                # Create sample offsets (hybrid exponential)
                sequential_samples = 50  # Default value
                exponential_factor = 2.5  # Default value
                seq_length = first_profile.seq_length
                exponential_lookback = first_profile.exponential_lookback
                
                # First samples: sequential
                initial_offsets = np.arange(0, sequential_samples)
                
                # Remaining samples: exponential spacing
                remaining_samples = seq_length - sequential_samples
                if remaining_samples > 0:
                    positions = np.linspace(0, 1, remaining_samples)
                    exp_positions = np.exp(exponential_factor * positions) - 1
                    exp_positions = exp_positions / exp_positions[-1]
                    
                    start_pos = sequential_samples
                    offsets_exp = start_pos + exp_positions * (exponential_lookback - start_pos)
                    offsets_exp = np.round(offsets_exp).astype(np.int64)
                    
                    # Ensure minimum spacing
                    for i in range(1, len(offsets_exp)):
                        if offsets_exp[i] <= offsets_exp[i-1]:
                            offsets_exp[i] = offsets_exp[i-1] + 1
                    
                    sample_offsets = np.concatenate([initial_offsets, offsets_exp])
                else:
                    sample_offsets = initial_offsets[:seq_length]
                
                # Ensure we don't exceed maximum lookback
                sample_offsets = np.minimum(sample_offsets, exponential_lookback)
                
                window_creator = CPUWindowCreator(
                    sample_offsets=sample_offsets,
                    feature_dim=len(self.config.data.feature_columns),
                    predict_start=first_profile.predict_start,
                    predict_end=first_profile.predict_end,
                    batch_size=first_profile.batch_size,
                    num_workers=1
                )
            else:
                window_creator = None
            
            # Create profile and profile files for inference
            # Use first profile settings if available, otherwise create a default
            if hasattr(self.config, 'profiles') and self.config.profiles:
                profiles = self.config.profiles[:1]  # Use only first profile
                profile_files = {profiles[0].name: parquet_files}
                # Create a profile config that mimics data_config structure
                # Include feature_columns since SimplifiedRandomDataset needs it
                profile_config_obj = SimpleNamespace(
                    feature_columns=self.config.data.feature_columns,  # SimplifiedRandomDataset needs this
                    feature_categories=self.config.data.feature_categories,
                    **{k: v for k, v in vars(self.config.data).items() if k not in ['feature_columns', 'feature_categories']},
                    preprocessor=self.preprocessor,
                    window_creator=window_creator
                )
                profile_configs = {profiles[0].name: profile_config_obj}
            else:
                # Create a minimal profile for inference
                default_profile = SimpleNamespace(
                    name='inference',
                    seq_length=self.config.data.seq_length,
                    batch_size=self.batch_size,
                    predict_start=20,
                    predict_end=40,
                    min_batch=20,  # Default min batches per chunk
                    max_batch=30,  # Default max batches per chunk
                    # min_stride and max_stride removed - use command line --stride parameter
                    exponential_lookback=100,
                    chunk_size_lines=400000,  # Default chunk size for inference
                    adaptive_stats_db=None
                )
                profiles = [default_profile]
                profile_files = {'inference': parquet_files}
                # Create a profile config that mimics data_config structure
                # Include feature_columns since SimplifiedRandomDataset needs it
                profile_config_obj = SimpleNamespace(
                    feature_columns=self.config.data.feature_columns,  # SimplifiedRandomDataset needs this
                    feature_categories=self.config.data.feature_categories,
                    **{k: v for k, v in vars(self.config.data).items() if k not in ['feature_columns', 'feature_categories']},
                    preprocessor=self.preprocessor,
                    window_creator=window_creator
                )
                profile_configs = {'inference': profile_config_obj}
            
            # Use seed from config (which comes from experiment.seed)
            inference_seed = self.config.seed if hasattr(self.config, 'seed') else 42
            logger.info(f"Using seed={inference_seed} for inference dataset")
            
            # Use sequential dataset if in sequential mode
            if self.sequential_mode:
                return SequentialInferenceDataset(
                    file_list=parquet_files,
                    data_config=self.config.data,
                    chunking_config=self.config.chunking if hasattr(self.config, 'chunking') else SimpleNamespace(chunk_size_lines=4000),
                    profiles=profiles,
                    profile_files=profile_files,
                    profile_configs=profile_configs,
                    stride=self.stride,
                    overlap_ratio=self.overlap_ratio,
                    rank=0,
                    world_size=1,
                    seed=inference_seed,
                    model_config=self.config.model if hasattr(self.config, 'model') else None,
                    binary_classification_config=self.config.binary_classification if hasattr(self.config, 'binary_classification') else None
                )
            else:
                return SimplifiedRandomDataset(
                    file_list=parquet_files,
                    data_config=self.config.data,
                    profiles=profiles,
                    profile_files=profile_files,
                    profile_configs=profile_configs,
                    rank=0,
                    world_size=1,
                    seed=inference_seed,
                    model_config=self.config.model if hasattr(self.config, 'model') else None,
                    binary_classification_config=self.config.binary_classification if hasattr(self.config, 'binary_classification') else None
                )
        
        # Handle direct data sources
        if isinstance(self.data_source, pd.DataFrame):
            data = self.data_source
        elif isinstance(self.data_source, str) and self.data_source.endswith('.parquet'):
            data = pd.read_parquet(self.data_source)
        elif isinstance(self.data_source, list):
            data = self.data_source  # LightningInferenceDataset handles file lists
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
        
        return LightningInferenceDataset(
            data=data,
            window_size=self.config.data.seq_length,
            stride=self.stride if self.stride is not None else 1000,
            preprocessor=self.preprocessor
        )
    
    def _is_parquet_source(self) -> bool:
        """Check if data source is parquet file(s)."""
        if isinstance(self.data_source, str):
            return self.data_source.endswith('.parquet')
        elif isinstance(self.data_source, list):
            return all(f.endswith('.parquet') for f in self.data_source)
        return False
        
    def setup(self, stage: Optional[str] = None):
        """Setup data for inference."""
        if stage == "predict" or stage is None:
            self.predict_dataset = self._create_dataset()
    
    def predict_dataloader(self) -> DataLoader:
        """Create DataLoader for predictions."""
        def inference_collate_fn(batch):
            """Custom collate function to handle 4D tensor issues during inference."""
            if len(batch) == 0:
                return {}
            
            # Handle different batch structures
            if isinstance(batch[0], dict):
                # Check if the dataset is already returning batched data
                if 'features' in batch[0] and batch[0]['features'].dim() == 3:
                    # Data is already batched from CPUWindowCreator
                    # The dataset returns batches, not individual samples
                    # So we should concatenate, not stack
                    if len(batch) == 1:
                        # Single batch from dataset, just return it
                        return batch[0]
                    else:
                        # Multiple batches need to be concatenated
                        collated = {}
                        for key in batch[0].keys():
                            if key == 'features':
                                # Concatenate along batch dimension
                                features_list = [item[key] for item in batch]
                                features = torch.cat(features_list, dim=0)
                                collated[key] = features
                            elif key == 'labels':
                                # Concatenate labels
                                labels_list = [item[key] for item in batch]
                                collated[key] = torch.cat(labels_list, dim=0)
                            elif key == 'metadata':
                                # Combine metadata
                                collated[key] = [item[key] for item in batch]
                            else:
                                # Handle other tensors
                                values = [item[key] for item in batch]
                                if all(isinstance(v, torch.Tensor) for v in values):
                                    if values[0].dim() > 0:
                                        collated[key] = torch.cat(values, dim=0)
                                    else:
                                        collated[key] = torch.stack(values, dim=0)
                                else:
                                    collated[key] = values
                        return collated
                
                # Standard unbatched format - stack individual samples
                collated = {}
                for key in batch[0].keys():
                    if key == 'features':
                        # Stack features and ensure 3D tensor
                        features_list = [item[key] for item in batch]
                        features = torch.stack(features_list, dim=0)
                        
                        # Fix 4D tensor issue: if we get [batch, 1, seq_len, features], squeeze dim 1
                        if features.dim() == 4 and features.shape[1] == 1:
                            features = features.squeeze(1)
                        elif features.dim() == 4:
                            # If it's truly 4D (not squeezable), this is an error
                            raise ValueError(f"Unexpected 4D features tensor with shape {features.shape} - cannot squeeze dimension 1")
                        elif features.dim() != 3:
                            raise ValueError(f"Expected 3D features tensor after collation, got {features.dim()}D with shape {features.shape}")
                        
                        collated[key] = features
                    elif key == 'metadata':
                        # Combine metadata
                        collated[key] = [item[key] for item in batch]
                    else:
                        # Stack other tensors normally
                        values = [item[key] for item in batch]
                        if all(isinstance(v, torch.Tensor) for v in values):
                            collated[key] = torch.stack(values, dim=0)
                        else:
                            collated[key] = values
                return collated
            else:
                # Fallback to default collation
                return torch.utils.data.default_collate(batch)
        
        # Simplified logic: SequentialInferenceDataset always batches internally
        # For inference.py, we primarily use SequentialInferenceDataset with --sequential flag
        if isinstance(self.predict_dataset, SequentialInferenceDataset):
            # SequentialInferenceDataset batches internally using profile.batch_size
            dataloader_batch_size = 1
            drop_last = False  # Don't drop - SequentialInferenceDataset handles padding
            logger.info(f"  Using batch_size=1 in DataLoader (SequentialInferenceDataset batches internally)")
        elif isinstance(self.predict_dataset, SimplifiedRandomDataset) and hasattr(self.predict_dataset, 'window_creator'):
            # SimplifiedRandomDataset with window_creator also batches internally
            dataloader_batch_size = 1
            drop_last = False  # Don't drop - window_creator handles batching
            logger.info(f"  Using batch_size=1 in DataLoader (batching done by window_creator)")
        elif isinstance(self.predict_dataset, LightningInferenceDataset):
            # LightningInferenceDataset for single file inference (legacy mode)
            dataloader_batch_size = self.batch_size
            drop_last = False  # Don't drop last batch in inference
            logger.info(f"  Using batch_size={self.batch_size} in DataLoader (LightningInferenceDataset)")
        else:
            # Unknown dataset type - this should not happen
            dataset_type = type(self.predict_dataset).__name__
            error_msg = (
                f"Unexpected dataset type: {dataset_type}. "
                f"Expected one of: SequentialInferenceDataset, SimplifiedRandomDataset, or LightningInferenceDataset. "
                f"Cannot determine appropriate DataLoader configuration."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Creating DataLoader: batch_size={dataloader_batch_size}, num_workers={self.num_workers}, drop_last={drop_last}")
        logger.info(f"  Dataset type: {type(self.predict_dataset).__name__}")
        logger.info(f"  Sequential mode: {self.sequential_mode}")
        
        return DataLoader(
            self.predict_dataset,
            batch_size=dataloader_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=drop_last,  # Now controlled per dataset type
            collate_fn=inference_collate_fn
        )


class InferenceMetricsCallback(Callback):
    """Callback for tracking inference performance metrics."""
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize metrics callback.
        
        Args:
            log_interval: Interval for logging performance metrics
        """
        self.log_interval = log_interval
        self.batch_times = []
        self.batch_count = 0
        
    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Record batch start time."""
        self.batch_start_time = time.time()
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Record batch end time and log metrics."""
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        self.batch_count += 1
        
        # Clear GPU cache periodically to prevent OOM
        if self.batch_count % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log performance metrics periodically
        if self.batch_count % self.log_interval == 0:
            avg_time = np.mean(self.batch_times[-self.log_interval:])
            throughput = batch['features'].shape[0] / avg_time
            
            # Also log GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(
                    f"Batch {self.batch_count}: "
                    f"Avg time: {avg_time:.3f}s, "
                    f"Throughput: {throughput:.1f} samples/s, "
                    f"GPU Mem: {allocated:.1f}/{reserved:.1f}GB"
                )
            else:
                logger.info(
                    f"Batch {self.batch_count}: "
                    f"Avg time: {avg_time:.3f}s, "
                    f"Throughput: {throughput:.1f} samples/s"
                )


class LightningInference:
    """Main class for Lightning-based inference.
    
    Provides:
    - Unified interface for all inference types
    - Trainer.predict() integration
    - Performance optimization
    - Result post-processing
    """
    
    def __init__(
        self,
        config: Config,
        checkpoint_path: str,
        preprocessor_path: Optional[str] = None,
        device: str = "auto",
        precision: str = "bf16",
        sequential_mode: bool = False,
        stride: Optional[int] = None,
        overlap_ratio: float = 0.0,
        use_flash_attention: bool = True,  # Default to True
        use_gradient_checkpointing: bool = True,  # Default to True
        memory_efficient: bool = True  # Default to True
    ):
        """
        Initialize Lightning inference.
        
        Args:
            config: Configuration object
            checkpoint_path: Path to model checkpoint
            preprocessor_path: Path to saved preprocessor
            device: Device selection ("auto", "gpu", "cpu")
            precision: Precision for inference
            sequential_mode: Whether to use sequential processing
            stride: Stride for sequential mode (default: chunk_size)
            overlap_ratio: Overlap ratio for sequential chunks
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.precision = precision
        self.sequential_mode = sequential_mode
        self.stride = stride
        self.overlap_ratio = overlap_ratio
        self.use_flash_attention = use_flash_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.memory_efficient = memory_efficient
        
        # Set memory optimization settings
        if memory_efficient and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # Set memory allocation config to reduce fragmentation
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Reduce memory fragmentation
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
            logger.info("✓ Memory efficient mode enabled with expandable segments")
        
        logger.info(f"Inference optimizations:")
        logger.info(f"  - Precision: {self.precision}")
        logger.info(f"  - Flash attention: {self.use_flash_attention}")
        logger.info(f"  - Gradient checkpointing: {self.use_gradient_checkpointing}")
        logger.info(f"  - Memory efficient: {self.memory_efficient}")
        
        # Ensure feature_columns is generated before model loading
        if not hasattr(self.config.data, 'feature_columns'):
            if hasattr(self.config.data, 'feature_categories'):
                self._generate_feature_columns_from_categories()
            
        # Load preprocessor
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        
        # Setup trainer
        self.trainer = self._setup_trainer(device)
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"Lightning inference initialized on {self.trainer.accelerator}")
    
    def _generate_feature_columns_from_categories(self):
        """Generate feature_columns from feature_categories."""
        if not hasattr(self.config.data, 'feature_categories'):
            raise ValueError("No feature_categories to generate columns from")
        
        # Define category order (same as in config.py)
        CATEGORY_ORDER = ['time', 'price', 'volume', 'count', 'spread', 'volume_imbalance', 'meta', 'window', 'other']
        
        feature_columns = []
        seen_columns = set()
        
        # Process categories in order
        for category in CATEGORY_ORDER:
            if category in self.config.data.feature_categories:
                for col in self.config.data.feature_categories[category]:
                    if col in seen_columns:
                        logger.warning(f"Duplicate column '{col}' found in category '{category}' - skipping")
                        continue
                    feature_columns.append(col)
                    seen_columns.add(col)
        
        # Handle any new categories not in CATEGORY_ORDER
        for category in self.config.data.feature_categories:
            if category not in CATEGORY_ORDER:
                logger.warning(f"Unknown category '{category}' found - adding at end")
                for col in self.config.data.feature_categories[category]:
                    if col not in seen_columns:
                        feature_columns.append(col)
                        seen_columns.add(col)
        
        # Set the generated columns
        self.config.data.feature_columns = feature_columns
        logger.info(f"Generated {len(feature_columns)} feature columns from categories")
        
    def _load_preprocessor(self, preprocessor_path: Optional[str]) -> Optional[CPUPreprocessorCompat]:
        """Load preprocessor if path is provided."""
        if preprocessor_path and Path(preprocessor_path).exists():
            logger.info(f"Loading preprocessor from: {preprocessor_path}")
            preprocessor = CPUPreprocessorCompat()
            preprocessor.load_stats(preprocessor_path)
            return preprocessor
        return None
    
    def _setup_trainer(self, device: str) -> Trainer:
        """Setup Lightning Trainer for inference."""
        # Determine accelerator and devices
        if device == "auto":
            accelerator = "auto"
            devices = "auto"
        elif device == "cpu":
            accelerator = "cpu"
            devices = 1
        elif device in ["cuda", "gpu"]:
            # Handle both 'cuda' and 'gpu' inputs
            accelerator = "gpu"
            devices = 1
        else:
            # Default to auto
            accelerator = "auto"
            devices = "auto"
        
        # Create trainer with inference-optimized settings
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=self.precision,
            enable_progress_bar=True,
            enable_model_summary=False,
            callbacks=[InferenceMetricsCallback()],
            logger=False,  # Disable Lightning logger for inference
            enable_checkpointing=False,
            inference_mode=True  # Enable inference optimizations
        )
        
        logger.info(f"✓ Trainer configured: accelerator={accelerator}, devices={devices}, precision={self.precision}")
        
        return trainer
    
    def _load_model(self) -> OrderBookLightningModule:
        """Load model from checkpoint."""
        logger.info(f"Loading model from: {self.checkpoint_path}")
        
        # Load checkpoint once
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        
        # Extract hyperparameters
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Create model instance from hyperparameters
        model_config = self.config.model
        if 'model_config' in hparams:
            # Update model config from checkpoint
            saved_config = hparams['model_config']
            # Handle both dict and dataclass formats
            if hasattr(saved_config, '__dict__'):
                saved_config = saved_config.__dict__
            elif not isinstance(saved_config, dict):
                saved_config = {}
            
            # Validate critical dimensions - don't override if mismatch
            if 'input_dim' in saved_config:
                checkpoint_input_dim = saved_config['input_dim']
                if hasattr(self.config.data, 'feature_columns'):
                    expected_input_dim = len(self.config.data.feature_columns)
                    if checkpoint_input_dim != expected_input_dim:
                        error_msg = f"Checkpoint input_dim ({checkpoint_input_dim}) does not match feature_columns length ({expected_input_dim}). Cannot proceed with inference."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                
                # Don't override model config if dimensions don't match
                if hasattr(model_config, 'input_dim') and model_config.input_dim != checkpoint_input_dim:
                    error_msg = f"Model config input_dim ({model_config.input_dim}) does not match checkpoint input_dim ({checkpoint_input_dim}). Cannot proceed with inference."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Only apply config from checkpoint if dimensions match
            for key, value in saved_config.items():
                if hasattr(model_config, key):
                    # Skip overriding if it's a dimension-related field and values don't match
                    if key in ['input_dim', 'n_features', 'input_features'] and getattr(model_config, key) != value:
                        error_msg = f"Config mismatch for {key}: model has {getattr(model_config, key)}, checkpoint has {value}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    setattr(model_config, key, value)
        
        # Extract config from checkpoint
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        else:
            hparams = {}
        
        # Import the actual transformer model
        from models.transformer import OrderBookTransformer
        
        # Validate model config has the correct input_dim
        if not hasattr(self.config.model, 'input_dim') or self.config.model.input_dim == 0:
            # Set input_dim based on feature_columns
            if hasattr(self.config.data, 'feature_columns'):
                self.config.model.input_dim = len(self.config.data.feature_columns)
                logger.info(f"Setting model input_dim to {self.config.model.input_dim} based on feature_columns")
            else:
                error_msg = "Could not determine input_dim from feature_columns - missing feature_columns in config"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Validate input_dim matches feature_columns if both exist
        if hasattr(self.config.data, 'feature_columns') and hasattr(self.config.model, 'input_dim'):
            expected_dim = len(self.config.data.feature_columns)
            if self.config.model.input_dim != expected_dim:
                error_msg = f"Model input_dim ({self.config.model.input_dim}) does not match feature_columns length ({expected_dim})"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Create transformer model with model config
        transformer = OrderBookTransformer(self.config.model)
        
        # Create lightning module
        model = OrderBookLightningModule(
            model=transformer,
            optimization_config=self.config.optimization,
            model_config=self.config.model,
            data_config=self.config.data
        )
        
        # Enable gradient checkpointing if requested (default: True)
        # Note: Gradient checkpointing is typically for training, might not be needed for inference
        if self.use_gradient_checkpointing and False:  # Disabled for now - may cause issues in inference
            if hasattr(transformer, 'gradient_checkpointing_enable'):
                transformer.gradient_checkpointing_enable()
                logger.info("✓ Gradient checkpointing enabled for inference")
            elif hasattr(transformer, 'enable_gradient_checkpointing'):
                transformer.enable_gradient_checkpointing()
                logger.info("✓ Gradient checkpointing enabled for inference")
        
        # Enable Flash Attention if requested (default: True)
        if self.use_flash_attention:
            self._enable_flash_attention(transformer)
        
        # Load state dict
        state_dict = checkpoint['state_dict']
        
        # Fix input projection dimension mismatch
        if 'model.embedding.input_projection.weight' in state_dict:
            expected_weight = state_dict['model.embedding.input_projection.weight']
            checkpoint_input_dim = expected_weight.shape[1]  # Get actual input dimension from checkpoint
            model_input_dim = model.model.embedding.input_projection.weight.shape[1]
            
            # Check if dimensions mismatch
            if model_input_dim != checkpoint_input_dim:
                error_msg = f"Input dimension mismatch: model has {model_input_dim} features, but checkpoint has {checkpoint_input_dim} features. Cannot load checkpoint."
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Now load the state dict
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        
        # Ensure no gradients are tracked
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info(f"✓ Model loaded and moved to device: {device}")
        logger.info(f"  Model is on CUDA: {next(model.parameters()).is_cuda}")
        logger.info(f"  Gradients disabled for inference")
        
        return model
    
    def _enable_flash_attention(self, model):
        """Enable Flash Attention in the model if available."""
        try:
            # Check if flash_attn is installed
            import flash_attn
            
            # Modify attention layers to use Flash Attention
            for name, module in model.named_modules():
                if 'MultiheadAttention' in module.__class__.__name__:
                    logger.info(f"Enabling Flash Attention for {name}")
                    # Set flash attention flag if the module supports it
                    if hasattr(module, 'use_flash_attention'):
                        module.use_flash_attention = True
            
            logger.info("✓ Flash Attention enabled")
        except ImportError:
            logger.warning("Flash Attention not available, using standard attention")
        
        """transformer_model = OrderBookTransformer(
            model_config,
            binary_classification_config=self.config.binary_classification
        )
        
        # Create Lightning module
        lightning_model = OrderBookLightningModule(
            model=transformer_model,
            optimization_config=self.config.optimization,
            model_config=model_config,
            binary_classification_config=self.config.binary_classification,
        )
        
        # Load the state dict
        lightning_model.load_state_dict(checkpoint['state_dict'])
        
        # Enable optimization
        lightning_model.eval()
        if hasattr(torch, 'compile') and str(self.trainer.accelerator).lower() == 'cuda':
            logger.info("Compiling model with torch.compile for faster inference")
            lightning_model = torch.compile(lightning_model, mode='reduce-overhead', backend='inductor')
        
        return lightning_model"""
    
    def predict(
        self,
        data_source: Union[str, pd.DataFrame, List[str]],
        batch_size: int = 32,
        num_workers: int = 0,
        return_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Run predictions using Lightning Trainer.
        
        Args:
            data_source: Input data source
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing aggregated predictions, probabilities, and optionally targets
        """
        # Clear cache before starting if memory_efficient is enabled
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ Memory cache cleared before inference")
            # Log GPU memory status
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory before inference: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
        
        # Create DataModule with sequential mode support
        datamodule = InferenceDataModule(
            data_source=data_source,
            config=self.config,
            preprocessor=self.preprocessor,
            batch_size=batch_size,
            num_workers=num_workers,
            use_chunked=True,  # Always use chunked processing for large files
            sequential_mode=self.sequential_mode,
            stride=self.stride,
            overlap_ratio=self.overlap_ratio
        )
        
        # Run predictions with optimization context
        logger.info("Starting Lightning predictions...")
        
        # Don't use torch.inference_mode() here since trainer already has inference_mode=True
        # Check if we should use autocast based on precision setting
        use_autocast = False
        dtype = None
        
        if torch.cuda.is_available():
            if self.precision in ['bf16', 'bfloat16']:
                use_autocast = True
                dtype = torch.bfloat16
            elif self.precision in ['16', 'float16', 'fp16']:
                use_autocast = True
                dtype = torch.float16
            # For 32-bit precision or other values, don't use autocast
            # This includes: '32', 'float32', 'fp32', '32-true', etc.
        
        if use_autocast and dtype is not None:
            logger.info(f"✓ Running inference with {dtype} precision (autocast)")
            # Use torch.amp.autocast for better compatibility
            with torch.amp.autocast('cuda', dtype=dtype):
                batch_predictions = self.trainer.predict(
                    self.model,
                    datamodule=datamodule,
                    return_predictions=return_predictions
                )
        else:
            # This handles fp32, cpu inference, or any other precision setting
            logger.info(f"Running inference with precision: {self.precision} (no autocast)")
            batch_predictions = self.trainer.predict(
                self.model,
                datamodule=datamodule,
                return_predictions=return_predictions
            )
        
        logger.info(f"Completed predictions for {len(batch_predictions)} batches")
        
        # Aggregate predictions from all batches
        all_predictions = []
        all_probabilities = []
        all_targets = []
        has_targets = False
        
        for batch_result in batch_predictions:
            # Extract predictions (logits) from batch
            logits = batch_result['predictions']
            
            # Convert from bfloat16 to float32 if needed
            if logits.dtype == torch.bfloat16:
                logits = logits.float()
            
            # Apply sigmoid for binary classification to get probabilities
            if logits.shape[-1] == 2:
                # Binary classification with 2 outputs (negative, positive)
                # Apply softmax to get probabilities for both classes
                probs = torch.softmax(logits, dim=-1)
                # Use the probability of the positive class (index 1)
                probabilities = probs[:, 1]
                predictions = (probabilities > 0.5).float()
            elif len(logits.shape) == 1 or logits.shape[-1] == 1:
                # Binary classification with single output
                probabilities = torch.sigmoid(logits.squeeze(-1))
                predictions = (probabilities > 0.5).float()
            else:
                # Multi-class
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1).float()
                # For multi-class, we need to handle probabilities differently
                # Just use the max probability for now
                probabilities = probabilities.max(dim=-1)[0]
            
            all_predictions.extend(predictions.cpu().float().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().float().numpy().tolist())
            
            # Check if targets are available in the batch
            if 'targets' in batch_result:
                targets = batch_result['targets'].cpu().float()
                # Squeeze targets if they have shape [batch_size, 1] to [batch_size]
                if len(targets.shape) == 2 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)
                all_targets.extend(targets.numpy().tolist())
                has_targets = True
        
        # Build result dictionary
        result = {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities)
        }
        
        if has_targets:
            result['targets'] = np.array(all_targets)
        
        # Memory cleanup after predictions if memory_efficient is enabled
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("✓ Memory cache cleared after inference")
        
        return result
    
    def predict_files(
        self,
        file_paths: List[str],
        batch_size: int = 32,
        save_results: bool = True,
        output_dir: str = "./predictions"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run predictions on multiple files.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Batch size for inference
            save_results: Whether to save results
            output_dir: Directory for saving results
            
        Returns:
            Dictionary mapping file paths to predictions
        """
        all_predictions = {}
        
        for file_path in tqdm(file_paths, desc="Processing files"):
            logger.info(f"Processing: {file_path}")
            
            # Run predictions
            predictions = self.predict(
                data_source=file_path,
                batch_size=batch_size,
                return_predictions=True
            )
            
            all_predictions[file_path] = predictions
            
            # Save if requested
            if save_results:
                output_path = Path(output_dir) / f"{Path(file_path).stem}_predictions.pt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(predictions, output_path)
                logger.info(f"Saved predictions to: {output_path}")
        
        return all_predictions
    
    def benchmark(
        self,
        sample_data: Union[pd.DataFrame, np.ndarray],
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            sample_data: Sample data for benchmarking
            batch_sizes: List of batch sizes to test
            num_iterations: Number of iterations per batch size
            
        Returns:
            Benchmark results
        """
        results = {
            'device': str(self.trainer.accelerator),
            'precision': self.precision,
            'benchmarks': {}
        }
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size: {batch_size}")
            
            # Create temporary DataModule
            datamodule = InferenceDataModule(
                data_source=sample_data,
                config=self.config,
                preprocessor=self.preprocessor,
                batch_size=batch_size,
                num_workers=0
            )
            
            # Warmup
            _ = self.trainer.predict(self.model, datamodule, return_predictions=False)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.trainer.predict(self.model, datamodule, return_predictions=False)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_iterations
            
            # Calculate metrics
            total_samples = len(sample_data) if hasattr(sample_data, '__len__') else batch_size
            throughput = total_samples / avg_time
            
            results['benchmarks'][f'batch_{batch_size}'] = {
                'avg_inference_time': avg_time,
                'throughput_samples_per_sec': throughput,
                'total_time': total_time,
                'num_iterations': num_iterations
            }
        
        return results



"""
JSON5-based configuration system.

This module loads all configuration from a JSON5 file.
No defaults or fallbacks - all values must be in the JSON5 file.

Usage:
    from config import Config
    config = Config()  # Loads from config.json5 or TRAINING_CONFIG_PATH env var
    config = Config("custom_config.json5")  # Load custom config
"""

import json5
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DictConfig(dict):
    """Configuration dict that supports both dict-style and attribute access."""
    
    def __init__(self, data=None):
        """Initialize from dict."""
        if data is None:
            data = {}
        super().__init__(data)
    
    def __getattr__(self, name):
        """Get attribute from dict."""
        if name in self:
            value = self[name]
            # Recursively convert nested dicts
            if isinstance(value, dict) and not isinstance(value, DictConfig):
                value = DictConfig(value)
                self[name] = value
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Set attribute in dict."""
        self[name] = value
    
    def to_dict(self):
        """Convert to regular dict (for compatibility)."""
        return dict(self)


class Config:
    """Configuration loaded exclusively from JSON5 file."""
    
    def _deep_merge_dict(self, base: dict, override: dict) -> dict:
        """
        Deep merge override dictionary into base dictionary.
        
        Args:
            base: Base dictionary
            override: Override dictionary
            
        Returns:
            Merged dictionary (modifies base in-place and returns it)
        """
        for key, value in override.items():
            if key in base:
                if isinstance(base[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    self._deep_merge_dict(base[key], value)
                else:
                    # Override the value
                    base[key] = value
            else:
                # Add new key
                base[key] = value
        return base
    
    def _load_portfolio_profiles(self):
        """Load profiles from portfolio folder."""
        # Determine portfolio path relative to config file
        if os.path.isabs(self.json5_path):
            config_dir = Path(os.path.dirname(self.json5_path))
        else:
            config_dir = Path(__file__).parent
        
        portfolio_path = config_dir / 'portfolio'
        
        if not portfolio_path.exists():
            logger.warning(f"Portfolio folder not found at: {portfolio_path}")
            return
        
        # Load all .json5 files from portfolio folder
        profiles = []
        json5_files = sorted(portfolio_path.glob('*.json5'))
        
        if not json5_files:
            logger.warning(f"No .json5 files found in portfolio folder: {portfolio_path}")
            return
        
        logger.info(f"Loading profiles from portfolio folder: {portfolio_path}")
        
        files_processed = 0
        multi_profile_files = []
        
        for profile_file in json5_files:
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json5.load(f)
                    
                    # Support both single profile (dict) and multiple profiles (list)
                    if isinstance(profile_data, list):
                        # Multiple profiles in one file
                        profiles.extend(profile_data)
                        multi_profile_files.append(profile_file.name)
                        logger.debug(f"Loaded {len(profile_data)} profiles from {profile_file.name}")
                    elif isinstance(profile_data, dict):
                        # Check if this is the new format with _base and profiles
                        if '_base' in profile_data and 'profiles' in profile_data:
                            # New format: merge base config with each profile
                            base_config = profile_data['_base']
                            for profile in profile_data['profiles']:
                                # Merge base config with profile-specific config
                                merged_profile = {**base_config, **profile}
                                profiles.append(merged_profile)
                            multi_profile_files.append(profile_file.name)
                            logger.debug(f"Loaded {len(profile_data['profiles'])} profiles with base config from {profile_file.name}")
                        else:
                            # Single profile (backward compatibility)
                            profiles.append(profile_data)
                            logger.debug(f"Loaded 1 profile from {profile_file.name}")
                    else:
                        logger.warning(f"Unexpected data type in {profile_file.name}: {type(profile_data)}")
                        continue
                    
                    files_processed += 1
            except Exception as e:
                logger.error(f"Error loading profile {profile_file}: {e}")
                continue
        
        if profiles:
            self._data['profiles'] = profiles
            logger.info(f"Loaded {len(profiles)} profiles from {files_processed} files in portfolio folder")
            if multi_profile_files:
                logger.info(f"Files with multiple profiles: {', '.join(multi_profile_files)}")
    
    def __init__(self, json5_path: Optional[str] = None, inference_mode: bool = False):
        """
        Load configuration from JSON5 file.
        
        Args:
            json5_path: Path to JSON5 configuration file.
                       If not provided, checks TRAINING_CONFIG_PATH env var,
                       then defaults to 'config.json5'.
            inference_mode: If True, skip validation of training-specific paths.
        """
        self.inference_mode = inference_mode
        # Determine config path
        if json5_path is None:
            json5_path = os.environ.get('TRAINING_CONFIG_PATH', 'config.json5')
        
        # If config path is relative and doesn't exist, try from project root
        if not os.path.isabs(json5_path) and not os.path.exists(json5_path):
            project_root = Path(__file__).parent
            potential_path = project_root / json5_path
            if potential_path.exists():
                json5_path = str(potential_path)
        
        self.json5_path = json5_path
        
        # Load JSON5 configuration
        if not os.path.exists(json5_path):
            raise FileNotFoundError(
                f"Configuration file not found: {json5_path}\n"
                f"Please create a configuration file or set TRAINING_CONFIG_PATH environment variable."
            )
        
        logger.info(f"Loading configuration from: {json5_path}")
        
        with open(json5_path, 'r') as f:
            self._data = json5.load(f)
        
        # Load profiles from portfolio folder if it exists and no profiles in main config
        if 'profiles' not in self._data or len(self._data.get('profiles', [])) == 0:
            self._load_portfolio_profiles()
        
        # Check for config.dev.json5 and merge if exists
        dev_config_path = None
        if os.path.isabs(json5_path):
            # If absolute path, check in the same directory
            dev_config_path = os.path.join(os.path.dirname(json5_path), 'config.dev.json5')
        else:
            # Check in the same directory as the main config
            project_root = Path(__file__).parent
            dev_config_path = project_root / 'config.dev.json5'
            if not dev_config_path.exists():
                # Also check in current directory
                dev_config_path = Path('config.dev.json5')
        
        if dev_config_path and os.path.exists(dev_config_path):
            logger.info(f"Loading development overrides from: {dev_config_path}")
            with open(dev_config_path, 'r') as f:
                dev_data = json5.load(f)
                # Deep merge dev config into main config
                self._deep_merge_dict(self._data, dev_data)
                logger.info(f"Applied configuration overrides from config.dev.json5")
        
        # Create configuration sections
        self._create_sections()
        
        # Validate configuration
        self.validate()
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Experiment: {self.experiment.experiment_name}")
    
    def _create_sections(self):
        """Create configuration sections as DictConfig objects."""
        # Handle profiles - new multi-profile support
        if 'profiles' in self._data:
            # Multi-profile mode - filter out profiles with weight 0
            all_profiles = [DictConfig(p) for p in self._data['profiles']]
            self.profiles = [p for p in all_profiles if p.weight > 0]
            
            # Ensure at least one profile has positive weight
            if not self.profiles:
                raise ValueError("At least one profile must have weight > 0")
            
            # Normalize weights for active profiles
            total_weight = sum(p.weight for p in self.profiles)
            for p in self.profiles:
                p.weight = p.weight / total_weight
                
            # For backward compatibility, create data section from first profile merged with common data
            common_data = self._data.get('data', {})
            first_profile_data = {k: v for k, v in dict(self.profiles[0]).items() 
                                  if k not in ['name', 'weight']}
            self.data = DictConfig({**first_profile_data, **common_data})
        else:
            # Legacy single-profile mode - convert to profiles format
            data_dict = self._data['data']
            profile = DictConfig({
                'name': 'default',
                'weight': 1.0,
                **data_dict
            })
            self.profiles = [profile]
            self.data = DictConfig(data_dict)
        
        # Auto-generate feature_columns from feature_categories
        self._build_feature_columns_from_categories()
        
        # Create other sections from JSON5 data - no defaults
        self.model = DictConfig(self._data['model'])
        
        # Add n_features to model config (calculated from feature columns)
        self.model.n_features = len(self.data.feature_columns)
        self.optimization = DictConfig(self._data['optimization'])
        self.training = DictConfig(self._data['training'])
        self.experiment = DictConfig(self._data['experiment'])
        self.binary_classification = DictConfig(self._data['binary_classification'])
        self.checkpoint = DictConfig(self._data['checkpoint'])
        self.logging = DictConfig(self._data['logging'])
        self.gradient_tracking = DictConfig(self._data['gradient_tracking'])
        # chunking removed - now each profile has its own chunk_size_lines
        self.memory_optimization = DictConfig(self._data['memory_optimization'])
        self.visualization = DictConfig(self._data['visualization'])
        self.distribution_verification = DictConfig(self._data['distribution_verification'])
        
        # Global seed if present
        if 'seed' in self._data:
            self.seed = self._data['seed']
        else:
            self.seed = self.experiment['seed']
    
    
    
    
    
    
    
    
    def _build_feature_columns_from_categories(self):
        """Build feature_columns from feature_categories with consistent ordering."""
        # Define category order - ensures consistent feature ordering across runs
        CATEGORY_ORDER = [
            'time',              # Time features first (date, seconds, etc.)
            'price',             # Price features (bid/ask prices, mid_price)
            'volume',            # Volume features (bid/ask quantities)
            'count',             # Order count features
            'spread',            # Spread-related features
            'volume_imbalance',  # Volume imbalance features
            'meta',              # Metadata (symbol, data_type)
            'window',            # Window features (predict_start, predict_end)
            'other'              # Uncategorized features
        ]
        
        if not hasattr(self.data, 'feature_categories'):
            raise ValueError("feature_categories must be configured in the data section of config.json5")
        
        feature_columns = []
        seen_columns = set()
        
        # Process categories in defined order
        for category in CATEGORY_ORDER:
            if category in self.data.feature_categories:
                for col in self.data.feature_categories[category]:
                    if col in seen_columns:
                        raise ValueError(f"Column '{col}' appears in multiple categories")
                    seen_columns.add(col)
                    feature_columns.append(col)
        
        # Handle any new categories not in CATEGORY_ORDER
        for category in self.data.feature_categories:
            if category not in CATEGORY_ORDER:
                logger.warning(f"Unknown category '{category}' found - adding at end")
                for col in self.data.feature_categories[category]:
                    if col not in seen_columns:
                        feature_columns.append(col)
                        seen_columns.add(col)
                    else:
                        raise ValueError(f"Column '{col}' appears in multiple categories")
        
        self.data.feature_columns = feature_columns
        logger.info(f"Auto-generated {len(feature_columns)} feature columns from categories")
    
    def get_n_features(self) -> int:
        """Get the number of features based on feature columns."""
        if hasattr(self.data, 'feature_columns') and self.data.feature_columns:
            return len(self.data.feature_columns)
        else:
            raise ValueError("feature_columns could not be generated from feature_categories")
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate profiles
        assert len(self.profiles) > 0, "At least one profile is required"
        
        # Track seq_lengths across profiles for warning
        seq_lengths = []
        
        for profile in self.profiles:
            profile_name = profile.get('name', 'unnamed')
            
            # Required fields validation
            required_fields = [
                'name', 'weight', 'file_pattern', 'batch_size', 'seq_length',
                'exponential_lookback', 'predict_start', 'predict_end', 'normalisation',
                'adaptive_stats_db'
            ]
            
            # Add train_folder to required fields only if not in inference mode
            if not self.inference_mode:
                required_fields.append('train_folder')
            
            for field in required_fields:
                assert field in profile, f"Required field '{field}' missing in profile {profile_name}"
            
            # Profile-specific validation
            assert profile['batch_size'] > 0, f"Batch size must be positive for profile {profile_name}"
            assert profile['seq_length'] > 0, f"Sequence length must be positive for profile {profile_name}"
            assert profile['weight'] >= 0, f"Weight must be non-negative for profile {profile_name}"
            assert profile['predict_start'] >= 0, f"predict_start must be non-negative for profile {profile_name}"
            assert profile['predict_end'] > profile['predict_start'], f"predict_end must be > predict_start for profile {profile_name}"
            assert profile['exponential_lookback'] > 0, f"exponential_lookback must be positive for profile {profile_name}"
            assert isinstance(profile['normalisation'], (int, float)), f"normalisation must be a number for profile {profile_name}"

            if not self.inference_mode:
                assert os.path.exists(profile.train_folder), f"Train folder does not exist for profile {profile_name}: {profile.train_folder}"
            # Only warn about missing train folders, don't fail (they may be remote paths)
            if not os.path.exists(profile.train_folder):
                logger.warning(f"Train folder does not exist for profile {profile_name}: {profile.train_folder}")
            
            # Normalization methods validation
            norm_methods = [
                'price_norm_method', 'volume_norm_method', 'count_norm_method'
            ]
            for method in norm_methods:
                assert method in profile, f"Required normalization method '{method}' missing in profile {profile_name}"
            seq_lengths.append(profile['seq_length'])
        
        # Warn if profiles have significantly different seq_lengths
        if len(set(seq_lengths)) > 1:
            min_seq = min(seq_lengths)
            max_seq = max(seq_lengths)
            if max_seq > min_seq * 1.5:  # More than 50% difference
                logger.warning(f"Profiles have significantly different seq_lengths: min={min_seq}, max={max_seq}")
                logger.warning("This may impact model performance. Consider using similar seq_lengths across profiles.")
        
        # Common data validation (for fields that remain in data section)
        if hasattr(self.data, 'num_workers'):
            assert self.data['num_workers'] >= 0, "Number of workers must be non-negative"
        # Model validation
        assert self.model['d_model'] % self.model['n_heads'] == 0, "d_model must be divisible by n_heads"
        
        # Ensure directories exist
        os.makedirs(self.experiment['log_dir'], exist_ok=True)
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self._data
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path='{self.json5_path}', experiment='{self.experiment['experiment_name']}')"

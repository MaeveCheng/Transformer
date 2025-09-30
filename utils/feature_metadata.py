"""Feature metadata for tracking feature categories through the pipeline."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import json


@dataclass
class FeatureMetadata:
    """Metadata about features and their categories.
    
    This class provides information about which features belong to which categories,
    enabling category-specific processing in the model.
    """
    # Feature names (using tuple for immutability and hashability)
    feature_names: tuple[str, ...]
    
    # Category indices (numpy arrays for efficient indexing)
    price_indices: np.ndarray
    volume_indices: np.ndarray
    count_indices: np.ndarray
    time_indices: np.ndarray
    window_indices: np.ndarray  # For predict_start/predict_end
    metadata_indices: np.ndarray  # For symbol/data_type
    other_indices: np.ndarray
    
    # Total feature count
    n_features: int
    
    # Optional: mid price index for special handling
    mid_price_index: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'feature_names': list(self.feature_names),  # Convert tuple to list for JSON serialization
            'price_indices': self.price_indices.tolist(),
            'volume_indices': self.volume_indices.tolist(),
            'count_indices': self.count_indices.tolist(),
            'time_indices': self.time_indices.tolist(),
            'window_indices': self.window_indices.tolist(),
            'metadata_indices': self.metadata_indices.tolist(),
            'other_indices': self.other_indices.tolist(),
            'n_features': self.n_features,
            'mid_price_index': self.mid_price_index
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureMetadata':
        """Create from dictionary."""
        return cls(
            feature_names=tuple(data['feature_names']),  # Convert list back to tuple
            price_indices=np.array(data['price_indices'], dtype=np.int64),
            volume_indices=np.array(data['volume_indices'], dtype=np.int64),
            count_indices=np.array(data['count_indices'], dtype=np.int64),
            time_indices=np.array(data['time_indices'], dtype=np.int64),
            window_indices=np.array(data.get('window_indices', []), dtype=np.int64),
            metadata_indices=np.array(data.get('metadata_indices', []), dtype=np.int64),
            other_indices=np.array(data['other_indices'], dtype=np.int64),
            n_features=data['n_features'],
            mid_price_index=data.get('mid_price_index')
        )
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> 'FeatureMetadata':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_category_for_index(self, idx: int) -> str:
        """Get the category name for a given feature index."""
        if idx in self.price_indices:
            return 'price'
        elif idx in self.volume_indices:
            return 'volume'
        elif idx in self.count_indices:
            return 'count'
        elif idx in self.time_indices:
            return 'time'
        elif idx in self.window_indices:
            return 'window'
        elif idx in self.metadata_indices:
            return 'metadata'
        elif idx in self.other_indices:
            return 'other'
        else:
            return 'unknown'
    
    def get_category_mask(self, category: str) -> np.ndarray:
        """Get a boolean mask for features belonging to a category."""
        mask = np.zeros(self.n_features, dtype=bool)
        
        if category == 'price':
            mask[self.price_indices] = True
        elif category == 'volume':
            mask[self.volume_indices] = True
        elif category == 'count':
            mask[self.count_indices] = True
        elif category == 'time':
            mask[self.time_indices] = True
        elif category == 'window':
            mask[self.window_indices] = True
        elif category == 'metadata':
            mask[self.metadata_indices] = True
        elif category == 'other':
            mask[self.other_indices] = True
            
        return mask
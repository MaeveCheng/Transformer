"""
Scripts package for Order Book Transformer.

This package contains the main executable scripts for:
- Training the model (train.py)
- Evaluating the model (evaluate.py)
- Running inference (inference.py)

SECTION-8: Evaluation & Inference scripts
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "Order Book Transformer Team"

# Get package directory
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
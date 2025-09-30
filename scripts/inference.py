#!/usr/bin/env python3.10
"""
Unified inference and evaluation script for Order Book Transformer model.

This script provides two main modes:
1. predict: Run inference on a single file or directory
   - Single file: generates predictions.parquet with evaluation
   - Directory: processes all parquet files with evaluation
   - Always evaluates when targets are available
   - Always saves predictions and plots

2. batch: Process multiple files individually with separate outputs
   - Each file gets its own prediction output with evaluation
   - Always evaluates when targets are available
   - Shows aggregate summary for all files
   - Always generates evaluation plots

Features:
- Simplified parquet-only data format
- Direct Config usage (no external config files)
- Auto-selected GPU strategy
- Standard ML metrics (accuracy, ROC, precision/recall)
- Automatic evaluation, prediction saving, and plot generation
- Consistent Lightning-based processing for all modes

SECTION-8: Evaluation & Inference
Dependencies: lightning.inference (LightningInference)
"""

# Standard library
import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Filter warnings early
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ML libraries
import torch
# Enable TF32 for better performance on RTX 5090
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm


# Project imports
from config import Config
from lightning.inference import LightningInference
from utils.logging_utils import setup_logging
from utils.evaluation import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_probability_distribution,
    generate_all_plots,
    print_evaluation_summary as print_metrics_summary
)
from sklearn.metrics import confusion_matrix, classification_report

# Setup logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_checkpoint(checkpoint_path: str) -> None:
    """Validate that checkpoint file exists."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def get_device(accelerator: str) -> str:
    """Get device with proper validation."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return 'cpu'
    return 'cuda'


def get_output_directory(output_path: str) -> Path:
    """Get output directory from a path, handling .parquet extensions."""
    output_dir = Path(output_path)
    if output_path.endswith('.parquet'):
        output_dir = output_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def calculate_ternary_metrics(predictions, targets, probabilities, buy_threshold=0.6, sell_threshold=0.6):
    """Calculate metrics for ternary classification."""
    # Apply thresholds to get final decisions
    final_predictions = apply_ternary_thresholds(probabilities, buy_threshold, sell_threshold)

    # Calculate confusion matrix
    cm = confusion_matrix(targets, final_predictions, labels=[0, 1, 2])

    # Calculate classification report
    report = classification_report(targets, final_predictions,
                                  labels=[0, 1, 2],
                                  target_names=['Hold', 'Buy', 'Sell'],
                                  output_dict=True)

    # Extract metrics
    metrics = {
        'confusion_matrix': cm,
        'accuracy': report['accuracy'],
        'hold_precision': report['Hold']['precision'],
        'hold_recall': report['Hold']['recall'],
        'hold_f1': report['Hold']['f1-score'],
        'buy_precision': report['Buy']['precision'],
        'buy_recall': report['Buy']['recall'],
        'buy_f1': report['Buy']['f1-score'],
        'sell_precision': report['Sell']['precision'],
        'sell_recall': report['Sell']['recall'],
        'sell_f1': report['Sell']['f1-score'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold
    }

    # Add trading-specific metrics
    trade_rate = (final_predictions != 0).sum() / len(final_predictions)
    buy_rate = (final_predictions == 1).sum() / len(final_predictions)
    sell_rate = (final_predictions == 2).sum() / len(final_predictions)

    metrics['trade_rate'] = trade_rate
    metrics['buy_rate'] = buy_rate
    metrics['sell_rate'] = sell_rate

    return metrics


def apply_ternary_thresholds(probabilities, buy_threshold=0.6, sell_threshold=0.6):
    """Apply thresholds to ternary probabilities to get final decisions.

    Args:
        probabilities: Array of shape [N, 3] with class probabilities
        buy_threshold: Minimum probability for buy decision
        sell_threshold: Minimum probability for sell decision

    Returns:
        Array of decisions: 0=Hold, 1=Buy, 2=Sell
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(f"Expected probabilities shape [N, 3], got {probabilities.shape}")

    decisions = np.zeros(len(probabilities), dtype=int)

    for i, probs in enumerate(probabilities):
        hold_prob, buy_prob, sell_prob = probs

        # Method 1: Highest probability with threshold check
        if buy_prob > buy_threshold and buy_prob > sell_prob:
            decisions[i] = 1  # Buy
        elif sell_prob > sell_threshold and sell_prob > buy_prob:
            decisions[i] = 2  # Sell
        else:
            decisions[i] = 0  # Hold (default)

    return decisions


def threshold_sweep(predictions, targets, probabilities,
                    buy_thresholds=None, sell_thresholds=None):
    """Sweep through different threshold combinations to find optimal settings.

    Args:
        predictions: Model predictions (may be ignored if probabilities provided)
        targets: True labels
        probabilities: Class probabilities [N, 3]
        buy_thresholds: List of buy thresholds to test
        sell_thresholds: List of sell thresholds to test

    Returns:
        DataFrame with results for each threshold combination
    """
    if buy_thresholds is None:
        buy_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    if sell_thresholds is None:
        sell_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []

    for buy_t in buy_thresholds:
        for sell_t in sell_thresholds:
            # Apply thresholds
            final_preds = apply_ternary_thresholds(probabilities, buy_t, sell_t)

            # Calculate metrics
            report = classification_report(targets, final_preds,
                                         labels=[0, 1, 2],
                                         target_names=['Hold', 'Buy', 'Sell'],
                                         output_dict=True)

            # Store results
            result = {
                'buy_threshold': buy_t,
                'sell_threshold': sell_t,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'trade_rate': (final_preds != 0).sum() / len(final_preds),
                'buy_rate': (final_preds == 1).sum() / len(final_preds),
                'sell_rate': (final_preds == 2).sum() / len(final_preds),
                'hold_f1': report['Hold']['f1-score'],
                'buy_f1': report['Buy']['f1-score'],
                'sell_f1': report['Sell']['f1-score']
            }
            results.append(result)

    return pd.DataFrame(results)


def generate_prediction_filename(input_file: Path, output_dir: Path, format: str = 'parquet') -> str:
    """Generate prediction output filename from input file."""
    output_path = output_dir / input_file.with_suffix('').name
    extension = '.csv' if format == 'csv' else '.parquet'
    output_path = output_path.with_suffix(extension).with_stem(f"{output_path.stem}_predictions")
    return str(output_path)


def run_inference(
    runner: LightningInference,
    data_source: str,
    batch_size: int,
    confidence_threshold: float = 0.5,
    buy_threshold: float = None,
    sell_threshold: float = None,
    classification_mode: str = 'binary'
) -> Dict[str, dict]:
    """Run inference with automatic evaluation when targets are available.

    Args:
        runner: LightningInference instance
        data_source: Path to parquet file
        batch_size: Batch size for processing
        confidence_threshold: Threshold for binary classification
        buy_threshold: Threshold for buy decision in ternary mode
        sell_threshold: Threshold for sell decision in ternary mode
        classification_mode: 'binary' or 'ternary'
    """
    # Only support parquet files
    if not data_source.endswith('.parquet'):
        raise ValueError("Only parquet files are supported")
    
    # Try to run with requested batch size, fall back on OOM
    current_batch_size = batch_size if batch_size else 16
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            predictions = runner.predict(
                data_source=data_source,
                batch_size=current_batch_size,
                return_predictions=True
            )
            break  # Success, exit retry loop
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and retry < max_retries - 1:
                # Reduce batch size and retry
                current_batch_size = max(1, current_batch_size // 2)
                logger.warning(f"GPU OOM with batch_size={batch_size}, retrying with batch_size={current_batch_size}")
                # Clear GPU cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                raise  # Re-raise if not OOM or max retries reached
    
    # The predict method now returns a dictionary with predictions and probabilities
    results = {'predictions': predictions}
    
    # Always evaluate when targets are available
    if 'targets' in predictions:
        # Calculate metrics using unified evaluation module
        if classification_mode == 'ternary':
            # For ternary mode, apply thresholds and calculate metrics
            metrics = calculate_ternary_metrics(
                predictions=predictions['predictions'],
                targets=predictions['targets'],
                probabilities=predictions['probabilities'],
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold
            )
        else:
            metrics = calculate_metrics(
                predictions=predictions['predictions'],
                targets=predictions['targets'],
                probabilities=predictions['probabilities'],
                threshold=confidence_threshold
            )
        results['metrics'] = metrics
    
    return results


def save_predictions_parquet(predictions: Dict, output_path: str, classification_mode: str = 'binary'):
    """Simple parquet export for predictions."""
    # Ensure all arrays are properly shaped
    preds = np.asarray(predictions['predictions'])
    probs = np.asarray(predictions['probabilities'])

    if classification_mode == 'ternary':
        # For ternary, probabilities should be [N, 3]
        if probs.ndim == 1:
            raise ValueError(f"Expected 2D probabilities for ternary mode, got shape {probs.shape}")

        df_data = {
            'prediction': preds.flatten(),
            'hold_prob': probs[:, 0] if probs.ndim == 2 else probs,
            'buy_prob': probs[:, 1] if probs.ndim == 2 else np.zeros_like(probs),
            'sell_prob': probs[:, 2] if probs.ndim == 2 else np.zeros_like(probs),
            'window_idx': predictions.get('window_indices', range(len(preds)))
        }
    else:
        # Binary mode
        preds = preds.flatten()
        probs = probs.flatten()

        if len(preds) != len(probs):
            raise ValueError(f"Predictions ({len(preds)}) and probabilities ({len(probs)}) have different lengths")

        df_data = {
            'prediction': preds,
            'probability': probs,
            'window_idx': predictions.get('window_indices', range(len(preds)))
        }

    df = pd.DataFrame(df_data)
    
    if 'targets' in predictions:
        targets = np.asarray(predictions['targets']).flatten()
        if len(targets) != len(preds):
            logger.warning(f"Targets ({len(targets)}) and predictions ({len(preds)}) have different lengths, skipping targets")
        else:
            df['target'] = targets
    
    df.to_parquet(output_path)


def save_predictions_csv(predictions: Dict, output_path: str, classification_mode: str = 'binary'):
    """CSV export for predictions."""
    # Ensure all arrays are properly shaped
    preds = np.asarray(predictions['predictions'])
    probs = np.asarray(predictions['probabilities'])

    if classification_mode == 'ternary':
        # For ternary, probabilities should be [N, 3]
        if probs.ndim == 1:
            raise ValueError(f"Expected 2D probabilities for ternary mode, got shape {probs.shape}")

        df_data = {
            'prediction': preds.flatten(),
            'hold_prob': probs[:, 0] if probs.ndim == 2 else probs,
            'buy_prob': probs[:, 1] if probs.ndim == 2 else np.zeros_like(probs),
            'sell_prob': probs[:, 2] if probs.ndim == 2 else np.zeros_like(probs),
            'window_idx': predictions.get('window_indices', range(len(preds)))
        }
    else:
        # Binary mode
        preds = preds.flatten()
        probs = probs.flatten()

        if len(preds) != len(probs):
            raise ValueError(f"Predictions ({len(preds)}) and probabilities ({len(probs)}) have different lengths")

        df_data = {
            'prediction': preds,
            'probability': probs,
            'window_idx': predictions.get('window_indices', range(len(preds)))
        }

    df = pd.DataFrame(df_data)
    
    if 'targets' in predictions:
        targets = np.asarray(predictions['targets']).flatten()
        if len(targets) != len(preds):
            logger.warning(f"Targets ({len(targets)}) and predictions ({len(preds)}) have different lengths, skipping targets")
        else:
            df['target'] = targets
    
    df.to_csv(output_path, index=False)


def save_predictions(predictions: Dict, output_path: str, format: str = 'parquet', classification_mode: str = 'binary'):
    """Save predictions in specified format."""
    if format == 'csv':
        # Change extension to .csv if needed
        if output_path.endswith('.parquet'):
            output_path = output_path.replace('.parquet', '.csv')
        save_predictions_csv(predictions, output_path, classification_mode)
    else:
        save_predictions_parquet(predictions, output_path, classification_mode)


def save_metrics_csv(metrics: Dict, output_path: str, filename: str = None):
    """Export metrics to CSV file in the standard format."""
    # Extract key metrics for CSV
    flat_metrics = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'accuracy': metrics.get('accuracy', 0)
    }

    # Add binary-specific metrics if available
    if 'roc_auc' in metrics:
        flat_metrics['roc_auc'] = metrics['roc_auc']
    if 'average_precision' in metrics:
        flat_metrics['average_precision'] = metrics['average_precision']

    # Add ternary-specific metrics if available
    if 'macro_avg_f1' in metrics:
        flat_metrics['macro_avg_f1'] = metrics['macro_avg_f1']
    if 'trade_rate' in metrics:
        flat_metrics['trade_rate'] = metrics['trade_rate']
    if 'buy_rate' in metrics:
        flat_metrics['buy_rate'] = metrics['buy_rate']
    if 'sell_rate' in metrics:
        flat_metrics['sell_rate'] = metrics['sell_rate']

    # Add confusion matrix elements based on size
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        if len(cm) == 2 and len(cm[0]) == 2:
            # Binary confusion matrix
            flat_metrics['true_negatives'] = cm[0][0]
            flat_metrics['false_positives'] = cm[0][1]
            flat_metrics['false_negatives'] = cm[1][0]
            flat_metrics['true_positives'] = cm[1][1]
        elif len(cm) == 3 and len(cm[0]) == 3:
            # Ternary confusion matrix - flatten all 9 values
            for i in range(3):
                for j in range(3):
                    label_i = ['hold', 'buy', 'sell'][i]
                    label_j = ['hold', 'buy', 'sell'][j]
                    flat_metrics[f'true_{label_i}_pred_{label_j}'] = cm[i][j]

    # Add filename if provided
    if filename:
        flat_metrics = {'file': filename, **flat_metrics}

    # Create DataFrame with single row
    df = pd.DataFrame([flat_metrics])
    df.to_csv(output_path, index=False)
    logger.info(f"Metrics exported to: {output_path}")














def print_aggregate_summary(all_metrics: List[Dict], title: str = "AGGREGATE EVALUATION SUMMARY"):
    """Print aggregate evaluation summary for multiple files."""
    avg_accuracy = np.mean([m['metrics']['accuracy'] for m in all_metrics])

    print("\n" + "="*60)
    print(title)
    print("="*60)
    print(f"\nProcessed {len(all_metrics)} files with evaluation")
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")

    # Check if we have binary or ternary metrics
    if all_metrics and 'roc_auc' in all_metrics[0]['metrics']:
        # Binary metrics
        avg_roc_auc = np.mean([m['metrics']['roc_auc'] for m in all_metrics])
        avg_ap = np.mean([m['metrics']['average_precision'] for m in all_metrics])
        print(f"Average ROC AUC: {avg_roc_auc:.4f}")
        print(f"Average Precision: {avg_ap:.4f}")
        print("\nPer-file results:")
        for m in all_metrics:
            print(f"  {m['file']}: Acc={m['metrics']['accuracy']:.4f}, AUC={m['metrics']['roc_auc']:.4f}")
    elif all_metrics and 'macro_avg_f1' in all_metrics[0]['metrics']:
        # Ternary metrics
        avg_macro_f1 = np.mean([m['metrics']['macro_avg_f1'] for m in all_metrics])
        avg_trade_rate = np.mean([m['metrics']['trade_rate'] for m in all_metrics])
        print(f"Average Macro F1: {avg_macro_f1:.4f}")
        print(f"Average Trade Rate: {avg_trade_rate:.4f}")
        print("\nPer-file results:")
        for m in all_metrics:
            print(f"  {m['file']}: Acc={m['metrics']['accuracy']:.4f}, F1={m['metrics']['macro_avg_f1']:.4f}, Trade={m['metrics']['trade_rate']:.4f}")
    else:
        # Generic metrics
        print("\nPer-file results:")
        for m in all_metrics:
            print(f"  {m['file']}: Acc={m['metrics']['accuracy']:.4f}")

    print("="*60)


def handle_predict_mode(args, config):
    """Handle prediction mode - supports both single files and directories."""
    # Determine classification mode from config
    classification_mode = getattr(config.model, 'classification_mode', 'binary')
    logger.info(f"Classification mode: {classification_mode}")

    # Get decision thresholds from args or config
    if classification_mode == 'ternary':
        # Use decision_thresholds from config (confidence-based thresholds for inference)
        buy_threshold = getattr(args, 'buy_confidence', None)
        sell_threshold = getattr(args, 'sell_confidence', None)

        # Fallback to config values if not provided via CLI
        if buy_threshold is None:
            buy_threshold = config.binary_classification.decision_thresholds.buy_confidence
        if sell_threshold is None:
            sell_threshold = config.binary_classification.decision_thresholds.sell_confidence

        logger.info(f"Buy confidence threshold: {buy_threshold}, Sell confidence threshold: {sell_threshold}")
    else:
        buy_threshold = None
        sell_threshold = None
    input_path = Path(args.input)
    
    # Validate input path exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    
    # Check if input is file or directory
    if input_path.is_file():
        # Single file mode - validate it's a parquet file
        if not str(input_path).endswith('.parquet'):
            raise ValueError(f"Input file must be a parquet file, got: {input_path}")
        files_to_process = [args.input]
        output_paths = [args.output]
        
    elif input_path.is_dir():
        # Directory mode - process all parquet files
        files_to_process = list(input_path.glob('*.parquet'))
        
        if not files_to_process:
            logger.error(f"No parquet files found in {args.input}")
            return 1
        
        logger.info(f"Found {len(files_to_process)} parquet files to process")
        
        # Get output directory using helper
        output_dir = get_output_directory(args.output)
        
        # Generate output paths for each file
        output_format = getattr(args, 'output_format', 'parquet')
        output_paths = [generate_prediction_filename(file_path, output_dir, format=output_format) for file_path in files_to_process]
    else:
        raise ValueError(f"Input path is neither file nor directory: {args.input}")
    
    # Setup device using helper function
    device = get_device(args.accelerator)
    
    # Log if sequential mode is enabled
    if getattr(args, 'sequential', False):
        logger.info("Sequential inference mode enabled")
        logger.info(f"Stride: {getattr(args, 'stride', 'default (chunk_size)')}")
        logger.info(f"Overlap ratio: {getattr(args, 'overlap_ratio', 0.0)}")
    
    # Always use LightningInference for all processing
    runner = LightningInference(
        config=config,
        checkpoint_path=args.checkpoint,
        preprocessor_path=args.preprocessor,
        device=device,
        sequential_mode=getattr(args, 'sequential', False),
        stride=getattr(args, 'stride', None),
        overlap_ratio=getattr(args, 'overlap_ratio', 0.0)
    )
    
    all_metrics = []
    
    # Process each file
    for file_path, output_path in zip(files_to_process, output_paths):
        logger.info(f"Running inference on: {file_path}")
        
        results = run_inference(
            runner=runner,
            data_source=str(file_path),
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            classification_mode=classification_mode
        )
        
        # Save predictions in specified format
        output_format = getattr(args, 'output_format', 'parquet')
        save_predictions(results['predictions'], output_path, format=output_format, classification_mode=classification_mode)
        logger.info(f"Predictions saved to: {output_path} (format: {output_format})")
        
        if 'metrics' in results:
            all_metrics.append({
                'file': Path(file_path).name,
                'metrics': results['metrics']
            })

            # Perform threshold sweep if requested (only for ternary mode)
            if classification_mode == 'ternary' and getattr(args, 'threshold_sweep', False):
                logger.info("Performing threshold sweep...")
                if 'targets' in results['predictions'] and 'probabilities' in results['predictions']:
                    sweep_results = threshold_sweep(
                        predictions=results['predictions']['predictions'],
                        targets=results['predictions']['targets'],
                        probabilities=results['predictions']['probabilities']
                    )

                    # Save sweep results
                    sweep_output = getattr(args, 'sweep_output', 'threshold_sweep_results.csv')
                    sweep_results.to_csv(sweep_output, index=False)
                    logger.info(f"Threshold sweep results saved to: {sweep_output}")

                    # Print best thresholds
                    best_idx = sweep_results['macro_f1'].idxmax()
                    best_row = sweep_results.iloc[best_idx]
                    print("\n" + "="*60)
                    print("BEST THRESHOLD COMBINATION (by macro F1)")
                    print("="*60)
                    print(f"Buy Threshold: {best_row['buy_threshold']:.2f}")
                    print(f"Sell Threshold: {best_row['sell_threshold']:.2f}")
                    print(f"Macro F1: {best_row['macro_f1']:.4f}")
                    print(f"Accuracy: {best_row['accuracy']:.4f}")
                    print(f"Trade Rate: {best_row['trade_rate']:.4f}")
                    print(f"Buy Rate: {best_row['buy_rate']:.4f}")
                    print(f"Sell Rate: {best_row['sell_rate']:.4f}")
                    print("="*60)
                else:
                    logger.warning("Threshold sweep requires both targets and probabilities")
    
    # Print evaluation summary and generate plots when metrics are available
    if all_metrics:
        if len(all_metrics) == 1:
            # Single file - print detailed metrics
            print_metrics_summary(all_metrics[0]['metrics'])
            # Always save plots
            output_dir = Path(output_paths[0]).parent
            metrics = all_metrics[0]['metrics']
            
            # Export metrics to CSV if requested
            if getattr(args, 'export_metrics_csv', False):
                metrics_output = getattr(args, 'metrics_output', None)
                if metrics_output:
                    metrics_path = Path(metrics_output)
                else:
                    metrics_path = output_dir / 'evaluation_metrics.csv'
                save_metrics_csv(metrics, str(metrics_path), filename=all_metrics[0]['file'])
            
            # For single file mode, we don't have access to raw predictions here
            # Plots that require predictions will be skipped
            generate_evaluation_plots(
                results={'metrics': metrics, 'predictions': None},
                output_dir=output_dir
            )
        else:
            # Multiple files - print aggregate summary
            print_aggregate_summary(all_metrics)
            
            # Save aggregate plots
            output_dir = get_output_directory(args.output)
            
            # Export aggregate metrics to CSV if requested
            if getattr(args, 'export_metrics_csv', False):
                metrics_output = getattr(args, 'metrics_output', None)
                if metrics_output:
                    metrics_path = Path(metrics_output)
                else:
                    metrics_path = output_dir / 'aggregate_metrics.csv'
                
                # Create aggregate metrics CSV with all files in standard format
                aggregate_data = []
                for item in all_metrics:
                    metrics = item['metrics']
                    row = {
                        'file': item['file'],
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'accuracy': metrics.get('accuracy', 0),
                        'roc_auc': metrics.get('roc_auc', 0),
                        'average_precision': metrics.get('average_precision', 0)
                    }
                    
                    # Add confusion matrix elements
                    if 'confusion_matrix' in metrics:
                        cm = metrics['confusion_matrix']
                        if len(cm) >= 2 and len(cm[0]) >= 2:
                            row['true_negatives'] = cm[0][0]
                            row['false_positives'] = cm[0][1]
                            row['false_negatives'] = cm[1][0]
                            row['true_positives'] = cm[1][1]
                    
                    aggregate_data.append(row)
                
                df = pd.DataFrame(aggregate_data)
                df.to_csv(metrics_path, index=False)
                logger.info(f"Aggregate metrics exported to: {metrics_path}")
            
            # Use last file's metrics for aggregate plots (could be improved to use combined data)
            # For multi-file mode, we don't have predictions available for plots
            generate_evaluation_plots(
                results={'metrics': all_metrics[-1]['metrics'], 'predictions': None},
                output_dir=output_dir
            )
    
    return 0


def generate_evaluation_plots(results: Dict[str, dict], output_dir: Path):
    """Generate evaluation plots from results using unified evaluation module."""
    metrics = results['metrics']
    predictions = results.get('predictions', None)
    
    # Use unified plotting function
    generate_all_plots(
        metrics=metrics,
        predictions=predictions,
        output_dir=output_dir
    )




def handle_batch_mode(args, config):
    """Handle batch file processing mode with automatic evaluation."""
    # Determine classification mode from config
    classification_mode = getattr(config.model, 'classification_mode', 'binary')
    logger.info(f"Classification mode: {classification_mode}")

    # Get decision thresholds from args or config
    if classification_mode == 'ternary':
        # Use decision_thresholds from config (confidence-based thresholds for inference)
        buy_threshold = getattr(args, 'buy_confidence', None)
        sell_threshold = getattr(args, 'sell_confidence', None)

        # Fallback to config values if not provided via CLI
        if buy_threshold is None:
            buy_threshold = config.binary_classification.decision_thresholds.buy_confidence
        if sell_threshold is None:
            sell_threshold = config.binary_classification.decision_thresholds.sell_confidence

        logger.info(f"Buy confidence threshold: {buy_threshold}, Sell confidence threshold: {sell_threshold}")
    else:
        buy_threshold = None
        sell_threshold = None

    input_dir = Path(args.input_dir)
    
    # Validate input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {args.input_dir}")
    
    files = list(input_dir.glob('*.parquet'))  # Parquet only
    
    if not files:
        logger.error(f"No parquet files found in {args.input_dir}")
        return 1
    
    # Get device with validation
    device = get_device('gpu')
    
    # Log if sequential mode is enabled
    if getattr(args, 'sequential', False):
        logger.info("Sequential inference mode enabled for batch processing")
        logger.info(f"Stride: {getattr(args, 'stride', 'default (chunk_size)')}")
        logger.info(f"Overlap ratio: {getattr(args, 'overlap_ratio', 0.0)}")
    
    runner = LightningInference(
        config=config,
        checkpoint_path=args.checkpoint,
        preprocessor_path=args.preprocessor,
        device=device,
        sequential_mode=getattr(args, 'sequential', False),
        stride=getattr(args, 'stride', None),
        overlap_ratio=getattr(args, 'overlap_ratio', 0.0)
    )
    
    logger.info(f"Found {len(files)} files to process")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    all_results = []
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            results = run_inference(
                runner=runner,
                data_source=str(file_path),
                batch_size=args.batch_size,
                confidence_threshold=args.confidence_threshold,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                classification_mode=classification_mode
            )
            
            # Save individual file predictions in specified format
            output_format = getattr(args, 'output_format', 'parquet')
            output_path = generate_prediction_filename(file_path, output_dir, format=output_format)
            save_predictions(results['predictions'], output_path, format=output_format, classification_mode=classification_mode)
            
            if 'metrics' in results:
                all_metrics.append({
                    'file': file_path.name,
                    'metrics': results['metrics']
                })
                all_results.append(results)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    # Print evaluation summary and always generate plots when metrics available
    if all_metrics:
        print_aggregate_summary(all_metrics, title="BATCH EVALUATION SUMMARY")
        
        # Export metrics to CSV if requested
        if getattr(args, 'export_metrics_csv', False):
            metrics_output = getattr(args, 'metrics_output', None)
            if metrics_output:
                metrics_path = Path(metrics_output)
            else:
                metrics_path = output_dir / 'batch_metrics.csv'
            
            # Create batch metrics CSV with all files in standard format
            batch_data = []
            for item in all_metrics:
                metrics = item['metrics']
                row = {
                    'file': item['file'],
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'accuracy': metrics.get('accuracy', 0),
                    'roc_auc': metrics.get('roc_auc', 0),
                    'average_precision': metrics.get('average_precision', 0)
                }
                
                # Add confusion matrix elements
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    if len(cm) >= 2 and len(cm[0]) >= 2:
                        row['true_negatives'] = cm[0][0]
                        row['false_positives'] = cm[0][1]
                        row['false_negatives'] = cm[1][0]
                        row['true_positives'] = cm[1][1]
                
                batch_data.append(row)
            
            df = pd.DataFrame(batch_data)
            df.to_csv(metrics_path, index=False)
            logger.info(f"Batch metrics exported to: {metrics_path}")
        
        # Always generate plots - use last file's metrics as representative
        if all_results:
            # Get predictions from the last result if available
            last_predictions = all_results[-1].get('predictions') if all_results else None
            generate_evaluation_plots(
                results={'metrics': all_metrics[-1]['metrics'], 'predictions': last_predictions},
                output_dir=output_dir
            )
    
    return 0




def parse_arguments():
    """Parse command line arguments with subcommand structure."""
    parser = argparse.ArgumentParser(
        description="Unified inference and evaluation script for Order Book Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file prediction
  python3.10 scripts/inference.py predict --checkpoint model.ckpt --input data.parquet
  
  # Directory processing (evaluates automatically if targets exist)
  python3.10 scripts/inference.py predict --checkpoint model.ckpt --input ./test/ --output ./predictions/
  
  # Batch processing (processes files individually with separate outputs)
  python3.10 scripts/inference.py batch --checkpoint model.ckpt --input-dir ./data/
        """
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Predict mode - inference on files or directories
    predict_parser = subparsers.add_parser('predict', help='Run inference on a file or directory')
    predict_parser.add_argument('--input', type=str, required=True, 
                               help='Input parquet file or directory containing parquet files')
    predict_parser.add_argument('--output', type=str, default='predictions.parquet',
                               help='Output file/directory for predictions (default: predictions.parquet)')
    predict_parser.add_argument('--confidence-threshold', type=float, default=0.5,
                               help='Confidence threshold for binary classification (default: 0.5)')

    # Ternary classification decision thresholds (confidence-based)
    predict_parser.add_argument('--buy-confidence', type=float, default=None,
                               help='Confidence threshold for buy decision in ternary mode (default: from config)')
    predict_parser.add_argument('--sell-confidence', type=float, default=None,
                               help='Confidence threshold for sell decision in ternary mode (default: from config)')

    # Threshold sweep option
    predict_parser.add_argument('--threshold-sweep', action='store_true',
                               help='Perform threshold sweep to find optimal buy/sell thresholds')
    predict_parser.add_argument('--sweep-output', type=str, default='threshold_sweep_results.csv',
                               help='Output file for threshold sweep results (default: threshold_sweep_results.csv)')
    
    # Output format selection
    predict_parser.add_argument('--output-format', type=str, choices=['parquet', 'csv'], default='parquet',
                               help='Output format for predictions (default: parquet)')
    
    # Sequential inference options
    predict_parser.add_argument('--sequential', action='store_true',
                               help='Use sequential inference mode (process from beginning to end)')
    predict_parser.add_argument('--stride', type=int, default=None,
                               help='Stride for sequential processing (default: chunk_size)')
    predict_parser.add_argument('--overlap-ratio', type=float, default=0.0,
                               help='Overlap ratio between chunks for sequential mode (0.0-1.0, default: 0.0)')
    
    # CSV export options
    predict_parser.add_argument('--export-metrics-csv', action='store_true',
                               help='Export evaluation metrics to CSV file')
    predict_parser.add_argument('--metrics-output', type=str, default=None,
                               help='Path for metrics CSV output (default: auto-generated based on output path)')
    
    # Batch mode - process multiple files
    batch_parser = subparsers.add_parser('batch', help='Process multiple parquet files')
    batch_parser.add_argument('--input-dir', type=str, required=True,
                             help='Directory containing parquet files')
    batch_parser.add_argument('--output-dir', type=str, default='./batch_predictions',
                             help='Directory for output files (default: ./batch_predictions)')
    batch_parser.add_argument('--confidence-threshold', type=float, default=0.5,
                             help='Confidence threshold for binary classification (default: 0.5)')

    # Ternary classification decision thresholds for batch mode (confidence-based)
    batch_parser.add_argument('--buy-confidence', type=float, default=None,
                             help='Confidence threshold for buy decision in ternary mode (default: from config)')
    batch_parser.add_argument('--sell-confidence', type=float, default=None,
                             help='Confidence threshold for sell decision in ternary mode (default: from config)')
    
    # Output format selection for batch mode
    batch_parser.add_argument('--output-format', type=str, choices=['parquet', 'csv'], default='parquet',
                             help='Output format for predictions (default: parquet)')
    
    # Sequential inference options for batch mode
    batch_parser.add_argument('--sequential', action='store_true',
                             help='Use sequential inference mode (process from beginning to end)')
    batch_parser.add_argument('--stride', type=int, default=None,
                             help='Stride for sequential processing (default: chunk_size)')
    batch_parser.add_argument('--overlap-ratio', type=float, default=0.0,
                             help='Overlap ratio between chunks for sequential mode (0.0-1.0, default: 0.0)')
    
    # CSV export options for batch mode
    batch_parser.add_argument('--export-metrics-csv', action='store_true',
                             help='Export evaluation metrics to CSV file')
    batch_parser.add_argument('--metrics-output', type=str, default=None,
                             help='Path for metrics CSV output (default: auto-generated based on output dir)')
    
    # Add shared arguments to all subparsers
    for subparser in [predict_parser, batch_parser]:
        subparser.add_argument('--checkpoint', type=str, required=True,
                              help='Path to model checkpoint')
        subparser.add_argument('--config', type=str, default=None,
                              help='Path to config file (default: config.json5 or config.dev.json5)')
        subparser.add_argument('--batch-size', type=int, default=32,
                              help='Batch size for processing (default: 32)')
        subparser.add_argument('--accelerator', choices=['auto', 'gpu'], default='gpu',
                              help='Hardware accelerator (always uses GPU)')
        subparser.add_argument('--num-workers', type=int, default=8,
                              help='Number of data loading workers (default: 8)')
        subparser.add_argument('--preprocessor', type=str,
                              help='Path to saved preprocessor (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Require mode to be specified
    if not hasattr(args, 'mode') or args.mode is None:
        parser.error("Please specify a mode: predict or batch")
    
    return args


def main() -> int:
    """Main inference function using Lightning directly.
    
    Always performs evaluation when targets are available,
    saves predictions, and generates evaluation plots.
    
    Returns:
        0 on success, 1 on error
    """
    args = parse_arguments()
    
    # Setup logging
    setup_logging()
    
    # Load config from specified path or use defaults (with inference_mode=True)
    if hasattr(args, 'config') and args.config:
        logger.info(f"Loading config from: {args.config}")
        config = Config(json5_path=args.config, inference_mode=True)
    else:
        # Create config with default search (config.json5 or config.dev.json5)
        config = Config(inference_mode=True)
    
    # Override config with any CLI args if needed
    if hasattr(args, 'batch_size'):
        config.data.batch_size = args.batch_size
    
    # Validate checkpoint exists before mode dispatch
    try:
        validate_checkpoint(args.checkpoint)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    
    try:
        # Dispatch to mode handler
        if args.mode == 'predict':
            return handle_predict_mode(args, config)
        elif args.mode == 'batch':
            return handle_batch_mode(args, config)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        return 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU out of memory. Try reducing --batch-size")
            logger.error("Tip: You can also try enabling gradient checkpointing or using mixed precision")
        else:
            logger.error(f"Runtime error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        # Clean up shared memory on exit
        try:
            from data import cleanup_shared_memory
            cleanup_shared_memory()
        except ImportError:
            pass  # Module might not have cleanup function

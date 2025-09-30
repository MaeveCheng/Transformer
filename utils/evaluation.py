#!/usr/bin/env python3.10
"""
Unified evaluation module for Order Book Transformer.

This module consolidates all evaluation and visualization functionality
from scripts/inference.py, lightning/evaluation.py, and lightning/callbacks.py
into a single, consistent implementation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchmetrics
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
    AUROC,
    PrecisionRecallCurve,
    ROC,
    AveragePrecision
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    include_trading_metrics: bool = False,
    min_confidence: float = 0.6
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics for binary classification.
    
    This function consolidates metric calculations from:
    - scripts/inference.py: calculate_evaluation_metrics()
    - lightning/evaluation.py: MetricsCalculator.calculate_classification_metrics()
    
    Args:
        predictions: Binary predictions (0 or 1)
        targets: True labels (0 or 1)
        probabilities: Prediction probabilities [0, 1]
        threshold: Classification threshold (default: 0.5)
        include_trading_metrics: Whether to include trading-specific metrics
        min_confidence: Minimum confidence for trading metrics (default: 0.6)
        
    Returns:
        Dictionary containing all calculated metrics
    """
    # Always use GPU
    device = torch.device('cuda')
    
    # Convert to torch tensors and move to device
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.long, device=device)
    else:
        predictions = predictions.to(device)
        
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long, device=device)
    else:
        targets = targets.to(device)
        
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, dtype=torch.float32, device=device)
    else:
        probabilities = probabilities.to(device)
    
    # Re-threshold predictions if needed
    if threshold != 0.5:
        predictions = (probabilities >= threshold).long()
    
    # Initialize metrics
    num_classes = 2
    accuracy_metric = Accuracy(task='binary').to(device)
    precision_metric = Precision(task='binary').to(device)
    recall_metric = Recall(task='binary').to(device)
    f1_metric = F1Score(task='binary').to(device)
    confusion_metric = ConfusionMatrix(task='binary', num_classes=num_classes).to(device)
    auroc_metric = AUROC(task='binary').to(device)
    avg_precision_metric = AveragePrecision(task='binary').to(device)
    pr_curve_metric = PrecisionRecallCurve(task='binary').to(device)
    roc_metric = ROC(task='binary').to(device)
    
    # Calculate metrics - all computations stay on GPU
    accuracy = accuracy_metric(predictions, targets)
    # Calculate single-value metrics
    precision_value = precision_metric(predictions, targets)
    recall_value = recall_metric(predictions, targets)
    f1_value = f1_metric(predictions, targets)
    
    # Create per-class arrays for compatibility
    # For binary classification: index 0 = negative class, index 1 = positive class
    precision_per_class = torch.tensor([1.0 - precision_value, precision_value], device=device)
    recall_per_class = torch.tensor([1.0 - recall_value, recall_value], device=device)
    f1_per_class = torch.tensor([2 * (1 - f1_value) / (2 - f1_value) if f1_value < 1 else 0, f1_value], device=device)
    cm = confusion_metric(predictions, targets)
    roc_auc = auroc_metric(probabilities, targets)
    avg_precision = avg_precision_metric(probabilities, targets)
    
    # Get curves
    precision_curve, recall_curve, pr_thresholds = pr_curve_metric(probabilities, targets)
    fpr, tpr, roc_thresholds = roc_metric(probabilities, targets)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = torch.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx] if len(roc_thresholds) > optimal_idx else threshold
    
    # Create classification report from torchmetrics results
    # Note: precision, recall, f1 are per-class arrays
    report = {
        'Down': {
            'precision': float(precision_per_class[0]),
            'recall': float(recall_per_class[0]),
            'f1-score': float(f1_per_class[0]),
            'support': int((targets == 0).sum())
        },
        'Up': {
            'precision': float(precision_per_class[1]),
            'recall': float(recall_per_class[1]),
            'f1-score': float(f1_per_class[1]),
            'support': int((targets == 1).sum())
        },
        'accuracy': float(accuracy),
        'macro avg': {
            'precision': float(precision_per_class.mean()),
            'recall': float(recall_per_class.mean()),
            'f1-score': float(f1_per_class.mean()),
            'support': int(len(targets))
        },
        'weighted avg': {
            'precision': float((precision_per_class[0] * (targets == 0).sum() + 
                              precision_per_class[1] * (targets == 1).sum()) / len(targets)),
            'recall': float((recall_per_class[0] * (targets == 0).sum() + 
                           recall_per_class[1] * (targets == 1).sum()) / len(targets)),
            'f1-score': float((f1_per_class[0] * (targets == 0).sum() + 
                             f1_per_class[1] * (targets == 1).sum()) / len(targets)),
            'support': int(len(targets))
        }
    }
    
    # Compile results - minimize CPU transfers, only convert when necessary for serialization
    metrics = {
        'accuracy': float(accuracy.item()),
        'confusion_matrix': cm.cpu().numpy().tolist(),
        'classification_report': report,
        'roc_auc': float(roc_auc.item()),
        'average_precision': float(avg_precision.item()),
        'optimal_threshold': float(optimal_threshold.item()),
        'threshold': float(threshold),
        'curves': {
            'roc': {'fpr': fpr.cpu().numpy().tolist(), 'tpr': tpr.cpu().numpy().tolist()},
            'pr': {'precision': precision_curve.cpu().numpy().tolist(), 
                   'recall': recall_curve.cpu().numpy().tolist()}
        },
        'class_distribution': {
            'class_0_count': int((targets == 0).sum().item()),
            'class_1_count': int((targets == 1).sum().item()),
            'class_0_ratio': float((targets == 0).float().mean().item()),
            'class_1_ratio': float((targets == 1).float().mean().item())
        }
    }
    
    # Add per-class metrics for easier access
    metrics['precision_class_0'] = float(precision_per_class[0].item())
    metrics['recall_class_0'] = float(recall_per_class[0].item())
    metrics['f1_class_0'] = float(f1_per_class[0].item())
    metrics['precision_class_1'] = float(precision_per_class[1].item())
    metrics['recall_class_1'] = float(recall_per_class[1].item())
    metrics['f1_class_1'] = float(f1_per_class[1].item())
    
    # Trading metrics if requested
    if include_trading_metrics:
        # Convert to numpy for trading metrics calculation
        trading_metrics = calculate_trading_metrics(
            predictions.cpu().numpy(), 
            targets.cpu().numpy(), 
            probabilities.cpu().numpy(), 
            threshold=threshold, 
            min_confidence=min_confidence
        )
        metrics['trading_metrics'] = trading_metrics
    
    return metrics


def calculate_trading_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_confidence: float = 0.6
) -> Dict[str, float]:
    """
    Calculate trading-specific metrics.
    
    Consolidated from lightning/evaluation.py: MetricsCalculator.calculate_trading_metrics()
    
    Args:
        predictions: Binary predictions
        targets: True labels
        probabilities: Prediction probabilities
        threshold: Classification threshold
        min_confidence: Minimum confidence for trades
        
    Returns:
        Dictionary of trading metrics
    """
    # Filter by confidence
    confident_mask = np.abs(probabilities - 0.5) >= (min_confidence - 0.5)
    
    if confident_mask.sum() == 0:
        logger.warning("No predictions meet confidence threshold")
        return {
            'num_trades': 0,
            'accuracy_at_confidence': 0.0,
            'trade_ratio': 0.0,
            'long_accuracy': 0.0,
            'short_accuracy': 0.0,
            'min_confidence': float(min_confidence),
            'long_trade_count': 0,
            'short_trade_count': 0
        }
    
    confident_predictions = predictions[confident_mask]
    confident_targets = targets[confident_mask]
    
    # Trading metrics
    num_trades = confident_mask.sum()
    trade_ratio = num_trades / len(predictions)
    accuracy_at_confidence = (confident_predictions == confident_targets).mean()
    
    # Calculate directional accuracy
    long_mask = confident_predictions == 1
    short_mask = confident_predictions == 0
    
    long_accuracy = 0.0
    short_accuracy = 0.0
    
    if long_mask.sum() > 0:
        long_accuracy = (confident_predictions[long_mask] == confident_targets[long_mask]).mean()
    
    if short_mask.sum() > 0:
        short_accuracy = (confident_predictions[short_mask] == confident_targets[short_mask]).mean()
    
    return {
        'num_trades': int(num_trades),
        'trade_ratio': float(trade_ratio),
        'accuracy_at_confidence': float(accuracy_at_confidence),
        'long_accuracy': float(long_accuracy),
        'short_accuracy': float(short_accuracy),
        'min_confidence': float(min_confidence),
        'long_trade_count': int(long_mask.sum()),
        'short_trade_count': int(short_mask.sum())
    }


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    output_path: Union[str, Path],
    title: str = 'Confusion Matrix',
    class_names: List[str] = ['Down', 'Up']
) -> None:
    """
    Plot and save confusion matrix.
    
    Unified implementation from:
    - scripts/inference.py: plot_confusion_matrix()
    - lightning/callbacks.py: VisualizationCallback._plot_confusion_matrix()
    
    Args:
        cm: Confusion matrix array or list
        output_path: Path to save the plot
        title: Plot title
        class_names: Names for classes
    """
    cm = np.array(cm) if isinstance(cm, list) else cm
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to: {output_path}")


def plot_roc_curve(
    fpr: Union[np.ndarray, List[float]],
    tpr: Union[np.ndarray, List[float]],
    roc_auc: float,
    output_path: Union[str, Path],
    title: str = 'Receiver Operating Characteristic (ROC) Curve'
) -> None:
    """
    Plot and save ROC curve.
    
    Unified implementation from:
    - scripts/inference.py: plot_roc_curve()
    - lightning/callbacks.py: VisualizationCallback._plot_roc_curve()
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under curve
        output_path: Path to save the plot
        title: Plot title
    """
    fpr = np.array(fpr) if isinstance(fpr, list) else fpr
    tpr = np.array(tpr) if isinstance(tpr, list) else tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to: {output_path}")


def plot_precision_recall_curve(
    precision: Union[np.ndarray, List[float]],
    recall: Union[np.ndarray, List[float]],
    avg_precision: float,
    output_path: Union[str, Path],
    title: str = 'Precision-Recall Curve'
) -> None:
    """
    Plot and save precision-recall curve.
    
    Unified implementation from:
    - scripts/inference.py: plot_precision_recall_curve()
    - lightning/callbacks.py: VisualizationCallback._plot_precision_recall()
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        output_path: Path to save the plot
        title: Plot title
    """
    precision = np.array(precision) if isinstance(precision, list) else precision
    recall = np.array(recall) if isinstance(recall, list) else recall
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Precision-Recall curve saved to: {output_path}")


def plot_probability_distribution(
    probabilities: Union[np.ndarray, List[float]],
    targets: Union[np.ndarray, List[int]],
    output_path: Union[str, Path],
    title: str = 'Probability Distribution by True Class',
    bins: int = 50
) -> None:
    """
    Plot probability distribution by class.
    
    Unified implementation from:
    - scripts/inference.py: plot_probability_distribution()
    - lightning/callbacks.py: VisualizationCallback._plot_prediction_distribution()
    
    Args:
        probabilities: Prediction probabilities
        targets: True labels
        output_path: Path to save the plot
        title: Plot title
        bins: Number of histogram bins
    """
    probabilities = np.array(probabilities) if isinstance(probabilities, list) else probabilities
    targets = np.array(targets) if isinstance(targets, list) else targets
    
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by true class
    prob_class_0 = probabilities[targets == 0]
    prob_class_1 = probabilities[targets == 1]
    
    # Plot histograms
    plt.hist(prob_class_0, bins=bins, alpha=0.5, label='True: Down', 
             color='blue', density=True, edgecolor='black', linewidth=0.5)
    plt.hist(prob_class_1, bins=bins, alpha=0.5, label='True: Up', 
             color='red', density=True, edgecolor='black', linewidth=0.5)
    
    # Add vertical line at 0.5 threshold
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    
    plt.xlabel('Predicted Probability of Up Movement')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Probability distribution saved to: {output_path}")


def generate_all_plots(
    metrics: Dict[str, Any],
    predictions: Optional[Dict[str, np.ndarray]],
    output_dir: Union[str, Path],
    plot_types: Optional[List[str]] = None
) -> None:
    """
    Generate all evaluation plots from metrics and predictions.
    
    Args:
        metrics: Dictionary of calculated metrics (from calculate_metrics)
        predictions: Optional dictionary with 'probabilities' and 'targets'
        output_dir: Directory to save plots
        plot_types: List of plot types to generate (default: all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if plot_types is None:
        plot_types = ["confusion_matrix", "roc_curve", "precision_recall", "probability_distribution"]
    
    # Plot confusion matrix
    if "confusion_matrix" in plot_types and 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')
    
    # Plot ROC curve
    if "roc_curve" in plot_types and 'curves' in metrics and 'roc' in metrics['curves']:
        fpr = np.array(metrics['curves']['roc']['fpr'])
        tpr = np.array(metrics['curves']['roc']['tpr'])
        plot_roc_curve(fpr, tpr, metrics['roc_auc'], output_dir / 'roc_curve.png')
    
    # Plot precision-recall curve
    if "precision_recall" in plot_types and 'curves' in metrics and 'pr' in metrics['curves']:
        precision = np.array(metrics['curves']['pr']['precision'])
        recall = np.array(metrics['curves']['pr']['recall'])
        plot_precision_recall_curve(
            precision, recall, metrics['average_precision'], 
            output_dir / 'pr_curve.png'
        )
    
    # Plot probability distribution
    if ("probability_distribution" in plot_types and predictions is not None 
        and 'probabilities' in predictions and 'targets' in predictions):
        plot_probability_distribution(
            predictions['probabilities'],
            predictions['targets'],
            output_dir / 'probability_distribution.png'
        )


def print_evaluation_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted evaluation summary.
    
    Unified from scripts/inference.py: print_evaluation_summary()
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    # Per-class metrics
    print(f"\nClass 0 (Down):")
    print(f"  Precision: {metrics['precision_class_0']:.3f}")
    print(f"  Recall: {metrics['recall_class_0']:.3f}")
    print(f"  F1-Score: {metrics['f1_class_0']:.3f}")
    
    print(f"\nClass 1 (Up):")
    print(f"  Precision: {metrics['precision_class_1']:.3f}")
    print(f"  Recall: {metrics['recall_class_1']:.3f}")
    print(f"  F1-Score: {metrics['f1_class_1']:.3f}")
    
    # Class distribution
    if 'class_distribution' in metrics:
        dist = metrics['class_distribution']
        print(f"\nClass Distribution:")
        print(f"  Class 0: {dist['class_0_count']} ({dist['class_0_ratio']:.1%})")
        print(f"  Class 1: {dist['class_1_count']} ({dist['class_1_ratio']:.1%})")
    
    # Trading metrics if available
    if 'trading_metrics' in metrics:
        tm = metrics['trading_metrics']
        print(f"\nTrading Metrics (confidence >= {tm['min_confidence']}):")
        print(f"  Trades: {tm['num_trades']} ({tm['trade_ratio']:.1%} of total)")
        print(f"  Accuracy at confidence: {tm['accuracy_at_confidence']:.3f}")
        print(f"  Long accuracy: {tm['long_accuracy']:.3f} ({tm['long_trade_count']} trades)")
        print(f"  Short accuracy: {tm['short_accuracy']:.3f} ({tm['short_trade_count']} trades)")
    
    print("="*60)


def optimize_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    optimization_metric: str = "f1_score",
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    num_thresholds: int = 50
) -> Dict[str, Any]:
    """
    Find optimal classification threshold.
    
    Args:
        targets: True labels
        probabilities: Prediction probabilities
        optimization_metric: Metric to optimize ('f1_score', 'accuracy', 'precision', 'recall')
        threshold_range: Range of thresholds to test
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        Dictionary with optimal threshold and metrics at each threshold
    """
    # Always use GPU
    device = torch.device('cuda')
    
    # Convert to torch tensors
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.long, device=device)
    else:
        targets = targets.to(device)
        
    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.tensor(probabilities, dtype=torch.float32, device=device)
    else:
        probabilities = probabilities.to(device)
    
    # Initialize metrics
    accuracy_metric = Accuracy(task='binary').to(device)
    precision_metric = Precision(task='binary').to(device)
    recall_metric = Recall(task='binary').to(device)
    f1_metric = F1Score(task='binary').to(device)
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    results = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).long()
        
        # Calculate metrics on GPU
        accuracy = accuracy_metric(predictions, targets)
        precision = precision_metric(predictions, targets)
        recall = recall_metric(predictions, targets)
        f1 = f1_metric(predictions, targets)
        
        result = {
            'threshold': float(threshold),
            'accuracy': float(accuracy.item()),
            'precision': float(precision.item()),
            'recall': float(recall.item()),
            'f1_score': float(f1.item())
        }
        results.append(result)
    
    # Find optimal threshold
    results_df = pd.DataFrame(results)
    optimal_idx = results_df[optimization_metric].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_metrics = results_df.loc[optimal_idx].to_dict()
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics,
        'optimization_metric': optimization_metric,
        'all_results': results
    }
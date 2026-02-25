#!/usr/bin/env python3
"""
混同行列の生成と可視化ツール

Usage:
    python tools/generate_confusion_matrix.py --predictions predictions.npy --labels labels.npy --output confusion_matrix.png
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


def load_predictions_and_labels(pred_path: Path, label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """予測とラベルを読み込む"""
    predictions = np.load(pred_path)
    labels = np.load(label_path)
    
    # 1次元配列に変換
    if predictions.ndim > 1:
        predictions = predictions.argmax(axis=1) if predictions.shape[1] > 1 else predictions.flatten()
    if labels.ndim > 1:
        labels = labels.argmax(axis=1) if labels.shape[1] > 1 else labels.flatten()
    
    return predictions, labels


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str] | None = None, 
                          output_path: Path | None = None, normalize: bool = False,
                          title: str = "Confusion Matrix"):
    """混同行列を可視化"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += " (Normalized)"
    else:
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate confusion matrix from predictions and labels")
    parser.add_argument('--predictions', type=str, required=True, help='Path to predictions .npy file')
    parser.add_argument('--labels', type=str, required=True, help='Path to labels .npy file')
    parser.add_argument('--output', type=str, default='confusion_matrix.png', help='Output image path')
    parser.add_argument('--normalize', action='store_true', help='Normalize confusion matrix')
    parser.add_argument('--class_names', type=str, nargs='+', help='Class names (optional)')
    parser.add_argument('--num_classes', type=int, help='Number of classes (for validation)')
    
    args = parser.parse_args()
    
    pred_path = Path(args.predictions).expanduser().resolve()
    label_path = Path(args.labels).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    
    if not pred_path.exists():
        print(f"[ERR] Predictions file not found: {pred_path}")
        return 1
    
    if not label_path.exists():
        print(f"[ERR] Labels file not found: {label_path}")
        return 1
    
    # Load data
    print(f"[INFO] Loading predictions from: {pred_path}")
    print(f"[INFO] Loading labels from: {label_path}")
    predictions, labels = load_predictions_and_labels(pred_path, label_path)
    
    print(f"[INFO] Predictions shape: {predictions.shape}")
    print(f"[INFO] Labels shape: {labels.shape}")
    
    if len(predictions) != len(labels):
        print(f"[ERR] Mismatch: {len(predictions)} predictions vs {len(labels)} labels")
        return 1
    
    # Determine number of classes
    num_classes = args.num_classes or max(int(predictions.max()), int(labels.max())) + 1
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, labels=range(num_classes), zero_division=0))
    
    # Plot confusion matrix
    class_names = args.class_names if args.class_names else None
    plot_confusion_matrix(cm, class_names=class_names, output_path=output_path, 
                         normalize=args.normalize)
    
    # Also save normalized version if requested
    if not args.normalize:
        normalized_output = output_path.parent / f"{output_path.stem}_normalized{output_path.suffix}"
        plot_confusion_matrix(cm, class_names=class_names, output_path=normalized_output, 
                             normalize=True)
    
    print(f"[INFO] Confusion matrix generated: {output_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

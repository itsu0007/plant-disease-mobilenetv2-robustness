"""
Evaluation Metrics Computation
Calculates Top-1/Top-5 Accuracy, Precision, Recall, Macro/Weighted F1,
and Confusion Matrix for plant disease diagnostics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch

def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Computes overall accuracy, precision, recall, macro F1, and weighted F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    accuracy = float(np.mean(y_true == y_pred))
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    conf_mat = confusion_matrix(y_true, y_pred)
    
    # Detailed per-class metrics
    labels = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [class_names[i] if class_names and i < len(class_names) else str(i) for i in labels]
    
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": conf_mat.tolist(),
        "per_class_report": report_dict
    }

def compute_top_k_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5)
) -> List[float]:
    """
    Computes accuracy over the top-k predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            k_val = min(k, maxk)
            correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res

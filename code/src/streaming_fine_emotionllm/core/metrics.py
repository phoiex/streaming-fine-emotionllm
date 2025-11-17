from __future__ import annotations

import torch


@torch.no_grad()
def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Compute macro-averaged F1 without sklearn.

    Args:
        preds: shape (N,) int64 predicted class indices
        labels: shape (N,) int64 true class indices
        num_classes: total number of classes
    Returns:
        macro F1 (float)
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        if tp == 0 and (fp > 0 or fn > 0):
            f1 = 0.0
        elif tp == 0 and fp == 0 and fn == 0:
            # class not present in either; skip from avg by counting as 0
            f1 = 0.0
        else:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(sum(f1s) / len(f1s))


"""Değerlendirme metrikleri: challenge'ın kullandığı SRCC, LCC, KTAU, MSE."""
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "SRCC": spearmanr(y_true, y_pred).statistic,
        "LCC": pearsonr(y_true, y_pred).statistic,
        "KTAU": kendalltau(y_true, y_pred).statistic,
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "N": int(len(y_true)),
    }

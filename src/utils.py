"""Değerlendirme metrikleri: challenge'ın kullandığı SRCC, LCC, KTAU, MSE."""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

METRIC_COLS = ["SRCC", "LCC", "KTAU", "MSE"]


def compute_metrics(y_true, y_pred) -> dict:
    """Challenge metrikleri. Ana metrik SRCC; eşitlik LCC, KTAU, MSE ile bozulur.

    Not: tahminler sabitse (ör. MeanBaseline) korelasyonlar tanımsızdır, NaN döner.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.ptp(y_pred) == 0:  # sabit tahmin: korelasyon tanımsız
        srcc = lcc = ktau = float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            srcc = spearmanr(y_true, y_pred).statistic
            lcc = pearsonr(y_true, y_pred).statistic
            ktau = kendalltau(y_true, y_pred).statistic
    return {
        "SRCC": float(srcc),
        "LCC": float(lcc),
        "KTAU": float(ktau),
        "MSE": float(np.mean((y_true - y_pred) ** 2)),
        "N": int(len(y_true)),
    }


def evaluate_models(models: dict, train: pd.DataFrame, eval_sets: dict,
                    score_col: str = "score") -> pd.DataFrame:
    """Her modeli train üzerinde eğitir, her değerlendirme kümesinde metrikleri hesaplar.

    models    : {"Mean": MeanBaseline(), "TF-IDF + Ridge": TfidfRidge(), ...}
    eval_sets : {"validation": val_df, "test": test_df}
    Döner     : satırlar (model, küme), sütunlar SRCC, LCC, KTAU, MSE, N
    """
    rows = []
    for name, model in models.items():
        model.fit(train, train[score_col])
        for set_name, df in eval_sets.items():
            m = compute_metrics(df[score_col], model.predict(df))
            rows.append({"model": name, "set": set_name, **m})
    return pd.DataFrame(rows).set_index(["model", "set"])

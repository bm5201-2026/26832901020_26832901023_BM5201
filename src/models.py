"""Baseline modeller."""
import numpy as np


class MeanBaseline:
    """Her örnek için eğitim puanlarının ortalamasını tahmin eder (alt sınır referansı)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

"""Baseline modeller.

Bütün modeller aynı arayüzü kullanır:
    model.fit(df, y)      df: metadata tablosu (text, model_name, ... sütunları), y: puanlar
    model.predict(df)     -> numpy dizisi, df ile aynı uzunlukta

Böylece notebook'ta hepsi aynı döngüyle eğitilip değerlendirilebilir.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .data import TEXT_COL


class MeanBaseline:
    """Her örnek için eğitim puanlarının ortalamasını tahmin eder (alt sınır referansı)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class TfidfRidge:
    """Sadece metne bakan baseline: TF-IDF kelime ağırlıkları + Ridge regresyon.

    Ses dosyasını hiç kullanmaz; metnin tek başına puanı ne kadar açıkladığını gösterir.
    """

    def __init__(self, max_features: int = 3000, alpha: float = 10.0):
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.ridge = Ridge(alpha=alpha)

    def fit(self, df, y):
        X = self.tfidf.fit_transform(df[TEXT_COL].astype(str))
        self.ridge.fit(X, y)
        return self

    def predict(self, df):
        X = self.tfidf.transform(df[TEXT_COL].astype(str))
        return self.ridge.predict(X)

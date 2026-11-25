"""Baseline modeller.

Bütün modeller aynı arayüzü kullanır:
    model.fit(df, y)      df: metadata tablosu (text, model_name, ... sütunları), y: puanlar
    model.predict(df)     -> numpy dizisi, df ile aynı uzunlukta

Böylece notebook'ta hepsi aynı döngüyle eğitilip değerlendirilebilir.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MODEL_COL, TEXT_COL


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


class AudioFeatRidge:
    """A) Sadece sesin basit ölçülerine bakan model.

    7. bölümde hesapladığımız 4 ölçüyü kullanıyor: süre, ortalama ses yüksekliği,
    en yüksek ses seviyesi ve sessizlik oranı. Metne hiç bakmıyor.
    """

    def __init__(self, cols=("duration", "rms_db", "peak_db", "silence_ratio"), alpha=1.0):
        self.cols = list(cols)
        # ölçülerin birimleri farklı (saniye, dB, oran) o yüzden önce hepsini
        # aynı ölçeğe getiriyoruz (StandardScaler), sonra Ridge ile puan tahmin ediyoruz
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, df, y):
        self.model.fit(df[self.cols], y)
        return self

    def predict(self, df):
        return self.model.predict(df[self.cols])


class GeneratorMean:
    """B) Sesi hangi program ürettiyse o programın train'deki ortalama puanını verir.

    Sese de metne de bakmıyor, sadece program adını kullanıyor.
    Amacımız bu bilginin tek başına ne kadar işe yaradığını görmek. Yarışmanın
    asıl testinde programlar yeni olacağı için bu yöntem orada çalışmaz.
    """

    def fit(self, df, y):
        tablo = pd.DataFrame({"program": df[MODEL_COL].to_numpy(), "puan": np.asarray(y, dtype=float)})
        # her programın ortalama puanı
        self.ortalamalar_ = tablo.groupby("program")["puan"].mean().to_dict()
        # hiç görmediğimiz bir program gelirse kullanmak için genel ortalama
        self.genel_ortalama_ = float(tablo["puan"].mean())
        return self

    def predict(self, df):
        tahmin = df[MODEL_COL].map(self.ortalamalar_)
        # train'de olmayan program -> genel ortalama
        return tahmin.fillna(self.genel_ortalama_).to_numpy(dtype=float)


class ClapSimilarity:
    """C) CLAP ile ses ve metnin ne kadar benzediğine bakan model.

    Benzerlik değerleri önceden src/features.py ile hesaplanıp tabloya
    "clap_sim" sütunu olarak eklenmiş olmalı.

    Benzerlik -1 ile 1 arasında, puanlar ise 0-10 arasında. fit() sadece bu ölçeği
    çevirmek için basit bir doğru uyduruyor (puan = a * benzerlik + b).
    Sıralama değişmediği için SRCC, eğitimsiz benzerlikle aynı çıkıyor.
    """

    def __init__(self, col="clap_sim"):
        self.col = col

    def fit(self, df, y):
        self.olcek = LinearRegression().fit(df[[self.col]], y)
        return self

    def predict(self, df):
        return self.olcek.predict(df[[self.col]])

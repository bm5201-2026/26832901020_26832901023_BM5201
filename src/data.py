"""Veri yükleme yardımcıları (XACLE 2027 development dataset).

Beklenen klasör yapısı:
    <root>/meta_data/{split}.csv           dinleyici bazlı puanlar
    <root>/meta_data/{split}_average.csv   ses başına ortalama puan
    <root>/wav/{split}/<wave_file_name>
"""
from pathlib import Path

import pandas as pd
import soundfile as sf

DEFAULT_ROOT = Path("datasets/XACLE_Challenge_2027_development_dataset")
SPLITS = ("train", "validation", "test")

FILE_COL = "wave_file_name"
TEXT_COL = "text"
MODEL_COL = "model_name"
LISTENER_COL = "listener_id"
SCORE_COL = "score"  # yüklerken puan sütunu bu ada çevrilir


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split {SPLITS} içinden biri olmalı, gelen: {split!r}")


def _normalize_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """Puan sütununun adı dosyalar arasında farklı yazılmış olabilir
    (ör. 'average_semantic_alignement_score'); adı 'score' yapar."""
    candidates = [c for c in df.columns if "score" in c.lower()]
    if len(candidates) != 1:
        raise ValueError(f"Puan sütunu belirlenemedi, sütunlar: {list(df.columns)}")
    return df.rename(columns={candidates[0]: SCORE_COL})


def load_metadata(split: str, root: Path | str = DEFAULT_ROOT, average: bool = True) -> pd.DataFrame:
    """Metadata CSV'sini okur.

    average=True  -> {split}_average.csv (ses başına tek satır, ortalama puan)
    average=False -> {split}.csv (her dinleyici puanı ayrı satır)
    """
    _check_split(split)
    name = f"{split}_average.csv" if average else f"{split}.csv"
    df = pd.read_csv(Path(root) / "meta_data" / name)
    df = _normalize_score_column(df)
    df["split"] = split
    return df


def load_all(root: Path | str = DEFAULT_ROOT, average: bool = True) -> pd.DataFrame:
    """Üç split'i tek tabloda birleştirir (EDA için)."""
    return pd.concat([load_metadata(s, root, average) for s in SPLITS], ignore_index=True)


def audio_path(split: str, file_name: str, root: Path | str = DEFAULT_ROOT) -> Path:
    """Bir ses dosyasının tam yolu: wav/{split}/{file_name}."""
    _check_split(split)
    return Path(root) / "wav" / split / file_name


def audio_info(path: Path | str) -> dict:
    """Dosyayı decode etmeden süre, örnekleme hızı ve kanal sayısını okur."""
    info = sf.info(str(path))
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }

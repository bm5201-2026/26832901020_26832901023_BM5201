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

FILE_COL = "wav_file_name"
TEXT_COL = "text"
SCORE_COL = "score"
LISTENER_COL = "listener_id"
MODEL_ID_COL = "model_id"
MODEL_COL = "model_name"

# CSV'lerde aynı bilgi farklı adlarla yazılmış (ör. validation.csv'de 'Caption', 'REL',
# 'listener id'). Hepsini yukarıdaki standart adlara çeviriyoruz.
_COLUMN_ALIASES = {
    "caption": TEXT_COL,
    "rel": SCORE_COL,
    "average_score": SCORE_COL,
    "listener id": LISTENER_COL,
    "model_id": MODEL_ID_COL,
}
_EXPECTED = {
    True: [FILE_COL, TEXT_COL, SCORE_COL, MODEL_ID_COL, MODEL_COL],
    False: [FILE_COL, TEXT_COL, SCORE_COL, LISTENER_COL, MODEL_ID_COL, MODEL_COL],
}


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split {SPLITS} içinden biri olmalı, gelen: {split!r}")


def _normalize_columns(df: pd.DataFrame, average: bool) -> pd.DataFrame:
    """Sütun adlarını standart hale getirir ve beklenen sütunların varlığını kontrol eder."""
    df = df.rename(columns={c: _COLUMN_ALIASES.get(c.strip().lower(), c.strip()) for c in df.columns})
    missing = [c for c in _EXPECTED[average] if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütun(lar): {missing}; mevcut sütunlar: {list(df.columns)}")
    return df


def load_metadata(split: str, root: Path | str = DEFAULT_ROOT, average: bool = True) -> pd.DataFrame:
    """Metadata CSV'sini okur.

    average=True  -> {split}_average.csv (ses başına tek satır, ortalama puan)
    average=False -> {split}.csv (her dinleyici puanı ayrı satır)
    """
    _check_split(split)
    name = f"{split}_average.csv" if average else f"{split}.csv"
    df = pd.read_csv(Path(root) / "meta_data" / name)
    df = _normalize_columns(df, average)
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

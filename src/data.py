"""Veri yükleme yardımcıları (XACLE 2027 development dataset)."""
from pathlib import Path

import pandas as pd

DEFAULT_ROOT = Path("datasets/XACLE_Challenge_2027_development_dataset")
SPLITS = ("train", "validation", "test")


def load_metadata(split: str, root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """meta_data/{split}_average.csv dosyasını okur."""
    if split not in SPLITS:
        raise ValueError(f"split {SPLITS} içinden biri olmalı, gelen: {split!r}")
    return pd.read_csv(Path(root) / "meta_data" / f"{split}_average.csv")


def audio_path(split: str, file_name: str, root: Path | str = DEFAULT_ROOT) -> Path:
    """Bir ses dosyasının tam yolunu döndürür: wav/{split}/{file_name}."""
    return Path(root) / "wav" / split / file_name

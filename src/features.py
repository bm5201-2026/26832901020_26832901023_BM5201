"""CLAP ile ses ve metin arasındaki benzerliği hesaplama.

CLAP, sesi ve metni aynı türden sayı dizilerine (vektöre) çeviren hazır bir model.
Ses ile metin aynı şeyi anlatıyorsa vektörleri birbirine yakın çıkıyor.
Biz de iki vektörün ne kadar benzediğine bakıyoruz (kosinüs benzerliği).
Bu adımda hiç eğitim yapılmıyor, model olduğu gibi kullanılıyor.

İki farklı CLAP modeli var:
- C1) LAION CLAP (yarışma dışı, Hugging Face'ten): compute_clap_similarity
- C2) M2D-CLAP (yarışmanın resmi baseline'ındaki model): compute_m2d_similarity
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FILE_COL, TEXT_COL, audio_path

CLAP_NAME = "laion/clap-htsat-unfused"   # Hugging Face'teki hazır CLAP modeli
CLAP_SR = 48000                          # bu model 48 kHz ses istiyor, bizim sesler 16 kHz
CLAP_SEC = 10                            # model 10 saniyelik ses bekliyor


def _vektor(cikti):
    """transformers sürümüne göre çıktı ya direkt tensör ya da içinde tensör olan bir nesne oluyor."""
    import torch
    if torch.is_tensor(cikti):
        return cikti
    return cikti.pooler_output


def compute_clap_similarity(df: pd.DataFrame, root, batch_size: int = 32, device=None) -> pd.DataFrame:
    """df'teki her (ses, metin) çifti için CLAP benzerliğini hesaplar.

    Döner: split, wav_file_name, clap_sim sütunlu tablo.
    GPU ile ~15 dakika sürüyor, CPU'da çok daha uzun.
    """
    import librosa
    import torch
    from tqdm.auto import tqdm
    from transformers import ClapModel, ClapProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device)

    model = ClapModel.from_pretrained(CLAP_NAME).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP_NAME)

    # 1) Metinler
    # aynı cümle birkaç kere geçiyor, her farklı cümleyi bir kere hesaplamak yeterli
    metinler = df[TEXT_COL].astype(str).unique().tolist()
    metin_vektor = {}
    for i in tqdm(range(0, len(metinler), batch_size), desc="Metinler"):
        parca = metinler[i:i + batch_size]
        girdi = processor.tokenizer(parca, padding=True, truncation=True, return_tensors="pt")
        girdi = {k: v.to(device) for k, v in girdi.items()}
        with torch.no_grad():
            v = _vektor(model.get_text_features(**girdi))
        # uzunluğu 1 yapıyoruz, böylece çarpım direkt kosinüs benzerliği oluyor
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        for metin, vek in zip(parca, v):
            metin_vektor[metin] = vek

    # 2) Sesler
    satirlar = list(df[["split", FILE_COL, TEXT_COL]].itertuples(index=False, name=None))
    sonuc = []
    for i in tqdm(range(0, len(satirlar), batch_size), desc="Sesler"):
        parca = satirlar[i:i + batch_size]
        sesler = []
        for split, dosya, _ in parca:
            # sesi okurken 48 kHz'e çeviriyoruz
            y, _ = librosa.load(audio_path(split, dosya, root), sr=CLAP_SR)
            # 10 saniyeden uzun olanların fazlasını kesiyoruz (en fazla 0,24 sn)
            sesler.append(y[:CLAP_SR * CLAP_SEC])
        girdi = processor.feature_extractor(sesler, sampling_rate=CLAP_SR, return_tensors="pt")
        girdi = {k: v.to(device) for k, v in girdi.items()}
        with torch.no_grad():
            v = _vektor(model.get_audio_features(**girdi))
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

        # ses vektörü ile kendi metninin vektörü arasındaki benzerlik
        for (split, dosya, metin), ses_vek in zip(parca, v):
            benzerlik = float(np.dot(ses_vek, metin_vektor[str(metin)]))
            sonuc.append({"split": split, FILE_COL: dosya, "clap_sim": benzerlik})

    return pd.DataFrame(sonuc)


# ---------------------------------------------------------------------------
# C2) M2D-CLAP
# Yarışmanın 2027 resmi baseline'ında kullanılan model. 16 kHz ses bekliyor,
# bizim sesler de 16 kHz, yani dönüştürmeye gerek yok.
# ---------------------------------------------------------------------------
M2D_NAME = "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"   # resmi baseline'daki checkpoint
M2D_PY_URL = "https://raw.githubusercontent.com/nttcslab/m2d/master/examples/portable_m2d.py"
M2D_ZIP_URL = f"https://github.com/nttcslab/m2d/releases/download/v0.5.0/{M2D_NAME}.zip"
M2D_SR = 16000
M2D_SEC = 10   # resmi baseline da 10 saniye kullanıyor


def download_m2d(folder) -> Path:
    """M2D-CLAP'ı yükleyen kodu ve modelin ağırlıklarını indirir.

    Dosyalar zaten varsa tekrar indirmiyor. Drive'da bir klasör verirsek
    sadece ilk seferde iniyor.
    Döner: checkpoint-30.pth dosyasının yolu
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # 1) modeli yükleyen tek dosyalık kod
    py_file = folder / "portable_m2d.py"
    if not py_file.exists():
        print("İndiriliyor: portable_m2d.py")
        urllib.request.urlretrieve(M2D_PY_URL, py_file)

    # 2) modelin ağırlıkları (zip içinde)
    ckpt_dir = folder / M2D_NAME
    bulunan = list(ckpt_dir.rglob("checkpoint-30.pth")) if ckpt_dir.exists() else []
    if not bulunan:
        zip_file = folder / f"{M2D_NAME}.zip"
        print("İndiriliyor:", zip_file.name, "(birkaç dakika sürebilir)")
        urllib.request.urlretrieve(M2D_ZIP_URL, zip_file)
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(folder)
        zip_file.unlink()   # zip'e artık gerek yok
        bulunan = list(ckpt_dir.rglob("checkpoint-30.pth"))
    if not bulunan:
        raise FileNotFoundError(f"checkpoint-30.pth bulunamadı: {ckpt_dir}")

    # M2D kodu hangi metin modelini kullanacağını klasör adından anlıyor,
    # o yüzden dosyanın klasör adı M2D_NAME olmalı
    ckpt = next((p for p in bulunan if p.parent.name == M2D_NAME), bulunan[0])
    print("Checkpoint:", ckpt)
    return ckpt


def compute_m2d_similarity(df: pd.DataFrame, root, m2d_folder, batch_size: int = 32,
                           emb_path=None, device=None) -> pd.DataFrame:
    """df'teki her (ses, metin) çifti için M2D-CLAP benzerliğini hesaplar.

    m2d_folder : modelin indirileceği / indirildiği klasör
    emb_path   : verilirse ses ve metin vektörleri de .npz olarak kaydediliyor
                 (görülmemiş üretici testi ve Assignment 2 için lazım olacak)
    Döner      : split, wav_file_name, m2d_sim sütunlu tablo
    """
    import librosa
    import torch
    from tqdm.auto import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device)

    # modeli indir ve yükle
    ckpt = download_m2d(m2d_folder)
    sys.path.insert(0, str(Path(m2d_folder)))   # portable_m2d.py'yi import edebilmek için
    from portable_m2d import PortableM2D

    # flat_features=True: sadece CLAP vektörleri lazım (resmi baseline da böyle kullanıyor)
    model = PortableM2D(weight_file=str(ckpt), flat_features=True).to(device).eval()

    # 1) Metinler
    # aynı cümle birkaç kere geçiyor, her farklı cümleyi bir kere hesaplıyoruz
    # (metin kısmı için gereken BERT modeli ilk seferde otomatik iniyor)
    metinler = df[TEXT_COL].astype(str).unique().tolist()
    metin_vektor = {}
    for i in tqdm(range(0, len(metinler), batch_size), desc="Metinler"):
        parca = metinler[i:i + batch_size]
        with torch.no_grad():
            v = model.encode_clap_text(parca, truncate=True)
        # uzunluğu 1 yapıyoruz, böylece çarpım direkt kosinüs benzerliği oluyor
        v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
        for metin, vek in zip(parca, v):
            metin_vektor[metin] = vek

    # 2) Sesler
    uzunluk = M2D_SR * M2D_SEC   # 10 saniye = 160.000 örnek
    satirlar = list(df[["split", FILE_COL, TEXT_COL]].itertuples(index=False, name=None))
    sonuc, ses_vektorleri = [], []
    for i in tqdm(range(0, len(satirlar), batch_size), desc="Sesler"):
        parca = satirlar[i:i + batch_size]
        sesler = []
        for split, dosya, _ in parca:
            y, _ = librosa.load(audio_path(split, dosya, root), sr=M2D_SR, mono=True)
            # resmi baseline gibi: 10 sn'den uzunsa kes, kısaysa sonuna sessizlik (0) ekle
            y = y[:uzunluk]
            if len(y) < uzunluk:
                y = np.pad(y, (0, uzunluk - len(y)))
            sesler.append(y)
        batch = torch.tensor(np.stack(sesler), dtype=torch.float32, device=device)  # (batch, 160000)
        with torch.no_grad():
            v = model.encode_clap_audio(batch)
        v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()

        for (split, dosya, metin), ses_vek in zip(parca, v):
            benzerlik = float(np.dot(ses_vek, metin_vektor[str(metin)]))
            sonuc.append({"split": split, FILE_COL: dosya, "m2d_sim": benzerlik})
            ses_vektorleri.append(ses_vek)

    sonuc = pd.DataFrame(sonuc)

    # vektörleri de kaydedelim, sonra tekrar 15 dk beklemeyelim
    if emb_path is not None:
        np.savez_compressed(
            emb_path,
            split=sonuc["split"].to_numpy(),
            wav_file_name=sonuc[FILE_COL].to_numpy(),
            audio_emb=np.stack(ses_vektorleri).astype(np.float32),
            text_emb=np.stack([metin_vektor[str(t)] for _, _, t in satirlar]).astype(np.float32),
        )
        print("Vektörler kaydedildi:", emb_path)

    return sonuc

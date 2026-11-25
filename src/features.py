"""CLAP ile ses ve metin arasındaki benzerliği hesaplama.

CLAP, sesi ve metni aynı türden sayı dizilerine (vektöre) çeviren hazır bir model.
Ses ile metin aynı şeyi anlatıyorsa vektörleri birbirine yakın çıkıyor.
Biz de iki vektörün ne kadar benzediğine bakıyoruz (kosinüs benzerliği).
Bu adımda hiç eğitim yapılmıyor, model olduğu gibi kullanılıyor.
"""
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

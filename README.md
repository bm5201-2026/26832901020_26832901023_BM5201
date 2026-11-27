# XACLE Challenge 2027 – Text–Audio Alignment Prediction

BM5201 Data Mining dersi dönem projesi. [XACLE Challenge 2027](https://xacle.org/2027/)
(IEEE ICASSP 2027, The 2nd X-to-Audio Alignment Challenge) kapsamında, bir ses ile
metin açıklaması arasındaki anlamsal uyum için insanların vereceği puanı (0-10) tahmin eden modeller geliştiriyoruz.

## Grup Üyeleri

| Ad Soyad | Öğrenci No | GitHub |
|---|---|---|
| Aleyna İzgi | 26832901020 | [aley22](https://github.com/aley22) |
| Zeynep Saliha Sertçelik | 26832901023 | [deafiduck](https://github.com/deafiduck) |

## Yarışma ve Veri Kümesi

- **Görev:** (ses, metin) çifti için insan uyum puanını tahmin etmek. Ana metrik **SRCC**; eşitlik durumunda LCC, KTAU ve MSE kullanılır.
- **Veri:** XACLE Challenge 2027 development dataset. 13.500 ses-metin çifti (train 7.500 / validation 3.000 / test 3.000).
  - 4.436 farklı metin; her metin ortalama 3 farklı sesle eşleşiyor.
  - 8 üretici model (TangoFlux, Tango, Tango 2, AudioLDM, AudioLDM 2, AudioGen, Make-An-Audio 2, Stable Audio) + 1 gerçek kayıt kaynağı (AudioCaps).
  - Her ses train ve validation'da 4, test'te 8 dinleyici tarafından puanlanmış; hedef değişken bu puanların ortalaması.
  - Sesler 16 kHz, mono, yaklaşık 10 saniye.
- **2027'nin farkı:** Final değerlendirme setinde **eğitimde görülmemiş üretici modeller** olacak.

Veri repoya eklenmez.

## Repo Yapısı

```
README.md
requirements.txt
assignment1_eda_baseline.ipynb   # Ödev 1: EDA ve baseline'lar
src/
  data.py       # veri yükleme, ses ölçüleri
  features.py   # CLAP ses-metin benzerliği (LAION CLAP ve M2D-CLAP)
  models.py     # baseline modeller
  utils.py      # metrikler ve toplu değerlendirme
```

## Çalıştırma

1. Notebook'u Colab'da açın, **GPU (T4)** çalışma zamanı seçin.
2. Veri zip'ini Drive'da `MyDrive/XACLE/` veya `MyDrive/Colab Notebooks/XACLE/` klasörüne koyun.
3. **Tümünü çalıştır.** İlk hücre repoyu klonlar ve `src/` modüllerini kullanılabilir hale getirir.
4. CLAP benzerlikleri ilk çalıştırmada hesaplanır (GPU ile model başına 15-20 dk) ve Drive'a kaydedilir; sonraki çalıştırmalarda dosyadan okunur.

Yerelde: `pip install -r requirements.txt`, ardından notebook'u çalıştırın.

## Yaklaşım (Ödev 1)

**EDA:** veri kalitesi (eksik/tekrar), puan dağılımları, dinleyiciler arası farklılık, üretici modellere göre puanlar,
metin uzunlukları ve bölümler arası ortak metinler, ses formatı ve teknik ölçüler (süre, RMS, tepe seviyesi, sessizlik oranı).

Öne çıkan bulgular:
- Eksik değer ve tekrar eden satır yok.
- Üreticilerin ortalama puanları çok farklı (TangoFlux 8,33, AudioLDM 5,46).
- Sesin teknik özellikleri üreticiyi ele veriyor: yalnızca 4 basit ölçüyle üretici %59 doğrulukla tahmin edilebiliyor (şans düzeyi %33). Örneğin her üretici hep aynı süreyi üretiyor.

**Baseline'lar** (hepsi `src/models.py` içinde, aynı `fit(df, y)` / `predict(df)` arayüzüyle):

| Baseline | Neye bakıyor |
|---|---|
| Mean | Hiçbir şeye; herkese train ortalamasını verir |
| TF-IDF + Ridge | Sadece metne |
| A) Ses ölçüleri + Ridge | Sadece sesin teknik ölçülerine |
| B) Üretici ortalaması | Sadece sesi üreten modelin adına |
| C1) LAION CLAP benzerliği | Ses + metin, eğitimsiz (yarışma dışı model) |
| C2) M2D-CLAP benzerliği | Ses + metin, eğitimsiz (yarışmanın resmi baseline'ındaki model, 16 kHz) |

## Sonuçlar (validation)

| Model | SRCC | LCC | KTAU | MSE |
|---|---|---|---|---|
| Mean | – | – | – | 5.674 |
| TF-IDF + Ridge | 0.222 | 0.215 | 0.152 | 5.401 |
| A) Ses ölçüleri + Ridge | 0.179 | 0.127 | 0.124 | 5.592 |
| B) Üretici ortalaması | 0.406 | 0.430 | 0.301 | 4.620 |
| C1) LAION CLAP benzerliği | 0.430 | 0.486 | 0.297 | 4.365 |
| C2) M2D-CLAP benzerliği | 0.427 | 0.492 | 0.295 | 4.319 |
| Resmi 2026 baseline* | 0.384 | 0.396 | 0.265 | 4.836 |
| Resmi 2027 M2D-CLAP baseline* | 0.584 | 0.600 | 0.419 | 3.637 |

\* [XACLE 2027 resmi baseline sayfasında](https://github.com/XACLE-Challenge/XACLE_2027_baselinemodel) raporlanan değerler.
Mean sabit tahmin verdiği için korelasyon metrikleri tanımsızdır.

**Yorum:**
- Sadece üretici adını bilmek (B, 0,41) sesi ve metni karşılaştırmakla (C, 0,43) neredeyse aynı sonucu veriyor. Görülmemiş üreticilerde bu kısayol çalışmayacağı için validation skorları dikkatli yorumlanmalı.
- İki farklı CLAP modeli aynı sonucu verdi; eğitimsiz benzerlik yaklaşık 0,43'te kalıyor.
- Aynı M2D-CLAP modelinin üstüne insan puanlarıyla eğitilen küçük bir kısım ekleyen resmi baseline 0,58'e çıkıyor. **Ödev 2'nin yönü:** M2D-CLAP vektörleri üstüne eğitilen bir puan modeli ve görülmemiş üretici testi.

## İlerleme Günlüğü

| Tarih | Yapılan |
|---|---|
| 12.11.2026 | Proje iskeleti: README, `src` modülleri |
| 14.11.2026 | Veri yükleme (gerçek CSV sütunları, dinleyici bazlı puanlar); EDA: veriye genel bakış |
| 15.11.2026 | Puan dağılımları ve grafikler |
| 18.11.2026 | Üretici modellere göre analiz, metin analizi; ses formatı ve teknik ölçüler, üretici parmak izi, spektrogramlar |
| 19.11.2026 | İlk baseline denemeleri (Mean, TF-IDF + Ridge) |
| 25.11.2026 | Baseline kodu `src/`'ye taşındı; A, B ve LAION CLAP (C1) baseline'ları; M2D-CLAP (C2) eklendi |
| 27.11.2026 | Baseline bulguları ve README güncellendi |

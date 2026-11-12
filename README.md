# XACLE Challenge 2027 – Text–Audio Alignment Prediction

BM5201 Data Mining dersi dönem projesi. [XACLE Challenge 2027](https://xacle.org/2027/)
(IEEE ICASSP 2027 – The 2nd X-to-Audio Alignment Challenge) kapsamında, bir ses ile
metin açıklaması arasındaki anlamsal uyum için insan puanını tahmin eden modeller geliştiriyoruz.

## Grup Üyeleri

| Ad Soyad | Öğrenci No | GitHub |
|---|---|---|
| | | |
| | | |

## Veri Kümesi

XACLE Challenge 2027 development dataset (train 7.500 / validation 3.000 / test 3.000).
Veri repoya eklenmez; `datasets/XACLE_Challenge_2027_development_dataset/` altına indirilmelidir.

## Repo Yapısı

```
README.md
assignment1_eda_baseline.ipynb   # Ödev 1: EDA ve baseline'lar
assignment2_solution.ipynb       # Ödev 2: önerilen çözüm
src/
  data.py     # veri yükleme
  models.py   # modeller
  utils.py    # metrikler ve yardımcılar
```

## Çalıştırma

1. `pip install -r requirements.txt`
2. Veriyi `datasets/` altına indirin.
3. Notebook'ları sırayla çalıştırın (Colab GPU önerilir).

## Yaklaşım

_(doldurulacak)_

## Sonuçlar (validation)

| Model | SRCC | LCC | KTAU | MSE |
|---|---|---|---|---|
| Mean baseline | | | | |
| Resmi M2D-CLAP baseline (referans) | 0.584 | 0.600 | 0.419 | 3.637 |

## İlerleme Günlüğü

| Tarih | Yapılan |
|---|---|
| | Proje iskeleti oluşturuldu |

## Gönderim Geçmişi

| Tarih | Sistem | Skor |
|---|---|---|
| | | |

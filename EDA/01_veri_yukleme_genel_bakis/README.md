# 01 — Veri Yükleme & Genel Bakış

**Bağımsız klasör** — çalışması için başka hiçbir klasöre ihtiyaç duymaz.

[Teen Mental Health Dataset](https://www.kaggle.com/datasets/argonnxx/teen-mental-health) veri setini Kaggle'dan **otomatik olarak** (`kagglehub` ile, manuel indirme yok) çeker ve veriye ilk bakışı yapar.

## Yapılanlar
- `kagglehub.dataset_download()` ile otomatik indirme
- `head() / tail() / sample()` ile ilk bakış
- Veri tipi, eksik değer sayısı/oranı, unique değer tablosu
- `df.info()` ve `df.describe()` istatistik özetleri
- Eksik değer görselleştirmesi

## Kurulum
```bash
pip install -r requirements.txt
```
Kaggle API kimlik bilgileriniz tanımlı olmalı (`~/.kaggle/kaggle.json` ya da `kagglehub.login()`).

## Çalıştırma
```bash
python veri_yukleme.py
```

## Görseller

Bu klasördeki `figures/` içinde gerçek örnek PNG'ler zaten mevcut (sentetik demo veriyle önceden üretildi, script'i kendi Kaggle bağlantınızla çalıştırdığınızda gerçek veriyle üzerine yazılır):

![Eksik Değer Haritası](figures/01_eksik_deger_haritasi.png)
![Sütun Bazında Eksik Değer](figures/01_sutun_eksik_deger_sayisi.png)

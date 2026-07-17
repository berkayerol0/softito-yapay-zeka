# 03 — Tek Değişkenli Analiz (Univariate)

**Bağımsız klasör** — çalışması için başka hiçbir klasöre ihtiyaç duymaz. Veriyi kendisi indirir ve temizler.

## Yapılanlar
- Kaggle'dan otomatik indirme + temel temizlik (duplicate/eksik değer)
- Sayısal değişkenler: histogram + KDE eğrisi, çarpıklık (skewness) değeri
- Sayısal değişkenler için detaylı istatistik özeti (ortalama, medyan, std, min, max)
- Kategorik değişkenler: yüzdelik bar chart (en sık 10 kategori)

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python tek_degiskenli_analiz.py
```

## Görseller

![Sayısal Değişken Dağılımları](figures/03_sayisal_degisken_dagilimlari.png)
![Kategorik Değişken Dağılımları](figures/03_kategorik_degisken_dagilimlari.png)

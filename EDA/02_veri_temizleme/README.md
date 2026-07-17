# 02 — Veri Temizleme

**Bağımsız klasör** — çalışması için başka hiçbir klasöre ihtiyaç duymaz. Veriyi kendisi indirir.

## Yapılanlar
- Kaggle'dan otomatik indirme
- Duplicate (tekrar eden) satır kontrolü ve temizliği
- Eksik değer tablosu (sayı + oran)
- Eksik değer doldurma: sayısal → medyan, kategorik → mod
- IQR yöntemiyle aykırı değer (outlier) tespiti ve boxplot görselleştirmesi

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python veri_temizleme.py
```

## Görseller

![Aykırı Değer Boxplot](figures/02_aykiri_deger_boxplot.png)

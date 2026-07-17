# 04 — Çift Değişkenli Analiz (Bivariate)

**Bağımsız klasör** — çalışması için başka hiçbir klasöre ihtiyaç duymaz. Veriyi kendisi indirir ve temizler.

## Yapılanlar
- Kaggle'dan otomatik indirme + temel temizlik (duplicate/eksik değer)
- Sayısal değişkenler arası korelasyon matrisi (heatmap)
- Cinsiyete göre kırılımlı scatter plot
- Kategorik × kategorik ilişki (crosstab, stacked bar, yüzdelik)
- Kategorik × sayısal ilişki (violin plot + box plot)

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python cift_degiskenli_analiz.py
```

## Görseller

![Korelasyon Heatmap](figures/04_korelasyon_heatmap.png)
![Scatter İlişki](figures/04_scatter_iliski.png)
![Kategorik-Kategorik İlişki](figures/04_kategorik_kategorik_iliski.png)
![Kategorik-Sayısal İlişki](figures/04_kategorik_sayisal_iliski.png)

# 05 — Feature Engineering (Yeni Özellik Üretme)

**Bağımsız klasör** — çalışması için başka hiçbir klasöre ihtiyaç duymaz. Veriyi kendisi indirir ve temizler.

## Yapılanlar
- Kaggle'dan otomatik indirme + temel temizlik (duplicate/eksik değer)
- `age_group`: yaş gruplama (13-14 / 15-16 / 17-19)
- `screen_sleep_ratio`: sosyal medya saati / uyku saati oranı
- `risk_score` + `risk_category`: stres + anksiyete + bağımlılık toplamından birleşik risk skoru (Düşük/Orta/Yüksek)
- `gender_encoded`: kategorik değişken için label encoding örneği

## Kurulum
```bash
pip install -r requirements.txt
```

## Çalıştırma
```bash
python feature_engineering.py
```
Çıktı olarak bu klasöre `processed_data.csv` (tüm yeni özelliklerle final veri seti) kaydedilir.

## Görseller

![Yeni Özellik Dağılımları](figures/05_yeni_ozellik_dagilimlari.png)
![Risk Kategorisi Dağılımı](figures/05_risk_kategorisi_dagilimi.png)

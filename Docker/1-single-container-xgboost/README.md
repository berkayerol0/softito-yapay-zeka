# 🐳 student-ai-burnout-docker

> Öğrencilerde yapay zekâ (GenAI) kullanımının **tükenmişlik riskine** etkisini tahmin eden, **tek bir Dockerfile** ile konteynerize edilmiş bir XGBoost sınıflandırma modeli.

---

## 📌 Proje Ne Yapıyor?

50.000 öğrenciye ait veriyle (haftalık AI kullanım saati, GPA, sınav kaygısı, AI bağımlılığı algısı vb.) her öğrencinin **tükenmişlik riskini** (`High` / `Medium` / `Low`) tahmin eden bir **XGBoost** modeli eğitiyor. Model; veri temizleme, encoding, eğitim, değerlendirme ve görselleştirme adımlarının tamamını **tek komutla** çalıştırıyor.

---

## 📂 Dizin Yapısı

```
student-ai-burnout-docker/
├── Dockerfile                          # Tek servis: bu Python dosyasını çalıştırır
├── xgboost_docker.py                   # Ana script (veri → model → sonuç)
├── ai_student_impact_dataset.csv        # 50.000 satırlık veri seti
├── requirements.txt
├── .gitignore
├── README.md
│
├── output/                             # Script çalışınca üretilen çıktılar
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── predictions.csv
│
└── extras/                             # Docker'a dahil değil, ayrı bir pratik
    └── rfm_segmentation.py             # RFM + K-Means müşteri segmentasyonu (bonus)
```

---

## 🐳 Dockerfile Ne Yapıyor?

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY xgboost_docker.py .
COPY ai_student_impact_dataset.csv .

CMD ["python", "xgboost_docker.py"]
```

Adım adım:
1. Hafif bir Python 3.10 imajından başlar.
2. Gerekli kütüphaneleri (`requirements.txt`) kurar.
3. Script ve veri dosyasını konteynerin içine kopyalar.
4. Konteyner ayağa kalktığında otomatik olarak `xgboost_docker.py`'yi çalıştırır.

**Tek Dockerfile → tek Python dosyası → tek iş.** Docker Compose'a gerek yok çünkü ortada birden fazla servis (veritabanı, web sunucusu vb.) yok, sadece bir hesaplama görevi var.

---

## 🚀 Nasıl Çalıştırılır?

```bash
# 1) İmajı oluştur
docker build -t student-burnout-model .

# 2) Çalıştır (çıktıları lokal 'output' klasörüne almak için volume bağla)
docker run --rm -v "$(pwd)/output:/app/output" student-burnout-model
```

Çalışma bitince `output/` klasöründe grafikleri ve tahmin sonuçlarını bulursun.

---

## 📊 Sonuçlar

En etkili değişken açık ara **haftalık GenAI kullanım saati**:

![Feature Importance](output/feature_importance.png)

Confusion matrix (test setinde ~10.000 örnek):

![Confusion Matrix](output/confusion_matrix.png)

| Metrik | Değer |
|---|---|
| Doğruluk (Accuracy) | ~%52 |
| En önemli değişken | `Weekly_GenAI_Hours` |

> Not: Amaç en yüksek doğruluğu yakalamak değil; bir ML iş akışını **Docker ile paketleyip tekrarlanabilir hale getirmek.**

---

## 🎁 Bonus: RFM Müşteri Segmentasyonu (extras/)

`extras/rfm_segmentation.py`, bu Dockerfile'a **dahil değildir** — ayrı bir pratik olarak repoda duruyor. Klasik RFM (Recency/Frequency/Monetary) + K-Means ile e-ticaret müşterilerini segmentlere ayırır. Gerçek veri (`online_retail.xlsx`) yoksa otomatik olarak örnek/sentetik veri üretip demo amaçlı çalışır:

```bash
python extras/rfm_segmentation.py
```

---

## 🛠️ Kullanılan Teknolojiler

`Python 3.10` · `Docker` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn`

---

<p align="center"><i>Docker & ML öğrenme amaçlı bir portföy projesidir.</i></p>

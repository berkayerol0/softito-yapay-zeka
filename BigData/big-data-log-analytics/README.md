# big-data-log-analytics

> PySpark ile web sunucusu loglarının büyük ölçekte üretilip analiz edildiği bir big data pipeline'ı.

---

## 📌 Proje Ne Yapıyor?

İki aşamalı bir iş akışı:

1. **`generate_logs.py`** — 5.000.000 satırlık sentetik web sunucusu log verisi üretir, 10 ayrı CSV dosyasına böler (`logs/weblogs_000.csv` → `weblogs_009.csv`, her biri 500.000 satır). Her satır: IP, timestamp, HTTP metodu, URL, status kodu, yanıt süresi, user-agent, ülke, gönderilen byte bilgisini içerir.

2. **`analyze_logs.py`** — PySpark ile bu logları okuyup 10 farklı analiz çalıştırır:
   - HTTP durum kodu dağılımı
   - En çok istek alan URL'ler (Top 10)
   - Ülkelere göre istek sayısı ve ortalama yanıt süresi
   - Saatlik istek dağılımı
   - Yavaş URL'ler (ortalama yanıt > 2000ms)
   - HTTP hata oranları (4xx/5xx)
   - En aktif IP'ler (Top 10)
   - Partition sayısı ve toplam veri boyutu

---

## 📂 Dizin Yapısı

```
big-data-log-analytics/
├── generate_logs.py
├── analyze_logs.py
├── requirements.txt
├── .gitignore
├── README.md
└── logs/                  # generate_logs.py çalıştırılınca burada oluşur (repoya dahil değil)
```

---

## 🚀 Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt

# 1) Log verisini üret (5M satır, 10 dosya, logs/ klasörüne)
python generate_logs.py

# 2) Analizi çalıştır
python analyze_logs.py
```

> Not: `logs/` klasörü `.gitignore` ile hariç tutulur — 5M satırlık veri repoya dahil edilmez. Projeyi klonlayan herkes `generate_logs.py`'yi çalıştırarak kendi veri setini üretir.

---

## ⚙️ Teknik Detaylar

- Spark **Adaptive Query Execution** (`spark.sql.adaptive.enabled`) açık — sorgu planı çalışma zamanında optimize edilir.
- `analyze_logs.py` her adımın süresini ölçüp raporlar (`time.time()` ile).
- Veri `logs/*.csv` üzerinden tek seferde okunur; 10 dosya tek bir DataFrame'e birleşir.

---

## 🛠️ Kullanılan Teknolojiler

`Python` · `PySpark` · `Apache Spark SQL`

---

<p align="center"><i>Big data / PySpark pratiği amaçlı bir portföy projesidir.</i></p>

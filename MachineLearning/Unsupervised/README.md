# 🧩 Unsupervised Öğrenme

Etiketlenmemiş veride yapı/grup keşfeden **denetimsiz öğrenme** projelerinin bulunduğu klasör. Kümeleme (clustering), boyut indirgeme (dimensionality reduction) ve anomali tespiti (anomaly detection) tekniklerini kapsar.

---

## 📁 Projeler

### [01-kmeans](./01-kmeans)
`K-Means` · `PCA`

- **ecommerce-customer-segmentation** — Müşterileri satın alma davranışlarına göre (ortalama miktar, birim fiyat, sipariş sıklığı) K-Means ile segmentlere ayıran, istatistiksel optimum ile iş kararının çakıştığı gerçek bir senaryoyu (K=2 yerine bilinçli K=3 seçimi) gösteren bir müşteri segmentasyonu projesi. *(Silhouette 0.44)*

### [02-clustering-comparison](./02-clustering-comparison)
`K-Means` · `Hierarchical Clustering` · `DBSCAN` · `Gaussian Mixture Model` · `PCA`

- **insurance-market-segmentation** — Sigorta müşterilerini **4 farklı kümeleme algoritmasıyla** segmentlere ayırıp sonuçları Silhouette skoruyla kıyaslayan, "hangi algoritma bu veri için en uygun" sorusuna veriyle cevap arayan bir kıyaslama çalışması. *(En iyi: K-Means, Silhouette 0.38)*

### [03-isolation-forest](./03-isolation-forest)
`Isolation Forest`

- **isolation-forest-fraud** — Kredi kartı işlemlerinde, hiçbir etiket kullanmadan (tüm veriyle, etiketsiz eğitim) şüpheli işlemleri tespit eden bir dolandırıcılık anomali tespiti projesi. Etiketli modellerin (Random Forest) yakalayamayacağı, daha önce görülmemiş fraud örüntülerini yakalama potansiyeli gösterir. *(ROC-AUC 0.9504, PR-AUC 0.6735)*

### [04-one-class-svm](./04-one-class-svm)
`One-Class SVM`

- **ocsvm-intrusion-detection** — Ağ bağlantı trafiğinde, **sadece normal trafik örnekleriyle eğitilerek** saldırı/intrüzyon tespiti yapan bir ağ güvenliği projesi. Isolation Forest'tan farklı olarak "önce normalin sınırını öğren, dışına çıkanı işaretle" mantığıyla çalışır. *(ROC-AUC 0.9626, Recall %90)*

---

## 🗺️ Kapsam

```
01-kmeans                 →  Merkezi kümeleme, iş bağlamlı K seçimi
02-clustering-comparison  →  4 algoritma kıyaslaması, yöntem seçimi metodolojisi
03-isolation-forest       →  Anomali tespiti, tüm veriyle etiketsiz eğitim
04-one-class-svm          →  Anomali tespiti, sadece normal veriyle eğitim
```

`03` ve `04`, aynı problem sınıfına (anomali tespiti) **iki farklı metodolojik yaklaşımla** çözüm sunar — README'lerinde bu iki yaklaşımın ne zaman hangisinin tercih edileceği karşılaştırmalı olarak ele alınmıştır.

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

---

<p align="center"><i>Kümeleme, boyut indirgeme ve anomali tespiti pratiği amaçlı bir portföy klasörüdür.</i></p>

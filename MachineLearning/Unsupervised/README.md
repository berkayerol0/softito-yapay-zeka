# 🧩 Unsupervised Öğrenme

Etiketlenmemiş veride yapı/grup keşfeden **denetimsiz öğrenme** projelerinin bulunduğu klasör. Kümeleme (clustering) ve boyut indirgeme (dimensionality reduction) tekniklerini kapsar.

---

## 📁 Projeler

### [ecommerce-customer-segmentation](./ecommerce-customer-segmentation)
Müşterileri satın alma davranışlarına göre (ortalama miktar, birim fiyat, sipariş sıklığı) K-Means ile segmentlere ayıran, istatistiksel optimum ile iş kararının çakışmadığı gerçek bir senaryoyu (K=2 yerine bilinçli K=3 seçimi) gösteren bir müşteri segmentasyonu projesi.

`K-Means` · `PCA` · Silhouette 0.44 (K=3)

### [insurance-market-segmentation](./insurance-market-segmentation)
Sigorta müşterilerini **4 farklı kümeleme algoritmasıyla** (K-Means, Hierarchical, DBSCAN, Gaussian Mixture Model) segmentlere ayırıp sonuçları Silhouette skoruyla kıyaslayan, "hangi algoritma bu veri için en uygun" sorusuna veriyle cevap arayan bir kıyaslama çalışması.

`K-Means` · `Hierarchical Clustering` · `DBSCAN` · `GMM` · `PCA` · En iyi: K-Means (Silhouette 0.38)

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

---

<p align="center"><i>Kümeleme ve denetimsiz öğrenme metodolojisi pratiği amaçlı bir portföy klasörüdür.</i></p>

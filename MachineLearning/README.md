# 🤖 Machine Learning

Makine öğrenmesi algoritmalarını **denetimli (Supervised)** ve **denetimsiz (Unsupervised)** öğrenme olmak üzere iki ana kategoride, her biri sektörde gerçekten kullanılan bir problem üzerinden uygulayan proje koleksiyonu.

Her proje gerçek çalışan kod, gerçek üretilmiş sonuç ve görsellerle birlikte, **"Projenin Amacı"** ve **iş/sektör bağlamı** açıklamalarını içeren bir README ile belgelenmiştir.

---

## 📊 Supervised (Denetimli Öğrenme)

Etiketli veriyle eğitilen, "doğru cevabı" bilerek öğrenen algoritmalar.

| # | Klasör | Proje(ler) | Algoritma | Öne Çıkan Sonuç |
|---|---|---|---|---|
| 01 | [linear-regresyon](./Supervised/01-linear-regresyon) | pay-equity-analysis · superlig-goal-prediction | Linear Regression | R² 0.67 (ücret) · R² 0.83 (gol) |
| 02 | [logistic-regresyon](./Supervised/02-logistic-regresyon) | credit-scoring · churn-prediction | Logistic Regression | Accuracy ~%78 · ~%75 |
| 03 | [decision-tree](./Supervised/03-decision-tree) | decision_tree_clinical · mobile-price-decision-tree | Decision Tree | Accuracy ~%54 (3 sınıf) · ~%74 (4 sınıf) |
| 04 | [random-forest](./Supervised/04-random-forest) | fraud-detection-rf | Random Forest | ROC-AUC 0.81 · PR-AUC 0.53 |
| 05 | [lightgbm](./Supervised/05-lightgbm) | bank-campaign-lightgbm | LightGBM | Accuracy %91.20 · ROC-AUC 0.891 |
| 06 | [svm](./Supervised/06-svm) | svm-tumor-diagnosis | SVM (Linear/RBF) | Accuracy %97.37 · ROC-AUC 0.997 |
| 07 | [knn](./Supervised/07-knn) | knn-product-recommender | KNN | Precision@5 %92.0 · MAE 1.02 |
| 08 | [naive-bayes](./Supervised/08-naive-bayes) | naive-bayes-sentiment | Naive Bayes + TF-IDF | Accuracy %91.19 (3 sınıf) |
| — | [Model Karşılaştırma](./Supervised/Model%20Karşılaştırma) | logreg-vs-randomforest-diabetes · ctr-xgboost-vs-lightgbm | Çoklu algoritma kıyaslaması | ROC-AUC 0.74 vs 0.73 · 0.927 vs 0.926 |

---

## 🧩 Unsupervised (Denetimsiz Öğrenme)

Etiketlenmemiş veride yapı, grup veya sapma keşfeden algoritmalar.

| # | Klasör | Proje(ler) | Algoritma | Öne Çıkan Sonuç |
|---|---|---|---|---|
| 01 | [kmeans](./Unsupervised/01-kmeans) | ecommerce-customer-segmentation | K-Means + PCA | Silhouette 0.44 |
| 02 | [clustering-comparison](./Unsupervised/02-clustering-comparison) | insurance-market-segmentation | K-Means · Hierarchical · DBSCAN · GMM | En iyi: K-Means (0.38) |
| 03 | [isolation-forest](./Unsupervised/03-isolation-forest) | isolation-forest-fraud | Isolation Forest | ROC-AUC 0.9504 (etiketsiz) |
| 04 | [one-class-svm](./Unsupervised/04-one-class-svm) | ocsvm-intrusion-detection | One-Class SVM | ROC-AUC 0.9626 · Recall %90 |

---

## 🗺️ Genel Bakış

```
Machine Learning
├── Supervised (8 algoritma + 1 kıyaslama klasörü, 11 proje)
│   ├── Regresyon: Linear, Logistic
│   ├── Ağaç tabanlı: Decision Tree, Random Forest, LightGBM
│   ├── Diğer: SVM, KNN, Naive Bayes
│   └── Model Karşılaştırma: LogReg vs RF, XGBoost vs LightGBM
│
└── Unsupervised (4 algoritma, 4 proje)
    ├── Kümeleme: K-Means, (Hierarchical + DBSCAN + GMM kıyaslaması)
    └── Anomali Tespiti: Isolation Forest, One-Class SVM
```

**Toplam: 12 algoritma/kategori, 15 proje.**

---

## ⚠️ Veri Hakkında Genel Not

Bu koleksiyondaki projelerin büyük çoğunluğu, gizlilik/erişim kısıtları nedeniyle **gerçekçi sentetik veri** ile çalışır — gerçek veri seti kullanılmadığında bu her projenin kendi README'sinde açıkça belirtilir. Kod her zaman gerçekten çalıştırılmış, sonuçlar ve görseller gerçek çıktılardır; sadece ham veri simüle edilmiştir.

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

<p align="center"><i>Denetimli ve denetimsiz öğrenme algoritmalarının sektörel uygulama pratiği amaçlı bir portföy klasörüdür.</i></p>

# 🎯 Supervised (Denetimli Öğrenme)

Etiketli veriyle eğitilen, "doğru cevabı" bilerek öğrenen denetimli öğrenme algoritmalarının bulunduğu klasör. Her alt klasör bir algoritmaya odaklanır ve o algoritmanın sektörde gerçekten kullanıldığı bir problem üzerinden uygulanır.

---

## 📁 Projeler

### [01-linear-regresyon](./01-linear-regresyon)
`Linear Regression`

- **pay-equity-analysis** — Bir şirkette cinsiyete dayalı açıklanamayan ücret farkını tespit eden bir ücret şeffaflığı analizi. *(R² 0.67)*
- **superlig-goal-prediction** — Futbol takımı istatistiklerinden gol sayısı tahmini, basit ve çoklu regresyon kıyaslaması. *(R² 0.83, çoklu regresyon)*

### [02-logistic-regresyon](./02-logistic-regresyon)
`Logistic Regression`

- **credit-scoring** — Kredi başvurusu onay/red tahmini, açıklanabilir katsayılarla. *(Accuracy ~%78)*
- **churn-prediction** — SaaS/telekom müşteri kaybı (churn) tahmini ve erken uyarı sistemi. *(Accuracy ~%75)*

### [03-decision-tree](./03-decision-tree)
`Decision Tree`

- **decision_tree_clinical** — Hasta risk seviyesi tahmini, sınırlı derinlikte yorumlanabilir karar ağacı. *(Accuracy ~%54, 3 sınıf)*
- **mobile-price-decision-tree** — Mobil cihaz donanım özelliklerinden fiyat segmenti tahmini. *(Accuracy ~%74, 4 sınıf)*

### [04-random-forest](./04-random-forest)
`Random Forest`

- **fraud-detection-rf** — Kredi kartı işlemi dolandırıcılık tespiti, dengesiz sınıflarla çalışma ve PR-AUC odaklı değerlendirme. *(ROC-AUC 0.81, PR-AUC 0.53)*

### [05-lightgbm](./05-lightgbm)
`LightGBM`

- **bank-campaign-lightgbm** — Banka telefon kampanyasına katılım tahmini, early stopping ve öğrenme eğrisi analiziyle rekabetçi/yüksek doğruluk odaklı bir uygulama. *(Accuracy %91.20, ROC-AUC 0.891)*

### [06-svm](./06-svm)
`SVM`

- **svm-tumor-diagnosis** — Hücre ölçümlerinden tümör teşhisi (iyi huylu/kötü huylu), Linear ve RBF kernel kıyaslaması. *(Accuracy %97.37, ROC-AUC 0.997)*

### [07-knn](./07-knn)
`KNN`

- **knn-product-recommender** — Item-based collaborative filtering ile ürün öneri sistemi. *(Precision@5 %92.0, MAE 1.02)*

### [08-naive-bayes](./08-naive-bayes)
`Naive Bayes` · `TF-IDF`

- **naive-bayes-sentiment** — Ürün yorumlarında 3 sınıflı (Pozitif/Nötr/Negatif) duygu analizi. *(Accuracy %91.19)*

---

## 🗺️ Kapsam

```
01-linear-regresyon   →  Sürekli değer tahmini, açıklanabilir katsayılar
02-logistic-regresyon →  İkili sınıflandırma, açıklanabilirlik
03-decision-tree      →  Kural bazlı, görselleştirilebilir karar yapısı
04-random-forest      →  Ensemble, dengesiz sınıflarla çalışma
05-lightgbm           →  Gradient boosting, rekabetçi/yüksek doğruluk
06-svm                →  Kernel yöntemleri, yüksek boyutlu ayrım
07-knn                →  Örnek tabanlı öğrenme, öneri sistemleri
08-naive-bayes        →  Olasılıksal sınıflandırma, metin/NLP
```

---

## 🔗 İlgili Kategori

Bu klasördeki algoritmaların **birbirleriyle doğrudan kıyaslandığı** projeler ayrı bir kategoride tutulur:

### [Model Karşılaştırma](../Model%20Karşılaştırma)
- **logreg-vs-randomforest-diabetes** — Logistic Regression vs Random Forest, Cross-Validation ile model seçimi metodolojisi.
- **ctr-xgboost-vs-lightgbm** — XGBoost vs LightGBM, tıklanma tahmini üzerinde hız/doğruluk kıyaslaması.

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `pandas` · `matplotlib` · `seaborn`

---

<p align="center"><i>Denetimli öğrenme algoritmaları ve sektörel uygulama pratiği amaçlı bir portföy klasörüdür.</i></p>

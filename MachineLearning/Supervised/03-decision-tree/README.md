# 🌳 Decision Tree

Karar ağaçlarıyla kurulmuş, **açık ve takip edilebilir kurallara** dökülebilen sınıflandırma projelerinin bulunduğu klasör.

Decision Tree, Random Forest veya XGBoost kadar yüksek doğruluk vermeyebilir, ama modelin kararını **if-else mantığında birebir açıklayabilmesi** nedeniyle sağlık, finans gibi yorumlanabilirliğin kritik olduğu alanlarda hâlâ tercih edilir.

---

## 📁 Projeler

### [decision_tree_clinical](./decision_tree_clinical)
Bir hastanın tükenmişlik/risk seviyesini (Düşük/Orta/Yüksek) tahmin eden ve bu tahmini bir hekimin okuyup onaylayabileceği kadar basit, sınırlı derinlikte bir karar ağacına döken klinik karar destek prototipi.

`Decision Tree` · `scikit-learn` · Accuracy ~%54 (3 sınıf, yorumlanabilirlik için bilinçli sınırlandırılmış derinlik)

### [mobile-price-decision-tree](./mobile-price-decision-tree)
Mobil cihazların donanım özelliklerine (RAM, batarya, kamera, çözünürlük vb.) göre fiyat segmentini (Düşük/Orta/Yüksek/Çok Yüksek) tahmin eden, derinlik-doğruluk analiziyle overfitting'i de inceleyen bir sınıflandırma projesi.

`Decision Tree` · `scikit-learn` · Accuracy ~%74

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

---

<p align="center"><i>Yorumlanabilir (interpretable) makine öğrenmesi pratiği amaçlı bir portföy klasörüdür.</i></p>

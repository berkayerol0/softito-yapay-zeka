# 🎯 Logistic Regresyon

Sınıflandırma problemlerinde **açıklanabilirliğin kritik olduğu** sektörlerden (fintech, müşteri analitiği) seçilmiş, Logistic Regression ile kurulmuş projelerin bulunduğu klasör.

Logistic Regression, XGBoost veya derin öğrenme kadar "güçlü" görünmese de, model kararının **katsayılar üzerinden birebir açıklanabilmesi** nedeniyle regülasyona tabi veya yüksek sorumluluk gerektiren alanlarda hâlâ endüstri standardıdır.

---

## 📁 Projeler

### [credit-scoring](./credit-scoring)
Bir kredi başvurusunun onaylanıp onaylanmayacağını tahmin eden ve **modelin kararını neden verdiğini** katsayılar üzerinden açıklayan bir kredi skorlama sistemi. Fintech/bankacılık sektöründeki düzenleyici zorunluluklar (açıklanabilir model şartı) göz önünde bulundurularak tasarlandı.

`Logistic Regression` · `scikit-learn` · Accuracy ~%78, ROC-AUC ~0.86

### [churn-prediction](./churn-prediction)
Bir SaaS/telekom müşterisinin hizmeti bırakma (churn) olasılığını tahmin eden ve hangi faktörlerin (sözleşme tipi, destek talebi sayısı, kullanım yoğunluğu vb.) riski artırdığını gösteren bir erken uyarı sistemi.

`Logistic Regression` · `scikit-learn` · Accuracy ~%75, ROC-AUC ~0.80

---

## 🛠️ Ortak Teknolojiler

`Python` · `scikit-learn` · `pandas` · `matplotlib` · `seaborn`

---

<p align="center"><i>Açıklanabilir (interpretable) makine öğrenmesi pratiği amaçlı bir portföy klasörüdür.</i></p>

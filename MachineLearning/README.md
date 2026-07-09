# Machine Learning: Klasik Algoritmalardan Rekabetçi Modellere

## Proje Haritası — Supervised (Denetimli Öğrenme)

| # | Klasör | Yöntem | Bu Projede Ne Yaptık |
|---|---|---|---|
| 01 | [linear-regresyon](./Supervised/01-linear-regresyon) | Linear Regression | Bir şirkette cinsiyete dayalı **açıklanamayan ücret farkını** tespit ettik (ham fark %6.6 → kontrol edilmiş fark %6.1, R² 0.67); ayrı bir projede futbol takımı istatistiklerinden gol sayısı tahmin edip basit regresyonun (R² 0.75) çoklu regresyona (R² 0.83) göre ne kadar zayıf kaldığını gösterdik |
| 02 | [logistic-regresyon](./Supervised/02-logistic-regresyon) | Logistic Regression | Kredi başvurusu onay/red kararını katsayı bazında açıklanabilir hale getirdik (%78 doğruluk); SaaS müşteri kaybını (churn) tahmin edip hangi faktörün riski en çok artırdığını çıkardık (%75 doğruluk) |
| 03 | [decision-tree](./Supervised/03-decision-tree) | Decision Tree | Hasta risk seviyesini, bir hekimin okuyup onaylayabileceği kadar sınırlı derinlikte bir karar ağacına döktük (%54, 3 sınıf); mobil cihaz donanımından fiyat segmenti tahmin edip derinlik arttıkça doğruluğun nerede platoya girdiğini ölçtük (%74, 4 sınıf) |
| 04 | [random-forest](./Supervised/04-random-forest) | Random Forest | Kredi kartı işlemlerinde dolandırıcılık tespiti yaptık; sınıflar dengesiz olduğu için accuracy yerine neden PR-AUC'a bakılması gerektiğini gösterdik (ROC-AUC 0.81, PR-AUC 0.53) |
| 05 | [lightgbm](./Supervised/05-lightgbm) | LightGBM | Banka telefon kampanyasına katılım tahmini yaptık; early stopping ile 1000 ağaç üst sınırından modelin kendi kendine en iyi noktayı (281. iterasyon) bulmasını sağladık (%91.2 doğruluk, ROC-AUC 0.891) |
| 06 | [svm](./Supervised/06-svm) | SVM (Linear + RBF) | Hücre ölçümlerinden tümör teşhisi yaptık (%97.4 doğruluk); Linear ve RBF kernel'i kıyaslayıp verinin doğrusal ayrılabilir olduğunu, dolayısıyla daha basit kernel'in yeterli olduğunu kanıtladık |
| 07 | [knn](./Supervised/07-knn) | KNN | Kullanıcı davranışından (puanlama verisinden) yola çıkan bir ürün öneri sistemi kurduk; model ürün kategorisini hiç görmeden, en benzer bulduğu ürünlerin %92'sinin gerçekten aynı kategoriden çıktığını gösterdik |
| 08 | [naive-bayes](./Supervised/08-naive-bayes) | Naive Bayes + TF-IDF | Ürün yorumlarını 3 sınıfa (Pozitif/Nötr/Negatif) ayırdık (%91.2 doğruluk); modelin hangi kelimelere bakarak karar verdiğini çıkarıp Nötr sınıfın neden her zaman en zor sınıf olduğunu gösterdik |
| — | [Model Karşılaştırma](./Supervised/Model%20Karşılaştırma) | Çoklu Algoritma Kıyaslaması | Logistic Regression'ı Random Forest ile, XGBoost'u LightGBM ile aynı veride yan yana çalıştırıp hangisinin ne zaman öne çıktığını (ve bazen "beklenen" sonucun çıkmadığını) 5-Fold Cross-Validation ile ölçtük |

## Proje Haritası — Unsupervised (Denetimsiz Öğrenme)

| # | Klasör | Yöntem | Bu Projede Ne Yaptık |
|---|---|---|---|
| 01 | [kmeans](./Unsupervised/01-kmeans) | K-Means + PCA | Müşterileri satın alma davranışına göre segmentledik; istatistiksel olarak en iyi K değeri (K=2) yerine iş açısından daha kullanışlı K=3'ü bilinçli seçip bu kararı gerekçelendirdik |
| 02 | [clustering-comparison](./Unsupervised/02-clustering-comparison) | K-Means, Hierarchical, DBSCAN, GMM | Sigorta müşterilerini **4 farklı kümeleme algoritmasıyla** ayrı ayrı segmentledik, Silhouette skoruyla kıyasladık; DBSCAN'ın neden farklı sayıda küme bulduğunu ve bunun bir hata değil algoritmanın doğası olduğunu gösterdik |
| 03 | [isolation-forest](./Unsupervised/03-isolation-forest) | Isolation Forest | Kredi kartı işlemlerinde **hiç etiket kullanmadan** dolandırıcılık tespiti yaptık (ROC-AUC 0.95); etiketli modellerin yakalayamayacağı yeni/görülmemiş fraud örüntülerini bu yaklaşımın neden yakalayabileceğini açıkladık |
| 04 | [one-class-svm](./Unsupervised/04-one-class-svm) | One-Class SVM | Ağ trafiğinde **sadece normal örneklerle eğitilen** bir modelle saldırı tespiti yaptık (ROC-AUC 0.96, Recall %90); Isolation Forest'tan farklı olarak "önce normalin sınırını öğren" yaklaşımının ne zaman tercih edilmesi gerektiğini kıyasladık |

---

Makine öğrenmesinin temel taşlarını, en basit doğrusal modelden dengesiz veri/anomali tespitine kadar artan bir karmaşıklıkla ele aldık. Her proje gerçek çalışan kod, gerçek üretilmiş sonuç ve görsellerle belgelenmiştir; veri setleri gizlilik/erişim kısıtları nedeniyle gerçekçi sentetik verilerdir — bu her projenin kendi README'sinde ayrıca belirtilir.

## 🛠️ Kullanılan Teknolojiler

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `pandas` · `numpy` · `matplotlib` · `seaborn`

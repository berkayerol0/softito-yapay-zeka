# Deep Learning

Farklı veri türlerinde (görüntü ve zaman serisi) derin öğrenme tekniklerini gösteren
projelerden oluşur. Her alt klasör bağımsız çalıştırılabilir, kendi `README.md`'si,
`figures/` klasörü ve `requirements.txt`'i vardır.

## Genel Bakış

| Alan | Klasör | Veri Türü | Kapsadığı Teknikler |
|---|---|---|---|
| Görüntü (Vision) | [`CNN/`](CNN/) | Görüntü (statik) | Klasik görüntü işleme (OpenCV), özellik mühendisliği (PCA/augmentation/t-SNE), uçtan uca CNN sınıflandırma |
| Zaman Serisi (Sequence) | [`gru-zaman-serisi/`](gru-zaman-serisi/) | Sayısal zaman serisi | GRU ile tahmin, kronolojik train/test ayrımı, RMSE/MAE değerlendirme |

Bu iki alan kasıtlı olarak birbirinden çok farklı veri türlerini kapsıyor — biri mekansal
(bir görüntüdeki komşu pikseller arasındaki ilişki), diğeri zamansal (bir serideki ardışık
adımlar arasındaki ilişki) yapıyı öğrenmeyi gerektiriyor. Bu, aynı temel derin öğrenme
prensiplerinin (kayıp fonksiyonu, geri yayılım, aşırı öğrenmeyi önleme) farklı mimarilerle
(CNN vs. RNN/GRU) nasıl farklı problem türlerine uyarlandığını gösteriyor.

## CNN — Görüntü İşleme ve Sınıflandırma

3 projeden oluşur, klasikten derin öğrenmeye bir yolculuk izler:

1. **[03-opencv](CNN/03-opencv/)** — RGB/HSV/LAB renk uzayları, filtreleme, kenar tespiti,
   morfolojik işlemler, kontur analizi (ALPR/plaka tanıma senaryosu üzerinden)
2. **[02-goruntu-on-isleme](CNN/02-goruntu-on-isleme/)** — Veri artırma (augmentation),
   PCA ile boyut indirgeme, t-SNE ile özellik uzayı görselleştirme
3. **[01-fashion-mnist-cnn](CNN/01-fashion-mnist-cnn/)** — Fashion-MNIST üzerinde uçtan
   uca CNN sınıflandırma (%90.46 test doğruluğu)

**Öne çıkan görsel — CNN'in en çok zorlandığı sınıflar:**

![Confusion Matrix](CNN/01-fashion-mnist-cnn/figures/confusion_matrix.png)

Detaylı özet ve tüm projelerin bulguları için [`CNN/README.md`](CNN/README.md).

## GRU — Zaman Serisi Tahmini

**[gru-zaman-serisi](gru-zaman-serisi/)** — Sentetik bir ürün talebi serisinde (trend +
haftalık mevsimsellik + gürültü), geçmiş 14 güne bakarak bir sonraki günü tahmin eden bir
GRU modeli. Kronolojik train/test ayrımıyla veri sızıntısı önlendi.

**Sonuç:** Test setinde RMSE 2.44, MAE 1.93 — model haftalık mevsimsellik desenini büyük
ölçüde yakaladı, ani tepe noktalarında hafif bir gecikme davranışı gözlemlendi:

![Gerçek vs Tahmin](gru-zaman-serisi/figures/gercek_vs_tahmin.png)

## Ortak Prensipler

- Her proje **somut bir çıktı** (grafik, CSV, confusion matrix, metrik tablosu) üretir;
  yalnızca kod değil, gözle görülür ve yorumlanmış bir sonuç sunar.
- Bulgular dürüstçe raporlanır — augmentation'ın bazen performansı düşürmesi (CNN) veya
  GRU'nun ani değişimlerde gecikmesi (zaman serisi) gibi "beklenmedik" sonuçlar
  gizlenmez, aksine neden oluştuğu açıklanır.
- Her teknik, gerçek sektör kullanım alanlarıyla ilişkilendirilir (örn. HSV renk
  uzayının otonom sürüşte kullanımı, GRU'nun talep tahmininde kullanımı).

## Çalıştırma

```bash
cd <klasör-adı>/<alt-klasör-adı>
pip install -r requirements.txt
python <script-adı>.py
```

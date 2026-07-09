# -*- coding: utf-8 -*-
"""
GORUNTU ON ISLEME - Veri Artirma (Augmentation), PCA ve t-SNE
Amac: Bir goruntu siniflandirma veri setinde CNN'e gecmeden once klasik
      ozellik muhendisligi adimlarini (augmentation, PCA, t-SNE) uygulamak
      ve bunlarin model basarisina etkisini gostermek.

Not: Orijinal not defteri Kaggle uzerinden 'fresh-and-rotten-fruits-and-
     vegetables' veri setini indiriyordu; bu Kaggle API credential'i
     (kaggle.json) gerektirdigi icin, ayni veri yapisini (X_custom,
     y_custom, class_names) ureten SENTETIK bir goruntu veri seti
     kullanilir: 4 sinif (Daire, Kare, Ucgen, Yildiz), her biri gurultu
     ve rastgele konum/boyut varyasyonuyla cizilmis basit sekil
     goruntuleri. Asagidaki tum islem hatti (augmentation, MLP, PCA,
     t-SNE) DEGISTIRILMEDEN orijinal mantikla calisir.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

IMG_SIZE = 64
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Sentetik sekil veri seti uretiliyor (Kaggle yerine)...")

class_names = ["Daire", "Kare", "Ucgen", "Yildiz"]
class_to_idx = {name: i for i, name in enumerate(class_names)}
n_per_class = 150


def draw_shape(shape_name, size=64):
    """Tek bir sentetik sekil goruntusu (gri tonlama) cizer."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2 + np.random.randint(-6, 6), size // 2 + np.random.randint(-6, 6)
    r = np.random.randint(size // 4, size // 2 - 4)
    thickness = np.random.choice([-1, 2, 3])  # -1: dolu, aksi halde cizgi kalinligi
    color = int(np.random.randint(180, 256))

    if shape_name == "Daire":
        cv2.circle(img, (cx, cy), r, color, thickness)
    elif shape_name == "Kare":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness)
    elif shape_name == "Ucgen":
        pts = np.array([
            [cx, cy - r],
            [cx - r, cy + r],
            [cx + r, cy + r],
        ], dtype=np.int32)
        if thickness == -1:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness)
    elif shape_name == "Yildiz":
        pts = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            radius = r if i % 2 == 0 else r * 0.45
            pts.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])
        pts = np.array(pts, dtype=np.int32)
        if thickness == -1:
            cv2.fillPoly(img, [pts], color)
        else:
            cv2.polylines(img, [pts], True, color, thickness)

    # Gercekci hale getirmek icin hafif gauss gurultusu ekle
    noise = np.random.normal(0, 12, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


X_custom = []
y_custom = []
for class_name in class_names:
    label = class_to_idx[class_name]
    for _ in range(n_per_class):
        img = draw_shape(class_name, IMG_SIZE)
        img_flat = img.flatten() / 255.0
        X_custom.append(img_flat)
        y_custom.append(label)

X_custom = np.array(X_custom)
y_custom = np.array(y_custom)

print(f"Uretilen X_custom boyutu: {X_custom.shape}")
print(f"Uretilen y_custom boyutu: {y_custom.shape}")
print(f"Siniflar: {class_names}")

# Yuklenen goruntulerden birini gorsellestir
sample_index = np.random.randint(0, len(X_custom))
plt.figure(figsize=(3, 3))
plt.imshow(X_custom[sample_index].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
plt.title(f"Ornek Goruntu (Etiket: {class_names[y_custom[sample_index]]})")
plt.axis('off')
plt.savefig('figures/00_ornek_goruntu.png', dpi=150, bbox_inches='tight')
plt.close()

"""### Özel Görsel Verileriyle Eğitim/Test Bölmesi

Artık kendi görsel verileriniz `X_custom` ve etiketleriniz `y_custom` NumPy dizileri olarak hazır olduğuna göre, mevcut `digits` veri setinde yaptığınız gibi, bunları eğitim ve test setlerine ayırabilirsiniz. Ayrıca, `StandardScaler` kullanarak verileri standartlaştıracağız.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np # X_custom np.array ise bu gerekli

# Özel görsel veri setini eğitim ve test setlerine ayır
# stratify=y_custom, her iki kümede de sınıf oranlarının korunmasını sağlar
if len(X_custom) > 0:
    X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
        X_custom, y_custom, test_size=0.25, random_state=42, stratify=y_custom
    )

    # StandardScaler kullanarak özel veriyi standartlaştırın
    # StandardScaler'ı yalnızca eğitim verisine fit ediyoruz ve test verisine sadece transform uyguluyoruz
    scaler_custom = StandardScaler()
    X_train_sc_custom = scaler_custom.fit_transform(X_train_custom)
    X_test_sc_custom  = scaler_custom.transform(X_test_custom)

    print(f"\nÖzel veri için eğitim seti boyutu: {X_train_sc_custom.shape}")
    print(f"Özel veri için test seti boyutu: {X_test_sc_custom.shape}")
    print("✅ Kendi görsel verileriniz başarıyla yüklendi ve hazırlandı!")

    # Artık bu verileri kullanarak model eğitiminize veya diğer analizlerinize devam edebilirsiniz.
else:
    print("Hata: X_custom boş olduğu için eğitim/test bölmesi yapılamadı. Veri yükleme adımlarını kontrol edin.")
    # Boş değişkenler tanımla veya hata işlemesini yap
    X_train_custom = np.array([])
    X_test_custom = np.array([])
    y_train_custom = np.array([])
    y_test_custom = np.array([])
    X_train_sc_custom = np.array([])
    X_test_sc_custom = np.array([])

"""## Veri Artırma (Data Augmentation)

**Problem:** Gerçek projelerde etiketli veri az olur. Modeli overfit'ten korumak için
mevcut görüntüleri **dönüştürerek** yapay yeni örnekler üretilir.

**Neden işe yarar?** Model aynı nesnenin farklı açı/parlaklık/gürültü versiyonlarını görünce
gerçek dünya varyasyonlarını öğrenir.

```
Orijinal → Döndür → Kaydır → Ölçekle → Gürültü ekle → Ayna → Parlaklık değiştir
```

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def augment_image(img_flat_1d, image_size, seed=None):
    # Tek bir düzleştirilmiş (flattened) görüntüye rastgele augmentation uygula.
    # Döndür: sözlük — her dönüşümün sonucu ayrı ayrı

    rng = np.random.default_rng(seed)          # Tekrar üretilebilir rastgelelik

    # Düzleştirilmiş görüntüyü orijinal boyuta (image_size x image_size) geri getir
    img_2d = (img_flat_1d * 255.0).astype(np.uint8).reshape(image_size, image_size)

    results = {}
    results['original'] = img_flat_1d.copy()        # Orijinali sakla (düzleştirilmiş halde)

    h, w  = img_2d.shape

    # ── 1. Döndürme (Rotation) ──────────────────────────────────────────────
    angle = rng.uniform(-25, 25)               # -25° ile +25° arası rastgele açı
    M     = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)   # Dönme matrisi
    rotated = cv2.warpAffine(img_2d, M, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)   # Kenar boşluğunu doldur
    results['rotation'] = (rotated.astype(float) / 255.0).flatten()      # Geri normalize et ve düzleştir

    # ── 2. Kaydırma (Translation) ──────────────────────────────────────────
    tx = rng.uniform(-image_size*0.1, image_size*0.1) # Görüntü boyutuna göre kaydırma
    ty = rng.uniform(-image_size*0.1, image_size*0.1)
    M_trans = np.float32([[1, 0, tx],
                           [0, 1, ty]])
    shifted = cv2.warpAffine(img_2d, M_trans, (w, h),
                              borderMode=cv2.BORDER_REPLICATE)
    results['translation'] = (shifted.astype(float) / 255.0).flatten()

    # ── 3. Yatay Ayna (Horizontal Flip) ────────────────────────────────────
    flipped = np.fliplr(img_2d)             # Sütunları tersine çevir
    results['flip'] = (flipped.astype(float) / 255.0).flatten()

    # ── 4. Gaussian Gürültü ─────────────────────────────────────────────────
    sigma  = rng.uniform(5, 20)               # Rastgele gürültü şiddeti
    noise  = rng.normal(0, sigma, img_2d.shape).astype(np.int16)
    noisy  = np.clip(img_2d.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    results['noise'] = (noisy.astype(float) / 255.0).flatten()

    # ── 5. Parlaklık Değiştirme (Brightness) ───────────────────────────────
    factor = rng.uniform(0.6, 1.6)            # 0.6× (karartma) ile 1.6× (aydınlatma)
    bright = np.clip(img_2d.astype(float) * factor, 0, 255).astype(np.uint8)
    results['brightness'] = (bright.astype(float) / 255.0).flatten()

    # ── 6. Zoom (Kırp + Yeniden Boyutlandır) ───────────────────────────────
    zoom   = rng.uniform(0.7, 0.95)           # Görüntünün %70-95'ini al
    crop_h = int(h * zoom)
    crop_w = int(w * zoom)
    y0     = rng.integers(0, h - crop_h + 1)  # Rastgele kırpma başlangıcı
    x0     = rng.integers(0, w - crop_w + 1)
    cropped  = img_2d[y0:y0+crop_h, x0:x0+crop_w]
    zoomed   = cv2.resize(cropped, (w, h))    # Orijinal boyuta geri döndür
    results['zoom'] = (zoomed.astype(float) / 255.0).flatten()

    return results

# ── İlk birkaç örnek üzerinde augmentation göster ────────────────────────────
# X_custom ve y_custom'dan bir örnek alalım
if len(X_custom) > 0 and len(class_names) > 0:
    sample_idx = np.random.randint(0, len(X_custom)) # Rastgele bir örnek seç
    sample_img_flat = X_custom[sample_idx] # Düzleştirilmiş görüntü
    sample_label = y_custom[sample_idx]
    sample_class_name = class_names[sample_label]

    aug_results = augment_image(sample_img_flat, IMG_SIZE, seed=42)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    titles = ['Orijinal', 'Döndürme\n(±25°)', 'Kaydırma\n(±10%)', # Kaydırma oranını güncelle
              'Yatay Ayna', 'Gaussian\nGürültü', 'Parlaklık\nDeğiştirme',
              'Zoom\n(Kırp+Yeniden Boyutlandır)', 'Çakışma\n(Hepsi Birden)']

    for i, (key, img_flat_res) in enumerate(aug_results.items()):
        # Görseli düzleştirilmiş formdan 2D'ye çevir
        img_2d_res = img_flat_res.reshape(IMG_SIZE, IMG_SIZE)
        axes[i].imshow(img_2d_res, cmap='gray_r', interpolation='nearest')
        axes[i].set_title(titles[i], fontsize=11, fontweight='bold')
        axes[i].set_xlabel(f"Shape: {img_2d_res.shape}\nMin:{img_2d_res.min():.2f} Max:{img_2d_res.max():.2f}", fontsize=8) # Min/Max değerleri 0-1 aralığında
        axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Piksel ızgarasını göster
        axes[i].set_xticks(np.arange(-0.5, IMG_SIZE, 1), minor=True)
        axes[i].set_yticks(np.arange(-0.5, IMG_SIZE, 1), minor=True)
        axes[i].grid(which='minor', color='gray', linewidth=0.3, alpha=0.5)

    # Son panel: tüm augmentasyonların yığınlanmış görüntüsü
    # Dikkat: `values()` artık düzleştirilmiş (flattened) diziler döndürüyor
    combined_flat = np.mean([v for v in aug_results.values()], axis=0)
    combined_2d = combined_flat.reshape(IMG_SIZE, IMG_SIZE)
    axes[7].imshow(combined_2d, cmap='gray_r', interpolation='nearest')
    axes[7].set_title(titles[7], fontsize=11, fontweight='bold')
    axes[7].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle(f"Veri Artırma (Data Augmentation) — Sınıf: {sample_class_name}", # Rakam yerine Sınıf
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/01_veri_artirma_augmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom veya class_names boş, örnek görüntü yüklenemedi.")

import numpy as np
from sklearn.neural_network import MLPClassifier # MLPClassifier'ı içe aktar
from sklearn.metrics import accuracy_score       # accuracy_score'u içe aktar
from sklearn.preprocessing import StandardScaler # StandardScaler'ı içe aktar

def build_augmented_dataset(X_flat_data, y_data, image_size, n_aug=1, seed=0):
    # Her görüntüye n_aug kez augmentation uygulayarak veri setini büyüt.
    # Döner: (X_aug, y_aug) — orijinaller dahil

    X_list, y_list = [X_flat_data], [y_data]  # Orijinaller
    rng = np.random.default_rng(seed)

    for aug_round in range(n_aug):
        X_new = []
        for i, img_flat in enumerate(X_flat_data):
            s = rng.integers(0, 10000)
            # augment_image, numpy, cv2 ve plt gerektirir. Bu fonksiyona dışarıdan erişilmeli
            aug = augment_image(img_flat, image_size, seed=int(s))
            # Her turda farklı bir augmentation tipi seç (original hariç 6 dönüşüm)
            key = list(aug.keys())[1 + (aug_round % (len(aug.keys()) - 1))] # 'original' hariç döngüsel seç
            X_new.append(aug[key])
        X_list.append(np.array(X_new))
        y_list.append(y_data)

    return np.vstack(X_list), np.concatenate(y_list)

# Sadece eğitim verisi üzerinde augmentation uygula (test'e dokunma!) ile çalışacak
print("Augmentation uygulanıyor...")

if len(X_train_custom) > 0 and len(y_train_custom) > 0:
    # Augmented dataset oluştur
    X_aug, y_aug = build_augmented_dataset(X_train_custom, y_train_custom, IMG_SIZE, n_aug=1) # n_aug=1 olarak değiştirildi
    print(f"Orijinal eğitim seti  : {len(y_train_custom)} örnek")
    print(f"Augmented eğitim seti : {len(y_aug)} örnek  ({(len(y_aug) / len(y_train_custom)):.0f}× büyüdü)")

    # Her iki veriyle MLP eğit ve karşılaştır
    # StandardScaler zaten uygulandı (X_train_sc_custom, X_test_sc_custom)

    # Yeni StandardScaler'ı sadece augmented veri için kullan
    sc_aug = StandardScaler()
    X_aug_sc = sc_aug.fit_transform(X_aug)

    mlp_orig = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42)
    mlp_aug  = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42)

    mlp_orig.fit(X_train_sc_custom,  y_train_custom)
    mlp_aug.fit(X_aug_sc, y_aug)

    acc_orig = accuracy_score(y_test_custom, mlp_orig.predict(X_test_sc_custom))
    acc_aug  = accuracy_score(y_test_custom, mlp_aug.predict(X_test_sc_custom))

    print(f"\nTest Doğruluğu — Orijinal veri : {acc_orig*100:.2f}%")
    print(f"Test Doğruluğu — Augmented veri: {acc_aug*100:.2f}%")
    print(f"Augmentation kazancı            : {(acc_aug-acc_orig)*100:+.2f} puan")
else:
    print("Hata: X_train_custom veya y_train_custom boş olduğu için augmentation ve model eğitimi yapılamadı.")

"""## PCA ile Boyut İndirgeme

**Problem:** 64 özellik → görselleştirme imkânsız.  
**PCA (Principal Component Analysis):** Veriyi en fazla varyansı koruyan eksenlere yansıtır.

**Matematiksel özü:** Kovaryans matrisinin öz vektörleri = yeni eksenler.  
Her öz vektör bir **"eigenface"** (öz görüntü) gibi düşünülebilir.

```
64 boyut  →PCA→  2-3 boyut   (görselleştirme için)
64 boyut  →PCA→  30-40 boyut (sınıflandırma için, bilgi kaybı az)
```

"""

from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib.pyplot as plt

# ── PCA ile Boyut İndirgeme ───────────────────────────────────────────────────
# Kendi veri setimiz olan X_custom'ı kullanıyoruz
if len(X_custom) > 0:
    pca_full = PCA()                               # Tüm bileşenleri hesapla (fit)
    pca_full.fit(X_custom)                           # X_custom'a fit et

    # Kümülatif açıklanan varyans — kaç bileşen yeterli?
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)   # Her bileşenin katkısını topla
    n90  = np.searchsorted(cumvar, 0.90) + 1       # %90 için gereken bileşen sayısı
    n95  = np.searchsorted(cumvar, 0.95) + 1       # %95 için
    n99  = np.searchsorted(cumvar, 0.99) + 1       # %99 için

    print(f"%90 varyans için gerekli bileşen: {n90}")
    print(f"%95 varyans için gerekli bileşen: {n95}")
    print(f"%99 varyans için gerekli bileşen: {n99}")
    print(f"Orijinal boyut: {X_custom.shape[1]}  →  Sıkıştırma oranı (%90): {X_custom.shape[1]/n90:.1f}×")

    # ── Eigendigits (PCA bileşenlerini görselleştir) ──────────────────────────────
    # Her öz bileşen, veri setindeki bir 'temel şekil'i temsil eder
    # IMG_SIZE'a göre subplot boyutunu ayarlayalım
    num_components_to_show = min(len(pca_full.components_), 21) # Gösterilecek max bileşen sayısı
    num_cols = 7
    num_rows = int(np.ceil(num_components_to_show / num_cols))

    fig = plt.figure(figsize=(18, num_rows * 3 + 2)) # figsize'ı dinamik yapalım
    gs = GridSpec(num_rows + 1, num_cols, height_ratios=[0.5] + [1]*num_rows)

    # İlk satır: varyans grafiği
    ax_var = fig.add_subplot(gs[0, :])
    ax_var.bar(range(1, num_components_to_show + 1), pca_full.explained_variance_ratio_[:num_components_to_show]*100,
               color='steelblue', alpha=0.7, label='Tekil bileşen')
    ax_var.plot(range(1, num_components_to_show + 1), cumvar[:num_components_to_show]*100,
                'ro-', markersize=4, linewidth=1.5, label='Kümülatif')
    ax_var.axhline(y=90, color='orange', linestyle='--', linewidth=1, label='%90')
    ax_var.axhline(y=95, color='red',    linestyle='--', linewidth=1, label='%95')
    ax_var.set_xlabel("Bileşen Numarası")
    ax_var.set_ylabel("Açıklanan Varyans (%)")
    ax_var.set_title("PCA — Her Bileşenin Katkısı", fontweight='bold')
    ax_var.legend(loc='right', fontsize=8)
    ax_var.set_xlim(0.5, num_components_to_show + 0.5)

    # Eigendigit görüntüleri
    for i in range(num_components_to_show):
        ax = fig.add_subplot(gs[i // num_cols + 1, i % num_cols]) # Dinamik subplot yerleşimi
        eigenimg = pca_full.components_[i].reshape(IMG_SIZE, IMG_SIZE)   # Düzleştirilmiş boyutu IMG_SIZE x IMG_SIZE olarak yeniden şekillendir
        im = ax.imshow(eigenimg, cmap='RdBu_r',             # Kırmızı-Mavi: pozitif/negatif
                       vmin=-np.max(np.abs(eigenimg)), vmax=np.max(np.abs(eigenimg))) # Renk ölçeğini dinamik yap
        ax.set_title(f"PC {i+1}\n({pca_full.explained_variance_ratio_[i]*100:.1f}%)",
                     fontsize=8)
        ax.axis('off')

    # Colorbar
    fig.colorbar(im, ax=ax_var, orientation='horizontal', fraction=0.02, pad=0.04,
                 label='Bileşen ağırlığı (negatif ↔ pozitif)') # Colorbar'ı varyans grafiğine ata

    plt.suptitle("Eigendigits — PCA Temel Bileşenleri\n(Her görüntü bir 'temel şekil' kalıbını temsil eder)",
                 fontsize=13, fontweight='bold', y=0.98) # suptitle y pozisyonunu ayarla
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # suptitle için yer aç
    plt.savefig('figures/02_pca_eigendigits.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom boş olduğu için PCA uygulanamadı. Veri yükleme adımlarını kontrol edin.")

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# ── PCA ile Görüntü Sıkıştırma ve Yeniden Yapılandırma ───────────────────────
# n bileşenle sıkıştır → orijinal boyuta geri döndür ve kaliteyi karşılaştır
if len(X_custom) > 0 and len(class_names) > 0:
    # X_custom veri setimizin orijinal boyutunu alalım
    original_dim = X_custom.shape[1]
    # Bileşen listesini X_custom'ın boyutuna göre ayarlayalım
    n_components_list = [2, 5, 10, 20, 30] # İlk bileşenleri deneyelim
    if original_dim > 64: # Eğer görsel boyutu 8x8'den büyükse daha fazla bileşen ekleyebiliriz
        n_components_list.extend([64, 128])
    n_components_list = [n for n in n_components_list if n <= original_dim] # original_dim'den büyük olmasın

    sample_idx = np.random.randint(0, len(X_custom)) # Rastgele bir örnek seçelim
    original_flat   = X_custom[sample_idx]              # Orijinal düzleştirilmiş görüntü
    original_label = y_custom[sample_idx]
    original_class_name = class_names[original_label]

    fig, axes = plt.subplots(1, len(n_components_list)+1, figsize=(3 * (len(n_components_list)+1), 3.5))

    axes[0].imshow(original_flat.reshape(IMG_SIZE, IMG_SIZE), cmap='gray_r', interpolation='nearest')
    axes[0].set_title(f"Orijinal\n({original_dim} özellik)\nSınıf: {original_class_name}", fontsize=10)
    axes[0].axis('off')

    for j, n in enumerate(n_components_list):
        pca_n = PCA(n_components=n)                    # n bileşenli PCA
        pca_n.fit(X_custom)                              # Veri setine fit et

        # Sıkıştırmak için reshape(-1, 1) yerine [sample_idx:sample_idx+1] kullanıyoruz
        compressed = pca_n.transform(X_custom[sample_idx:sample_idx+1])  # Sıkıştır
        reconstructed = pca_n.inverse_transform(compressed)            # Geri yapılandır

        # Yeniden yapılandırma hatası (MSE)
        mse = np.mean((original_flat - reconstructed[0])**2)
        var_exp = np.sum(pca_n.explained_variance_ratio_) * 100

        axes[j+1].imshow(reconstructed[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray_r', interpolation='nearest')
        axes[j+1].set_title(f"n={n}\n%{var_exp:.0f} varyans\nMSE={mse:.2f}", fontsize=9)
        axes[j+1].axis('off')

    plt.suptitle("PCA Görüntü Sıkıştırma — Bileşen Sayısı Arttıkça Kalite Yükselir",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/03_pca_sikistirma_kalite.png', dpi=150, bbox_inches='tight')
    plt.close()
else:
    print("Hata: X_custom boş olduğu için PCA sıkıştırma uygulanamadı. Veri yükleme adımlarını kontrol edin.")

"""## t-SNE ile Özellik Uzayı Görselleştirme

**t-SNE (t-distributed Stochastic Neighbor Embedding):**  
Yüksek boyutlu veriyi 2D'ye indirger; benzer örnekler yakın, farklılar uzak olur.

> PCA lineerdir — t-SNE **non-lineer**. Küme yapısını çok daha iyi ortaya çıkarır.

"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# Önce PCA ile boyutu indir (t-SNE hızlanır), sonra t-SNE ile 2 boyuta
if len(X_custom) > 0 and len(y_custom) > 0 and len(class_names) > 0:
    # X_custom'ın boyutuna göre PCA bileşen sayısını ayarlayalım
    initial_pca_components = min(50, X_custom.shape[1] // 2) # Max 50 veya yarısı
    if initial_pca_components < 2: # En az 2 bileşen olmalı
        initial_pca_components = X_custom.shape[1]

    print(f"Adım 1: PCA ile {initial_pca_components} boyuta indir...")
    pca_initial = PCA(n_components=initial_pca_components, random_state=42)
    X_pca_reduced = pca_initial.fit_transform(X_custom)         # X_custom'ı kullan
    print(f"  PCA sonrası: {X_pca_reduced.shape}")

    print("Adım 2: t-SNE ile 2 boyuta indir (bu biraz sürebilir)...")
    tsne = TSNE(
        n_components=2,       # 2D'ye yansıt
        perplexity=30,        # Yerel komşu sayısı (5-50 arası tavsiye)
        max_iter=1000,          # İterasyon sayısı
        random_state=42,      # Tekrar üretilebilirlik
        learning_rate='auto', # Otomatik öğrenme hızı
        init='pca'            # PCA başlangıcı → daha stabil yakınsama
    )
    X_tsne = tsne.fit_transform(X_pca_reduced)          # PCA sonrası veriyi kullan
    print(f"  t-SNE sonrası: {X_tsne.shape}")

    # ── Görselleştir ─────────────────────────────────────────────────────────────n
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # class_names'in uzunluğuna göre renk paletini ayarla
    colors = plt.cm.get_cmap('tab10', len(class_names))   # Sınıf sayısına göre renkler

    # Sol: t-SNE scatter plot
    for label_idx in range(len(class_names)):
        mask = y_custom == label_idx                    # Bu sınıfa ait noktaların maskesi
        axes[0].scatter(
            X_tsne[mask, 0],                        # X koordinatı
            X_tsne[mask, 1],                        # Y koordinatı
            c=[colors(label_idx)],
            label=class_names[label_idx],
            alpha=0.6,                              # Şeffaflık — üst üste binen noktalar görülsün
            s=15,                                   # Nokta boyutu
            edgecolors='none'
        )

    axes[0].set_title("t-SNE — Tüm Veri Seti\n(Her renk bir sınıfı)",
                      fontsize=13, fontweight='bold')
    axes[0].legend(title="Sınıf", ncol=2, fontsize=9, loc='upper right')
    axes[0].set_xlabel("t-SNE Boyutu 1")
    axes[0].set_ylabel("t-SNE Boyutu 2")
    axes[0].set_facecolor('#f5f5f5')
    axes[0].grid(True, alpha=0.3)

    # Sağ: t-SNE üzerine gerçek görüntüleri koy (örnekleme ile)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].set_title("t-SNE — Gerçek Görüntülerle\n(Her nokta gerçek piksel değeri)",
                      fontsize=13, fontweight='bold', color='white')

    # Her sınıftan belirli sayıda örnek seç
    rng = np.random.default_rng(0)
    for label_idx in range(len(class_names)):
        mask_idx = np.where(y_custom == label_idx)[0]
        # Yeterli örnek yoksa replace=True kullan veya daha az örnek seç
        num_samples_to_show = min(15, len(mask_idx))
        if num_samples_to_show == 0: continue

        sample_indices = rng.choice(mask_idx, size=num_samples_to_show, replace=False)

        for idx in sample_indices:
            img_flat = X_custom[idx]                         # düzleştirilmiş görüntü
            img_2d = img_flat.reshape(IMG_SIZE, IMG_SIZE) # IMG_SIZE'a göre yeniden şekillendir
            x, y_pos = X_tsne[idx, 0], X_tsne[idx, 1]  # t-SNE koordinatları

            # Görüntüyü renklendir (sınıfa göre)
            colored = colors(label_idx)[:3]      # RGB renk

            imagebox = OffsetImage(img_2d, zoom=0.5, cmap='gray_r')   # Görüntü kutusu, zoom ayarla
            ab = AnnotationBbox(imagebox, (x, y_pos),
                                frameon=True,
                                bboxprops=dict(
                                    boxstyle='round,pad=0.1',
                                    facecolor=(*colored, 0.3),   # Yarı şeffaf renkli çerçeve
                                    edgecolor=(*colored, 0.9),
                                    linewidth=1.0
                                ))
            axes[1].add_artist(ab)                       # Sahneye ekle

    axes[1].set_xlim(X_tsne[:,0].min()-3, X_tsne[:,0].max()+3)
    axes[1].set_ylim(X_tsne[:,1].min()-3, X_tsne[:,1].max()+3)
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('white')

    plt.tight_layout()
    plt.savefig('figures/04_tsne_gorsellestirme.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n💡 t-SNE'de: Birbirinden ayrışan kümeler = model kolayca ayırt edebilir demektir.")
else:
    print("Hata: X_custom, y_custom veya class_names boş olduğu için t-SNE uygulanamadı. Veri yükleme adımlarını kontrol edin.")
"""
FASHION-MNIST ILE EVRISIMLI SINIR AGI (CNN) GORUNTU SINIFLANDIRMA
Amac: Ham kiyafet gorsellerini (T-shirt, pantolon, ayakkabi, canta vb.
      10 kategori) bir Evrisimli Sinir Agi (CNN) ile siniflandirmak.

Veri seti otomatik olarak Fashion-MNIST'in resmi GitHub deposundan
(zalandoresearch/fashion-mnist) indirilir ve yerel 'data/' klasorune
onbelleklenir - hicbir manuel indirme veya API anahtari gerekmez.
"""
import os
import gzip
import struct
import urllib.request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_if_missing(filename):
    path = os.path.join("data", filename)
    if not os.path.exists(path):
        print(f"    Indiriliyor: {filename}")
        urllib.request.urlretrieve(BASE_URL + filename, path)
    return path


def read_idx_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows, cols)


def read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_fashion_mnist():
    paths = {k: download_if_missing(v) for k, v in FILES.items()}
    X_train = read_idx_images(paths["train_images"])
    y_train = read_idx_labels(paths["train_labels"])
    X_test = read_idx_images(paths["test_images"])
    y_test = read_idx_labels(paths["test_labels"])
    return (X_train, y_train), (X_test, y_test)


print("Fashion-MNIST CNN Siniflandirma")
print(f"TensorFlow Versiyonu: {tf.__version__}")

gpu_devices = tf.config.list_physical_devices("GPU")
if len(gpu_devices) > 0:
    print(f"GPU aktif: {gpu_devices[0]}")
else:
    print("GPU bulunamadi, egitim CPU uzerinden yapilacak.")

print("\nVeri seti yukleniyor (ilk calistirmada indirilir, sonra onbellekten okunur)...")
(X_train, y_train), (X_test, y_test) = load_fashion_mnist()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("Veriler normalize ediliyor (0-255 -> 0.0-1.0)...")
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print(f"Egitim seti: {X_train.shape} | Test seti: {X_test.shape}")

print("\nCNN mimarisi olusturuluyor...")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

print("\nModel egitiliyor...")
history = model.fit(
    X_train, y_train, epochs=8, batch_size=64, validation_split=0.1, verbose=2
)

print("\nEgitim tamamlandi, grafikler kaydediliyor...")
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], "b-o", label="Egitim Dogrulugu")
plt.plot(history.history["val_accuracy"], "r-o", label="Dogrulama Dogrulugu")
plt.title("Model Dogruluk Degisimi (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Skor")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], "b-o", label="Egitim Kaybi")
plt.plot(history.history["val_loss"], "r-o", label="Dogrulama Kaybi")
plt.title("Model Kayip Degisimi (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Kayip Degeri")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150)
plt.close()

print("\nTest verisiyle final degerlendirme...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("=" * 55)
print(f" TEST DOGRULUGU (ACCURACY): %{test_acc * 100:.2f}")
print(f" TEST KAYBI (LOSS): {test_loss:.4f}")
print("=" * 55)

print("\nConfusion matrix ve siniflandirma raporu hazirlaniyor...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Fashion-MNIST (10 Sinif)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("\nOrnek tahminler gorsellestiriliyor...")
rng = np.random.default_rng(RANDOM_STATE)
sample_ids = rng.choice(len(X_test), size=8, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, idx in zip(axes.flat, sample_ids):
    img = X_test[idx].squeeze()
    gercek = y_test[idx]
    tahmin = y_pred[idx]
    color = "green" if gercek == tahmin else "red"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Gercek: {class_names[gercek]}\nTahmin: {class_names[tahmin]}",
                 color=color, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("figures/sample_predictions.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

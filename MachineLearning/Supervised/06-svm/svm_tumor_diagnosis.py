"""
TUMOR TESHISI - Support Vector Machine (SVM)
Amac: Hucre gorseli olcumlerinden (yaricap, doku, cevre, alan, puruzsuzluk
      vb.) bir tumorun IYI HUYLU mu (Benign) yoksa KOTU HUYLU mu (Malignant)
      oldugunu siniflandirmak. SVM, yuksek boyutlu ve net ayrilabilir
      veride guclu oldugu icin bu tur tibbi teshis problemlerinde klasik
      bir tercihtir (Wisconsin Breast Cancer veri seti literaturde en
      cok kullanilan kiyaslama problemlerinden biridir).

      Linear ve RBF (dogrusal olmayan) kernel karsilastirilarak "karar
      siniri dogrusal mi yoksa daha karmasik mi olmali" sorusuna veriyle
      cevap aranir.

Not: Gercek Wisconsin Breast Cancer veri seti bu ortamda bulunmadigi icin,
     ayni kolon yapisini ve gercekci hucre olcum iliskilerini (buyuk
     yaricap + duzensiz sekil -> kotu huylu olma olasiligi yuksek)
     yansitan SENTETIK bir veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Tumor Teshisi (SVM) - Veri Uretimi")

n = 569  # gercek Wisconsin veri setiyle ayni buyuklukte

malignant = np.random.choice([0, 1], n, p=[0.63, 0.37])  # gercek orana yakin

def gen(mean_b, mean_m, std, size, is_m):
    return np.where(is_m == 1,
                     np.random.normal(mean_m, std, size),
                     np.random.normal(mean_b, std, size))

radius_mean = np.clip(gen(12.1, 17.5, 2.8, n, malignant), 6, 30)
texture_mean = np.clip(gen(17.9, 21.6, 4.0, n, malignant), 9, 40)
perimeter_mean = np.clip(gen(78, 115, 18, n, malignant), 40, 190)
area_mean = np.clip(gen(460, 980, 250, n, malignant), 140, 2500)
smoothness_mean = np.clip(gen(0.092, 0.102, 0.014, n, malignant), 0.05, 0.16)
compactness_mean = np.clip(gen(0.080, 0.145, 0.045, n, malignant), 0.02, 0.35)
concavity_mean = np.clip(gen(0.046, 0.160, 0.06, n, malignant), 0.0, 0.43)
symmetry_mean = np.clip(gen(0.174, 0.192, 0.027, n, malignant), 0.11, 0.30)
fractal_dimension_mean = np.clip(gen(0.063, 0.063, 0.007, n, malignant), 0.05, 0.10)

df = pd.DataFrame({
    "radius_mean": radius_mean.round(2), "texture_mean": texture_mean.round(2),
    "perimeter_mean": perimeter_mean.round(2), "area_mean": area_mean.round(1),
    "smoothness_mean": smoothness_mean.round(4), "compactness_mean": compactness_mean.round(4),
    "concavity_mean": concavity_mean.round(4), "symmetry_mean": symmetry_mean.round(4),
    "fractal_dimension_mean": fractal_dimension_mean.round(4),
    "diagnosis": np.where(malignant == 1, "M", "B")
})
print(f"Veri seti boyutu: {df.shape}")
print(f"Tani dagilimi:\n{df['diagnosis'].value_counts().to_string()}")

feature_cols = [c for c in df.columns if c != "diagnosis"]
X = df[feature_cols]
y = (df["diagnosis"] == "M").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Linear Kernel SVM ===")
svm_linear = SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)
svm_linear.fit(X_train_scaled, y_train)
pred_linear = svm_linear.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, pred_linear)
proba_linear = svm_linear.predict_proba(X_test_scaled)[:, 1]
auc_linear = roc_auc_score(y_test, proba_linear)
print(f"Accuracy: {acc_linear:.4f} | ROC-AUC: {auc_linear:.4f}")

print("\n=== RBF Kernel SVM (dogrusal olmayan) ===")
svm_rbf = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
svm_rbf.fit(X_train_scaled, y_train)
pred_rbf = svm_rbf.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, pred_rbf)
proba_rbf = svm_rbf.predict_proba(X_test_scaled)[:, 1]
auc_rbf = roc_auc_score(y_test, proba_rbf)
print(f"Accuracy: {acc_rbf:.4f} | ROC-AUC: {auc_rbf:.4f}")

best_name = "Linear" if acc_linear >= acc_rbf else "RBF"
print(f"\nEn iyi performansi veren kernel: {best_name}")
best_pred = pred_linear if best_name == "Linear" else pred_rbf
print("\nSiniflandirma Raporu (en iyi kernel):")
print(classification_report(y_test, best_pred, target_names=["Iyi Huylu (B)", "Kotu Huylu (M)"]))

print("\nConfusion matrix kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
cm_linear = confusion_matrix(y_test, pred_linear)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["Iyi Huylu", "Kotu Huylu"], yticklabels=["Iyi Huylu", "Kotu Huylu"])
axes[0].set_title(f"Linear Kernel (Acc={acc_linear:.3f})")
axes[0].set_xlabel("Tahmin"); axes[0].set_ylabel("Gercek")

cm_rbf = confusion_matrix(y_test, pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=["Iyi Huylu", "Kotu Huylu"], yticklabels=["Iyi Huylu", "Kotu Huylu"])
axes[1].set_title(f"RBF Kernel (Acc={acc_rbf:.3f})")
axes[1].set_xlabel("Tahmin"); axes[1].set_ylabel("Gercek")
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

print("ROC egrisi kiyaslamasi kaydediliyor...")
fpr_l, tpr_l, _ = roc_curve(y_test, proba_linear)
fpr_r, tpr_r, _ = roc_curve(y_test, proba_rbf)
plt.figure(figsize=(7, 6))
plt.plot(fpr_l, tpr_l, color="#2563eb", label=f"Linear (AUC={auc_linear:.3f})", linewidth=2)
plt.plot(fpr_r, tpr_r, color="#7c3aed", label=f"RBF (AUC={auc_rbf:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi: Linear vs RBF Kernel")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("Karar siniri (decision boundary) gorsellestiriliyor (PCA ile 2 boyut)...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)

svm_2d = SVC(kernel="rbf", random_state=RANDOM_STATE)
svm_2d.fit(X_train_pca, y_train)

x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train,
                       cmap="coolwarm", edgecolors="k", s=35, alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("SVM Karar Siniri (RBF Kernel, PCA ile 2 Boyut)")
plt.tight_layout()
plt.savefig("figures/decision_boundary.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

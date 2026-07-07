"""
KREDI KARTI ISLEM DOLANDIRICILIK TESPITI - Random Forest
Amac: Bir kredi karti isleminin sahte (fraud) olup olmadigini tahmin etmek.
      Random Forest bu alanda tercih edilir cunku hem yuksek dogruluk
      saglar hem de XGBoost kadar hassas hiperparametre ayari gerektirmez,
      bu da production ortaminda hizli devreye alinabilmesini kolaylastirir.
      Feature importance ciktisi risk ekiplerine "hangi davranis suphelidir"
      diye dogrudan aktarilabilir.

Not: Gercek bir banka/odeme verisi kullanilmamistir (gizlilik/erisim
     kisitlari nedeniyle). Bilinen dolandiricilik oruntulerini (yuksek
     tutar + gece saati + yeni cihaz + yurtdisi islem -> yuksek risk)
     yansitan SENTETIK bir veri seti uretilir. Ayrica gercek dolandiricilik
     verilerinde oldugu gibi siniflar dengesizdir (fraud << normal islem).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Dolandiricilik Tespiti - Veri Uretimi")

n = 20000
fraud_rate = 0.03   # gercek hayatta oldugu gibi dengesiz siniflar

is_fraud = np.random.choice([0, 1], n, p=[1 - fraud_rate, fraud_rate])

amount = np.where(
    is_fraud == 1,
    np.random.lognormal(mean=5.6, sigma=1.2, size=n),
    np.random.lognormal(mean=4.3, sigma=1.2, size=n)
)

# Dolandiricilik islemleri gece saatlerinde hafifce daha yogun (kesin degil, orintili)
night_weights = np.array([1.6 if h in [1, 2, 3, 4] else 1.0 for h in range(24)])
night_weights = night_weights / night_weights.sum()
day_weights = np.array([1.15 if h in [1, 2, 3, 4] else 0.97 for h in range(24)])
day_weights = day_weights / day_weights.sum()

hour = np.array([
    np.random.choice(24, p=night_weights) if f == 1 else np.random.choice(24, p=day_weights)
    for f in is_fraud
])

is_new_device = np.where(
    is_fraud == 1,
    np.random.choice([0, 1], n, p=[0.45, 0.55]),
    np.random.choice([0, 1], n, p=[0.83, 0.17])
)
is_foreign = np.where(
    is_fraud == 1,
    np.random.choice([0, 1], n, p=[0.55, 0.45]),
    np.random.choice([0, 1], n, p=[0.90, 0.10])
)
distance_from_home_km = np.where(
    is_fraud == 1,
    np.random.exponential(180, n),
    np.random.exponential(35, n)
)
transactions_last_hour = np.where(
    is_fraud == 1,
    np.random.poisson(1.8, n),
    np.random.poisson(0.5, n)
)

# Etiketlere gercekci bir belirsizlik ekle: bazi fraud islemler "temiz" gorunur,
# bazi normal islemler "supheli" gorunur (gercek hayatta hicbir oruntu %100 net degildir)
label_noise_idx = np.random.random(n) < 0.015
is_fraud = np.where(label_noise_idx, 1 - is_fraud, is_fraud)

df = pd.DataFrame({
    "amount": amount.round(2),
    "hour": hour,
    "is_new_device": is_new_device,
    "is_foreign": is_foreign,
    "distance_from_home_km": distance_from_home_km.round(1),
    "transactions_last_hour": transactions_last_hour,
    "is_fraud": is_fraud
})
print(f"Uretilen islem sayisi: {len(df)}")
print(f"Fraud orani: %{100*df['is_fraud'].mean():.2f}")

feature_cols = ["amount", "hour", "is_new_device", "is_foreign",
                 "distance_from_home_km", "transactions_last_hour"]
X = df[feature_cols]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print("\nRandom Forest egitiliyor (class_weight=balanced, dengesiz siniflar icin)...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC : {auc:.4f}")
print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("\nGorseller kaydediliyor...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Dolandiricilik Tespiti")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#dc2626", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=150)
plt.close()

# Dengesiz siniflarda ROC yaniltici olabilir, Precision-Recall egrisi de eklenir
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="#991b1b", linewidth=2,
         label=f"PR (Avg Precision = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Egrisi (dengesiz siniflar icin)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/precision_recall_curve.png", dpi=150)
plt.close()

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="Reds_r", legend=False)
plt.title("Ozellik Onem Duzeyleri (Feature Importance)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
importance_df.to_csv("figures/feature_importance.csv", index=False)

print("Kaydedildi: figures/confusion_matrix.png")
print("Kaydedildi: figures/roc_curve.png")
print("Kaydedildi: figures/precision_recall_curve.png")
print("Kaydedildi: figures/feature_importance.png")

print("\nTamamlandi.")

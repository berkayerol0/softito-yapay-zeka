"""
KREDI KARTI/BANKA ISLEMI ANOMALI TESPITI - Isolation Forest
Amac: Kredi karti islemlerinde ETIKET (fraud/normal) KULLANMADAN, sadece
      islem ozelliklerinin genel dagilimina bakarak "normalden sapan"
      supheli islemleri tespit etmek. Bu, Random Forest fraud projesinden
      (04-random-forest) temelde farkli bir yaklasimdir: Random Forest
      GECMISTE ETIKETLENMIS sahte islem ornekleri gerektirirken, Isolation
      Forest hicbir etikete ihtiyac duymadan calisir - bu da onu DAHA ONCE
      GORULMEMIS, yeni fraud oruntulerini yakalamada degerli kilar.

      Gercek bankacilik sistemlerinde bu iki yaklasim BIRLIKTE kullanilir:
      Isolation Forest yeni/bilinmeyen oruntuleri yakalayan bir "on filtre",
      etiketli modeller (Random Forest, XGBoost) ise bilinen oruntuler icin
      ikinci, daha kesin bir katman olarak calisir.

Not: Gercek bir banka islem verisi bu ortamda bulunmadigi icin, gercekci
     islem oruntularini yansitan SENTETIK bir veri seti uretilir. Fraud
     etiketleri SADECE MODELIN BASARISINI DEGERLENDIRMEK icin kullanilir,
     modelin egitiminde hic kullanilmaz (gercek denetimsiz ogrenme).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, classification_report, confusion_matrix
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Kredi Karti Anomali Tespiti (Isolation Forest) - Veri Uretimi")

n = 20000
fraud_rate = 0.035

is_fraud_true = np.random.choice([0, 1], n, p=[1 - fraud_rate, fraud_rate])

amount = np.where(
    is_fraud_true == 1,
    np.random.lognormal(mean=5.6, sigma=1.2, size=n),
    np.random.lognormal(mean=4.3, sigma=1.2, size=n)
)
night_weights = np.array([1.6 if h in [1, 2, 3, 4] else 1.0 for h in range(24)])
night_weights = night_weights / night_weights.sum()
day_weights = np.array([1.15 if h in [1, 2, 3, 4] else 0.97 for h in range(24)])
day_weights = day_weights / day_weights.sum()
hour = np.array([
    np.random.choice(24, p=night_weights) if f == 1 else np.random.choice(24, p=day_weights)
    for f in is_fraud_true
])
is_new_device = np.where(
    is_fraud_true == 1,
    np.random.choice([0, 1], n, p=[0.45, 0.55]),
    np.random.choice([0, 1], n, p=[0.83, 0.17])
)
is_foreign = np.where(
    is_fraud_true == 1,
    np.random.choice([0, 1], n, p=[0.55, 0.45]),
    np.random.choice([0, 1], n, p=[0.90, 0.10])
)
distance_from_home_km = np.where(
    is_fraud_true == 1,
    np.random.exponential(180, n),
    np.random.exponential(35, n)
)
transactions_last_hour = np.where(
    is_fraud_true == 1,
    np.random.poisson(1.8, n),
    np.random.poisson(0.5, n)
)

df = pd.DataFrame({
    "amount": amount.round(2), "hour": hour, "is_new_device": is_new_device,
    "is_foreign": is_foreign, "distance_from_home_km": distance_from_home_km.round(1),
    "transactions_last_hour": transactions_last_hour, "is_fraud_true": is_fraud_true
})
print(f"Uretilen islem sayisi: {len(df)}")
print(f"Gercek fraud orani (SADECE degerlendirme icin, model bunu gormeyecek): %{100*df['is_fraud_true'].mean():.2f}")

feature_cols = ["amount", "hour", "is_new_device", "is_foreign",
                 "distance_from_home_km", "transactions_last_hour"]
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nIsolation Forest egitiliyor (ETIKETSIZ - fraud sutunu hic kullanilmiyor)...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=fraud_rate,   # beklenen anomali orani (is bilgisinden - etiketten degil)
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

anomaly_pred = iso_forest.predict(X_scaled)   # -1: anomali, 1: normal
anomaly_score = -iso_forest.score_samples(X_scaled)  # yuksek skor = daha supheli

df["predicted_anomaly"] = (anomaly_pred == -1).astype(int)
df["anomaly_score"] = anomaly_score

print("\nModel performansi DEGERLENDIRILIYOR (gercek etiketlerle kiyaslanarak)...")
auc = roc_auc_score(df["is_fraud_true"], df["anomaly_score"])
ap = average_precision_score(df["is_fraud_true"], df["anomaly_score"])
print(f"ROC-AUC: {auc:.4f}")
print(f"PR-AUC (Average Precision): {ap:.4f}")
print("\nSiniflandirma Raporu (isaretlenen anomaliler vs gercek fraud):")
print(classification_report(df["is_fraud_true"], df["predicted_anomaly"],
                             target_names=["Normal", "Anomali/Fraud"]))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(df["is_fraud_true"], df["predicted_anomaly"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=["Normal", "Anomali"], yticklabels=["Normal", "Fraud"])
plt.xlabel("Model Tahmini")
plt.ylabel("Gercek Durum")
plt.title("Isolation Forest - Anomali Tespiti Sonucu")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("ROC ve Precision-Recall egrileri kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(df["is_fraud_true"], df["anomaly_score"])
axes[0].plot(fpr, tpr, color="#7c3aed", linewidth=2, label=f"AUC={auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Egrisi")
axes[0].legend()

prec, rec, _ = precision_recall_curve(df["is_fraud_true"], df["anomaly_score"])
axes[1].plot(rec, prec, color="#7c3aed", linewidth=2, label=f"AP={ap:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=150)
plt.close()

print("Anomali skoru dagilimi kaydediliyor...")
plt.figure(figsize=(9, 6))
sns.histplot(data=df, x="anomaly_score", hue="is_fraud_true", bins=60,
             palette={0: "#3b82f6", 1: "#dc2626"}, alpha=0.6,
             element="step", stat="density", common_norm=False)
plt.xlabel("Anomali Skoru (yuksek = daha supheli)")
plt.title("Anomali Skoru Dagilimi (Gercek Fraud vs Normal)")
plt.legend(labels=["Fraud", "Normal"])
plt.tight_layout()
plt.savefig("figures/anomaly_score_distribution.png", dpi=150)
plt.close()

print("PCA ile 2 boyutlu gorsellestirme...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["predicted_anomaly"],
                       cmap="coolwarm", alpha=0.5, s=15)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Isolation Forest Anomali Tespiti (PCA ile 2 Boyut)")
plt.colorbar(scatter, label="0=Normal, 1=Anomali")
plt.tight_layout()
plt.savefig("figures/anomaly_pca.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

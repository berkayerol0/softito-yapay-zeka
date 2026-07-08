"""
MUSTERI KAYBI (CHURN) TAHMINI - Logistic Regression
Amac: Bir SaaS/telekom musterisinin onumuzdeki donemde hizmeti birakma
      (churn) olasiligini tahmin etmek ve HANGI faktorlerin churn riskini
      artirdigini/azalttigini aciklayabilmek. Musteri basarisi (customer
      success) ekipleri bu tur skorlari "riskli musteriye once ulasalim"
      seklinde aksiyon almak icin kullanir.

Not: Gercek bir sirket verisi kullanilmadigi icin, gercekci iliskiler
     (kisa sozlesme + dusuk kullanim + cok destek talebi -> yuksek churn)
     iceren SENTETIK bir veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)


# 1) SENTETIK MUSTERI VERISI URETIMI

print("=" * 60)
print("CHURN TAHMINI - VERI URETIMI")
print("=" * 60)

n = 4000
tenure_months = np.clip(np.random.exponential(20, n), 1, 72).astype(int)
monthly_charges = np.random.normal(65, 20, n).clip(15, 150)
contract = np.random.choice(["Aylik", "1 Yillik", "2 Yillik"], n, p=[0.5, 0.3, 0.2])
support_calls = np.random.poisson(2, n)
usage_score = np.clip(np.random.normal(60, 20, n), 0, 100)   # urun kullanim yogunlugu (0-100)
has_addon = np.random.choice([0, 1], n, p=[0.6, 0.4])
payment_delay_count = np.random.poisson(0.8, n)

contract_risk = {"Aylik": 1.0, "1 Yillik": 0.3, "2 Yillik": 0.05}
contract_factor = np.array([contract_risk[c] for c in contract])

score = (
    2.0 * contract_factor
    + 0.05 * support_calls
    - 0.03 * usage_score / 10
    - 0.02 * tenure_months / 5
    + 0.25 * payment_delay_count
    - 0.4 * has_addon
    + np.random.normal(0, 0.9, n)
)
churn_prob = 1 / (1 + np.exp(-(score - score.mean())))
churn = (churn_prob > 0.5).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges.round(2),
    "contract": contract,
    "support_calls": support_calls,
    "usage_score": usage_score.round(1),
    "has_addon": has_addon,
    "payment_delay_count": payment_delay_count,
    "churn": churn
})
print(f"Uretilen kayit sayisi: {len(df)}")
print(f"Churn orani: %{100*df['churn'].mean():.1f}")


# 2) ON ISLEME

print("\n[2] Kategorik degiskenleri encode etme...")
le_contract = LabelEncoder()
df["contract_enc"] = le_contract.fit_transform(df["contract"])

feature_cols = [
    "tenure_months", "monthly_charges", "support_calls",
    "usage_score", "has_addon", "payment_delay_count", "contract_enc"
]
X = df[feature_cols]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 3) MODEL EGITIMI

print("\n[3] Logistic Regression egitiliyor..")
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"    Accuracy: {acc:.4f}")
print(f"    ROC-AUC : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Kalici", "Churn"]))


# 4) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=["Kalici", "Churn"], yticklabels=["Kalici", "Churn"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Churn Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

# --- ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#ea580c", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=150)
plt.close()

# --- Katsayilar
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

plt.figure(figsize=(8, 5))
colors = ["#dc2626" if c > 0 else "#059669" for c in coef_df["Coefficient"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Model Katsayilari (Churn Riskine Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()
coef_df.to_csv("figures/coefficients.csv", index=False)

# --- Segment bazli churn orani (is aciklanabilirligi) ---
plt.figure(figsize=(7, 5))
churn_by_contract = df.groupby("contract")["churn"].mean().sort_values(ascending=False) * 100
sns.barplot(x=churn_by_contract.index, y=churn_by_contract.values, color="#f97316")
plt.ylabel("Churn Orani (%)")
plt.xlabel("Sozlesme Tipi")
plt.title("Sozlesme Tipine Gore Churn Orani")
plt.tight_layout()
plt.savefig("figures/churn_by_contract.png", dpi=150)
plt.close()

print("    Kaydedildi: figures/confusion_matrix.png")
print("    Kaydedildi: figures/roc_curve.png")
print("    Kaydedildi: figures/coefficients.png")
print("    Kaydedildi: figures/churn_by_contract.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)

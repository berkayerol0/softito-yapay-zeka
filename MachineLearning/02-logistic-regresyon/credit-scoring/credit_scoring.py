"""
KREDI SKORLAMA - Logistic Regression
Amac: Basvuru bilgilerine bakarak bir kredinin onaylanip onaylanmayacagini
      tahmin etmek VE modelin bu karari NEDEN verdigini (katsayilar uzerinden)
      aciklayabilmek. Fintech/bankacilikta duzenleyici zorunluluklar yuzunden
      "kara kutu olmayan" modeller (Logistic Regression, GLM) hala tercih edilir.

Not: Gercek bir banka veri seti kullanilmadigi icin, gercekci istatistiksel
     iliskiler (gelir arttikca onay olasiligi artar, borc arttikca azalir vb.)
     iceren SENTETIK bir veri seti uretilir. Kod bunu acikca belirtir.
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


# 1) SENTETIK KREDI BASVURU VERISI URETIMI

print("=" * 60)
print("KREDI SKORLAMA - VERI URETIMI")
print("=" * 60)

n = 4000
age = np.random.randint(21, 70, n)
income = np.random.lognormal(mean=10.2, sigma=0.5, size=n)          # yillik gelir
loan_amount = np.random.lognormal(mean=9.0, sigma=0.6, size=n)      # istenen kredi tutari
credit_history_years = np.clip(np.random.normal(7, 4, n), 0, 30)
existing_debt = np.random.lognormal(mean=8.5, sigma=0.7, size=n)
employment_years = np.clip(np.random.normal(5, 4, n), 0, 40)
education = np.random.choice(["Lise", "Lisans", "Yuksek Lisans"], n, p=[0.4, 0.45, 0.15])
purpose = np.random.choice(["Konut", "Tasit", "Egitim", "Ihtiyac"], n, p=[0.3, 0.25, 0.15, 0.3])

debt_to_income = existing_debt / income
loan_to_income = loan_amount / income

# Gercekci bir "gizli" skor fonksiyonu (bunu model gormeyecek, sadece etiket uretmek icin)
score = (
    0.9 * (income / income.mean())
    - 1.4 * debt_to_income
    - 1.1 * loan_to_income
    + 0.6 * (credit_history_years / 10)
    + 0.4 * (employment_years / 10)
    - 0.3 * (age < 25).astype(int)
    + np.random.normal(0, 0.8, n)          # gurultu
)
approval_prob = 1 / (1 + np.exp(-(score - score.mean())))
approved = (approval_prob > 0.5).astype(int)

df = pd.DataFrame({
    "age": age,
    "income": income.round(0),
    "loan_amount": loan_amount.round(0),
    "credit_history_years": credit_history_years.round(1),
    "existing_debt": existing_debt.round(0),
    "employment_years": employment_years.round(1),
    "education": education,
    "purpose": purpose,
    "debt_to_income": debt_to_income.round(3),
    "loan_to_income": loan_to_income.round(3),
    "approved": approved
})
print(f"Uretilen kayit sayisi: {len(df)}")
print(f"Onay orani: %{100*df['approved'].mean():.1f}")


# 2) ON ISLEME

print("\n[2] Kategorik degiskenleri encode etme...")
encoders = {}
for col in ["education", "purpose"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "age", "income", "loan_amount", "credit_history_years",
    "existing_debt", "employment_years", "debt_to_income", "loan_to_income",
    "education_enc", "purpose_enc"
]
X = df[feature_cols]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# 3) MODEL EGITIMI

print("\n[3] Logistic Regression egitiliyor...")
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"    Accuracy: {acc:.4f}")
print(f"    ROC-AUC : {auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["Reddedildi", "Onaylandi"]))


# 4) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Reddedildi", "Onaylandi"],
            yticklabels=["Reddedildi", "Onaylandi"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Kredi Onay Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#2563eb", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png", dpi=150)
plt.close()

# --- Katsayilar (aciklanabilirlik)
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False)

plt.figure(figsize=(8, 5))
colors = ["#059669" if c > 0 else "#dc2626" for c in coef_df["Coefficient"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Model Katsayilari (Onay Olasiligina Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()

coef_df.to_csv("figures/coefficients.csv", index=False)

print("    Kaydedildi: figures/confusion_matrix.png")
print("    Kaydedildi: figures/roc_curve.png")
print("    Kaydedildi: figures/coefficients.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)

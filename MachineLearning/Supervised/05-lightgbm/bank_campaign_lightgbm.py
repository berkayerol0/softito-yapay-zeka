"""
BANKA KAMPANYASI KATILIM TAHMINI - LightGBM
Amac: Bir bankanin telefonla yaptigi vadeli mevduat kampanyasinda, bir
      musterinin kampanyaya KATILIP KATILMAYACAGINI yuksek dogrulukla
      tahmin etmek. Bu, UCI Bank Marketing veri setine dayanan, veri
      bilimi camiasinda cok bilinen rekabetci bir kiyaslama (benchmark)
      problemidir - Kaggle'da da benzer versiyonlari siklikla kullanilir.

      LightGBM burada TEK BASINA, dogrulugu maksimize etmeye odakli
      kullanilir: early stopping, ogrenme egrisi takibi ve hiperparametre
      secimleriyle "en iyi sonucu nasil cikaririm" sorusuna odaklanilir.

Not: Gercek UCI Bank Marketing veri seti bu ortamda bulunmadigi icin, ayni
     kolon yapisini ve gercekci kampanya davranis oruntulerini (uzun
     gorusme suresi + onceki basarili kampanya + uygun yas araligi ->
     yuksek katilim olasiligi) yansitan SENTETIK bir veri seti uretilir.
     Gercek veri setinde oldugu gibi siniflar dengesizdir (katilim orani
     dusuktur, ~%11).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, accuracy_score, classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Banka Kampanyasi Katilim Tahmini - Veri Uretimi")

n = 25000

age = np.clip(np.random.normal(41, 12, n), 18, 90).astype(int)
job = np.random.choice(
    ["management", "technician", "blue-collar", "admin", "services",
     "retired", "student", "entrepreneur", "unemployed"],
    n, p=[0.16, 0.15, 0.18, 0.12, 0.10, 0.08, 0.05, 0.08, 0.08]
)
marital = np.random.choice(["married", "single", "divorced"], n, p=[0.58, 0.28, 0.14])
education = np.random.choice(["primary", "secondary", "tertiary"], n, p=[0.15, 0.5, 0.35])
balance = np.clip(np.random.normal(1400, 3000, n), -3000, 40000)
housing_loan = np.random.choice([0, 1], n, p=[0.44, 0.56])
personal_loan = np.random.choice([0, 1], n, p=[0.84, 0.16])
contact_cellular = np.random.choice([0, 1], n, p=[0.35, 0.65])
day_of_month = np.random.randint(1, 29, n)
month = np.random.choice(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], n
)
duration_sec = np.clip(np.random.exponential(220, n), 5, 3000)   # gorusme suresi
campaign_contacts = np.clip(np.random.poisson(2.5, n), 1, 30)
pdays = np.random.choice([-1] + list(range(1, 400)), n,
                          p=[0.82] + [0.18/399]*399)   # -1: daha once aranmamis
previous_contacts = np.clip(np.random.poisson(0.6, n), 0, 15)
poutcome_success = np.where(
    (pdays > 0), np.random.choice([0, 1], n, p=[0.65, 0.35]), 0
)

subscribe_score = (
    0.004 * duration_sec
    + 1.6 * poutcome_success
    + 0.3 * (education == "tertiary").astype(int)
    - 0.25 * campaign_contacts
    + 0.35 * ((age < 30) | (age > 60)).astype(int)
    + 0.0001 * balance
    - 0.5 * personal_loan
    + np.random.normal(0, 1.1, n)
)
target_rate = 0.11
subscribed = (subscribe_score > np.percentile(subscribe_score, 100 * (1 - target_rate))).astype(int)

df = pd.DataFrame({
    "age": age, "job": job, "marital": marital, "education": education,
    "balance": balance.round(0), "housing_loan": housing_loan, "personal_loan": personal_loan,
    "contact_cellular": contact_cellular, "day_of_month": day_of_month, "month": month,
    "duration_sec": duration_sec.round(0), "campaign_contacts": campaign_contacts,
    "pdays": pdays, "previous_contacts": previous_contacts,
    "poutcome_success": poutcome_success, "subscribed": subscribed
})
print(f"Uretilen musteri sayisi: {len(df)}")
print(f"Kampanyaya katilim orani: %{100*df['subscribed'].mean():.2f}")

print("\nKategorik degiskenler encode ediliyor...")
cat_cols = ["job", "marital", "education", "month"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "age", "job_enc", "marital_enc", "education_enc", "balance",
    "housing_loan", "personal_loan", "contact_cellular", "day_of_month",
    "month_enc", "duration_sec", "campaign_contacts", "pdays",
    "previous_contacts", "poutcome_success"
]
X = df[feature_cols]
y = df["subscribed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)
print(f"Egitim: {X_train.shape} | Dogrulama: {X_val.shape} | Test: {X_test.shape}")

print("\nLightGBM egitiliyor (early stopping ile, en iyi iterasyon otomatik secilir)...")
model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
)
print(f"Erken durdurmayla secilen en iyi iterasyon: {model.best_iteration_}")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)
test_ap = average_precision_score(y_test, y_proba)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test ROC-AUC : {test_auc:.4f}")
print(f"Test PR-AUC  : {test_ap:.4f}")
print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Katilmadi", "Katildi"]))

print("\nOgrenme egrisi (train vs validation AUC) kaydediliyor...")
results = model.evals_result_
plt.figure(figsize=(9, 6))
plt.plot(results["training"]["auc"], label="Egitim AUC", color="#2563eb")
plt.plot(results["valid_1"]["auc"], label="Dogrulama AUC", color="#dc2626")
plt.axvline(model.best_iteration_, color="gray", linestyle="--",
            label=f"Erken durdurma noktasi ({model.best_iteration_})")
plt.xlabel("Iterasyon (Agac Sayisi)")
plt.ylabel("AUC")
plt.title("LightGBM Ogrenme Egrisi (Early Stopping ile)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/learning_curve.png", dpi=150)
plt.close()

print("Confusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Katilmadi", "Katildi"], yticklabels=["Katilmadi", "Katildi"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Kampanya Katilim Tahmini")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("ROC ve Precision-Recall egrileri kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#059669", linewidth=2, label=f"AUC={test_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="gray")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Egrisi")
axes[0].legend()

prec, rec, _ = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec, prec, color="#059669", linewidth=2, label=f"AP={test_ap:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/roc_pr_curves.png", dpi=150)
plt.close()

print("Feature importance kaydediliyor...")
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
print(importance_df.head(10).to_string(index=False))
importance_df.to_csv("figures/feature_importance.csv", index=False)

plt.figure(figsize=(9, 6))
sns.barplot(data=importance_df.head(12), x="Importance", y="Feature",
            hue="Feature", palette="Greens_r", legend=False)
plt.title("LightGBM Feature Importance (Top 12)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

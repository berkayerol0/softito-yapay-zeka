"""
TIKLANMA TAHMINI (CTR PREDICTION): XGBoost vs LightGBM
Amac: Bir kullanicinin bir reklama/urune tiklayip tiklamayacagini (CTR -
      Click-Through Rate) tahmin etmek. Bu, her buyuk reklam/e-ticaret
      platformunun (Google Ads, Meta, Amazon) arka planda calistirdigi
      tipik bir problemdir.

      Tek bir kutuphane secmek yerine XGBoost ve LightGBM AYNI VERIDE
      egitilip HIZ ve DOGRULUK acisindan kiyaslanir - sektorde gercekten
      yapilan pratik budur: bir ML ekibi genelde ikisini de dener, hangisi
      kendi verisinde daha iyi calisiyorsa onu secer.

Not: Gercek bir reklam platformu verisi kullanilmamistir (gizlilik/erisim
     kisitlari). Bilinen CTR oruntulerini (ilgili kategori + iyi konum +
     tekrar gorunum -> yuksek tiklama olasiligi) yansitan SENTETIK bir
     veri seti uretilir. Gercek CTR verilerinde oldugu gibi siniflar
     dengesizdir (tiklama orani genelde dusuktur).
"""
import os
import time
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
    precision_recall_curve, accuracy_score, classification_report
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("CTR Tahmini - Veri Uretimi")

n = 60000

device = np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1])
category_match = np.random.choice([0, 1], n, p=[0.55, 0.45])  # reklam kullanici ilgisiyle eslesiyor mu
ad_position = np.random.choice([1, 2, 3, 4, 5], n, p=[0.35, 0.25, 0.2, 0.1, 0.1])
hour_of_day = np.random.randint(0, 24, n)
day_of_week = np.random.randint(0, 7, n)
prior_click_rate = np.clip(np.random.beta(2, 20, n), 0, 1)  # kullanicinin gecmis tiklama orani
ad_price_tier = np.random.choice(["low", "mid", "high"], n, p=[0.4, 0.4, 0.2])
num_previous_views = np.random.poisson(3, n)
is_weekend = np.isin(day_of_week, [5, 6]).astype(int)

position_effect = {1: 0.9, 2: 0.6, 3: 0.35, 4: 0.15, 5: 0.05}
pos_score = np.array([position_effect[p] for p in ad_position])
price_effect = {"low": 0.1, "mid": 0.0, "high": -0.15}
price_score = np.array([price_effect[p] for p in ad_price_tier])

click_score = (
    1.8 * category_match
    + 1.6 * pos_score
    + 2.5 * prior_click_rate
    + 0.05 * num_previous_views
    + price_score
    + 0.15 * ((hour_of_day >= 19) & (hour_of_day <= 23)).astype(int)
    + np.random.normal(0, 0.6, n)
)
click_prob = 1 / (1 + np.exp(-(click_score - click_score.mean())))
clicked = (click_prob > np.percentile(click_prob, 88)).astype(int)   # gercekci dusuk CTR (~%12)

df = pd.DataFrame({
    "device": device, "category_match": category_match, "ad_position": ad_position,
    "hour_of_day": hour_of_day, "day_of_week": day_of_week,
    "prior_click_rate": prior_click_rate.round(4), "ad_price_tier": ad_price_tier,
    "num_previous_views": num_previous_views, "is_weekend": is_weekend, "clicked": clicked
})
print(f"Uretilen gosterim (impression) sayisi: {len(df)}")
print(f"Tiklama orani (CTR): %{100*df['clicked'].mean():.2f}")

le_device = LabelEncoder()
df["device_enc"] = le_device.fit_transform(df["device"])
le_price = LabelEncoder()
df["ad_price_tier_enc"] = le_price.fit_transform(df["ad_price_tier"])

feature_cols = [
    "device_enc", "category_match", "ad_position", "hour_of_day", "day_of_week",
    "prior_click_rate", "ad_price_tier_enc", "num_previous_views", "is_weekend"
]
X = df[feature_cols]
y = df["clicked"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Egitim seti: {X_train.shape} | Test seti: {X_test.shape}")

print("\n=== XGBoost egitiliyor ===")
t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_train_time = time.time() - t0
print(f"Egitim suresi: {xgb_train_time:.2f} saniye")

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"Accuracy: {xgb_acc:.4f} | ROC-AUC: {xgb_auc:.4f} | PR-AUC: {xgb_ap:.4f}")

print("\n=== LightGBM egitiliyor ===")
t0 = time.time()
lgbm_model = LGBMClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.9,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
)
lgbm_model.fit(X_train, y_train)
lgbm_train_time = time.time() - t0
print(f"Egitim suresi: {lgbm_train_time:.2f} saniye")

lgbm_pred = lgbm_model.predict(X_test)
lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
lgbm_auc = roc_auc_score(y_test, lgbm_proba)
lgbm_ap = average_precision_score(y_test, lgbm_proba)
lgbm_acc = accuracy_score(y_test, lgbm_pred)
print(f"Accuracy: {lgbm_acc:.4f} | ROC-AUC: {lgbm_auc:.4f} | PR-AUC: {lgbm_ap:.4f}")

print("\nROC egrisi kiyaslamasi kaydediliyor...")
fpr_x, tpr_x, _ = roc_curve(y_test, xgb_proba)
fpr_l, tpr_l, _ = roc_curve(y_test, lgbm_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr_x, tpr_x, color="#dc2626", label=f"XGBoost (AUC={xgb_auc:.3f})", linewidth=2)
plt.plot(fpr_l, tpr_l, color="#059669", label=f"LightGBM (AUC={lgbm_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Egrisi: XGBoost vs LightGBM")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("Precision-Recall egrisi kaydediliyor (dengesiz siniflar icin)...")
prec_x, rec_x, _ = precision_recall_curve(y_test, xgb_proba)
prec_l, rec_l, _ = precision_recall_curve(y_test, lgbm_proba)
plt.figure(figsize=(7, 6))
plt.plot(rec_x, prec_x, color="#dc2626", label=f"XGBoost (AP={xgb_ap:.3f})", linewidth=2)
plt.plot(rec_l, prec_l, color="#059669", label=f"LightGBM (AP={lgbm_ap:.3f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Egrisi: XGBoost vs LightGBM")
plt.legend()
plt.tight_layout()
plt.savefig("figures/pr_comparison.png", dpi=150)
plt.close()

print("Egitim suresi kiyaslamasi kaydediliyor...")
plt.figure(figsize=(7, 5))
bars = plt.bar(["XGBoost", "LightGBM"], [xgb_train_time, lgbm_train_time],
                color=["#dc2626", "#059669"])
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
              f"{b.get_height():.2f}s", ha="center", fontweight="bold")
plt.ylabel("Egitim Suresi (saniye)")
plt.title(f"Egitim Suresi Kiyaslamasi ({len(X_train):,} satir)")
plt.tight_layout()
plt.savefig("figures/training_time_comparison.png", dpi=150)
plt.close()

print("Feature importance kiyaslamasi kaydediliyor...")
xgb_imp = pd.DataFrame({
    "Feature": feature_cols, "Importance": xgb_model.feature_importances_, "Model": "XGBoost"
})
lgbm_imp_raw = lgbm_model.feature_importances_ / lgbm_model.feature_importances_.sum()
lgbm_imp = pd.DataFrame({
    "Feature": feature_cols, "Importance": lgbm_imp_raw, "Model": "LightGBM"
})
imp_combined = pd.concat([xgb_imp, lgbm_imp])
imp_combined.to_csv("figures/feature_importance_comparison.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_combined, x="Importance", y="Feature", hue="Model",
            palette={"XGBoost": "#dc2626", "LightGBM": "#059669"})
plt.title("Feature Importance Kiyaslamasi")
plt.tight_layout()
plt.savefig("figures/feature_importance_comparison.png", dpi=150)
plt.close()

print("\n=== SONUC OZETI ===")
summary = pd.DataFrame({
    "Model": ["XGBoost", "LightGBM"],
    "Accuracy": [xgb_acc, lgbm_acc],
    "ROC_AUC": [xgb_auc, lgbm_auc],
    "PR_AUC": [xgb_ap, lgbm_ap],
    "Egitim_Suresi_sn": [xgb_train_time, lgbm_train_time],
})
print(summary.to_string(index=False))
summary.to_csv("figures/model_comparison_summary.csv", index=False)

speed_ratio = xgb_train_time / lgbm_train_time
print(f"\nLightGBM, XGBoost'tan {speed_ratio:.1f}x daha hizli egitildi." if speed_ratio > 1
      else f"\nXGBoost, LightGBM'den {1/speed_ratio:.1f}x daha hizli egitildi.")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

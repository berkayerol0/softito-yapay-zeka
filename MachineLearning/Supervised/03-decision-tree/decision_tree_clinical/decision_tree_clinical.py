"""
KLINIK KARAR KURALI CIKARMA - Decision Tree
Amac: Bir hastanin risk durumunu (dusuk/orta/yuksek) tahmin eden, ama daha
      da onemlisi bu tahmini ACIK VE TAKIP EDILEBILIR KURALLARA dokebilen
      bir model kurmak. Saglik sektorunde hekimler "neden bu hasta yuksek
      riskli" sorusuna somut, if-else mantiginda cevap isterler - bu yuzden
      Random Forest/XGBoost yerine bilincli olarak tek bir Decision Tree
      tercih edilmistir; agac gorsellestirilip klinik bir akis semasi gibi
      okunabilir.

Not: Gercek hasta verisi kullanilmamistir (gizlilik nedeniyle erisilemez).
     Bilinen klinik risk faktorlerini (yas, tansiyon, kolesterol, BMI,
     sigara kullanimi vb.) yansitan SENTETIK bir veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Klinik Risk Siniflandirmasi - Veri Uretimi")

n = 2500
age = np.random.randint(20, 85, n)
systolic_bp = np.clip(np.random.normal(125, 18, n), 90, 200)
cholesterol = np.clip(np.random.normal(200, 35, n), 120, 320)
bmi = np.clip(np.random.normal(26, 5, n), 16, 45)
smoker = np.random.choice([0, 1], n, p=[0.72, 0.28])
family_history = np.random.choice([0, 1], n, p=[0.75, 0.25])
exercise_hours_week = np.clip(np.random.exponential(2.5, n), 0, 14)

risk_score = (
    0.05 * (age - 40)
    + 0.045 * (systolic_bp - 120)
    + 0.03 * (cholesterol - 190)
    + 0.11 * (bmi - 25)
    + 2.4 * smoker
    + 1.6 * family_history
    - 0.35 * exercise_hours_week
    + np.random.normal(0, 1.8, n)
)

low_th, high_th = np.percentile(risk_score, [40, 75])
risk_level = np.where(risk_score <= low_th, "Dusuk",
              np.where(risk_score <= high_th, "Orta", "Yuksek"))

df = pd.DataFrame({
    "age": age,
    "systolic_bp": systolic_bp.round(0),
    "cholesterol": cholesterol.round(0),
    "bmi": bmi.round(1),
    "smoker": smoker,
    "family_history": family_history,
    "exercise_hours_week": exercise_hours_week.round(1),
    "risk_level": risk_level
})
print(f"Uretilen kayit sayisi: {len(df)}")
print(df["risk_level"].value_counts().to_string())

feature_cols = ["age", "systolic_bp", "cholesterol", "bmi",
                 "smoker", "family_history", "exercise_hours_week"]
X = df[feature_cols]
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nDecision Tree egitiliyor (max_depth=4, yorumlanabilirlik icin sinirli)...")
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\n" + classification_report(y_test, y_pred))

print("\nCikarilan karar kurallari (ilk birkac dal):")
rules_text = export_text(model, feature_names=feature_cols)
print(rules_text[:1200])
with open("figures/decision_rules.txt", "w") as f:
    f.write(rules_text)

print("\nGorseller kaydediliyor...")

cm = confusion_matrix(y_test, y_pred, labels=["Dusuk", "Orta", "Yuksek"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["Dusuk", "Orta", "Yuksek"],
            yticklabels=["Dusuk", "Orta", "Yuksek"])
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Klinik Risk Siniflandirmasi")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=feature_cols, class_names=model.classes_,
          filled=True, rounded=True, fontsize=9, max_depth=3)
plt.title("Karar Agaci - Klinik Risk Siniflandirmasi (ilk 3 seviye)")
plt.tight_layout()
plt.savefig("figures/decision_tree.png", dpi=150)
plt.close()

importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="Importance", y="Feature",
            hue="Feature", palette="Purples_r", legend=False)
plt.title("Ozellik Onem Duzeyleri (Feature Importance)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
importance_df.to_csv("figures/feature_importance.csv", index=False)

print("Kaydedildi: figures/confusion_matrix.png")
print("Kaydedildi: figures/decision_tree.png")
print("Kaydedildi: figures/feature_importance.png")
print("Kaydedildi: figures/decision_rules.txt")

print("\nTamamlandi.")

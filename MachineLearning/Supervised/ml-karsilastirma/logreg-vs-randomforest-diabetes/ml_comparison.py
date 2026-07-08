"""
MODEL KARSILASTIRMA: Logistic Regression vs Random Forest (Diyabet Teshisi)
Amac: Ayni tibbi teshis problemi (diyabet var/yok) uzerinde iki farkli
      modeli - basit/dogrusal (Logistic Regression) ve karmasik/agac
      tabanli (Random Forest) - kiyaslamak. Sadece accuracy'e degil,
      Confusion Matrix, ROC-AUC ve ozellikle 5-Fold Cross-Validation'a
      bakarak "hangi model gercekten daha guvenilir" sorusuna cevap
      aramak.

Not: Pima Indians Diabetes veri seti (768 kayit, halka acik, cok bilinen
     bir tibbi veri seti) bu ortamda bulunmadigi icin, ayni kolon yapisini
     ve gercekci istatistiksel iliskileri (yuksek glikoz/BMI/yas -> yuksek
     diyabet riski) yansitan SENTETIK bir veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Model Karsilastirma - Veri Uretimi (Pima Indians Diabetes tarzi)")

n = 768

age = np.random.randint(21, 81, n)
pregnancies = np.clip(np.random.poisson(3, n), 0, 17)
glucose = np.clip(np.random.normal(120, 30, n), 44, 199)
blood_pressure = np.clip(np.random.normal(69, 19, n), 24, 122)
skin_thickness = np.clip(np.random.normal(20, 16, n), 0, 99)
insulin = np.clip(np.random.normal(80, 115, n), 0, 846)
bmi = np.clip(np.random.normal(32, 7.9, n), 18, 67)
diabetes_pedigree = np.round(np.clip(np.random.exponential(0.47, n), 0.08, 2.42), 3)

risk_score = (
    0.035 * (glucose - 120)
    + 0.06 * (bmi - 32)
    + 0.02 * (age - 33)
    + 0.4 * (pregnancies / 3)
    + 0.6 * diabetes_pedigree
    + np.random.normal(0, 1.4, n)
)
outcome = (risk_score > np.percentile(risk_score, 65)).astype(int)

df = pd.DataFrame({
    "Pregnancies": pregnancies, "Glucose": glucose.round(0),
    "BloodPressure": blood_pressure.round(0), "SkinThickness": skin_thickness.round(0),
    "Insulin": insulin.round(0), "BMI": bmi.round(1),
    "DiabetesPedigreeFunction": diabetes_pedigree, "Age": age, "Outcome": outcome
})
print(f"Veri seti boyutu: {df.shape}")
print(f"Sinif dagilimi:\n{df['Outcome'].value_counts().to_string()}")

print("\nMantiken 0 olamayacak kolonlardaki 0 degerleri NaN ile degistiriliyor...")
zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
missing_before = df[zero_columns].isnull().sum().sum()
print(f"Tespit edilen mantiksiz sifir/eksik deger: {missing_before}")

df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("\nLogistic Regression icin veriler standartlastiriliyor (scaling)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== MODEL 1: Logistic Regression (Basit/Dogrusal Model) ===")
log_model = LogisticRegression(random_state=RANDOM_STATE)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_pred)
print(f"Accuracy: {log_acc:.4f}")
print(classification_report(y_test, log_pred))

print("\n=== MODEL 2: Random Forest (Karmasik/Agac Tabanli Model) ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)   # agac tabanli modeller scaling gerektirmez
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

print("\nConfusion matrix'ler kaydediliyor...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
log_cm = confusion_matrix(y_test, log_pred)
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gercek Durum')

rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gercek Durum')
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

print("ROC-AUC egrisi kaydediliyor...")
plt.figure(figsize=(8, 6), dpi=100)
log_pos_probs = log_model.predict_proba(X_test_scaled)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, log_pos_probs)
roc_auc_log = auc(fpr_log, tpr_log)
plt.plot(fpr_log, tpr_log, color='blue', label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')

rf_pos_probs = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pos_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlis Alarm Orani)')
plt.ylabel('True Positive Rate (Dogru Teshis Orani)')
plt.title('ROC Egrisi Karsilastirmasi')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/roc_comparison.png", dpi=150)
plt.close()

print("\n5-Fold Cross-Validation hesaplaniyor...")
log_cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

print(f"Logistic Regression CV skorlari: {np.round(log_cv_scores, 3)}")
print(f"Logistic Regression CV ortalama: {log_cv_scores.mean():.4f} (std: {log_cv_scores.std():.4f})")
print(f"Random Forest CV skorlari: {np.round(rf_cv_scores, 3)}")
print(f"Random Forest CV ortalama: {rf_cv_scores.mean():.4f} (std: {rf_cv_scores.std():.4f})")

folds = ["1. Katlama", "2. Katlama", "3. Katlama", "4. Katlama", "5. Katlama"]
plt.figure(figsize=(10, 6))
plt.plot(folds, log_cv_scores, marker='o', linewidth=2, color='blue',
         label=f'Logistic Regression (Ort: {log_cv_scores.mean():.3f})')
plt.plot(folds, rf_cv_scores, marker='s', linewidth=2, color='green',
         label=f'Random Forest (Ort: {rf_cv_scores.mean():.3f})')
plt.title('Modellerin 5-Fold Cross-Validation Katlama Performanslari', fontsize=14, fontweight='bold')
plt.xlabel('Cross-Validation Katlamalari', fontsize=12)
plt.ylabel('Dogruluk Skoru (Accuracy)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig("figures/cross_validation_comparison.png", dpi=150)
plt.close()

print("\n=== SONUC OZETI ===")
summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Tek_Test_Accuracy": [log_acc, rf_acc],
    "CV_Ortalama_Accuracy": [log_cv_scores.mean(), rf_cv_scores.mean()],
    "CV_Std": [log_cv_scores.std(), rf_cv_scores.std()],
    "ROC_AUC": [roc_auc_log, roc_auc_rf],
})
print(summary.to_string(index=False))
summary.to_csv("figures/model_comparison_summary.csv", index=False)

if (rf_acc > log_acc) and (rf_cv_scores.mean() < log_cv_scores.mean()):
    print("\nCELISKI TESPIT EDILDI: Random Forest tek test setinde daha iyi,")
    print("ama CV ortalamasinda Logistic Regression daha iyi/yakin cikti.")
    print("Bu, tek train/test split'in sansa bagli olabilecegini, CV'nin")
    print("daha guvenilir bir olcut oldugunu gosteren tipik bir ornektir.")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

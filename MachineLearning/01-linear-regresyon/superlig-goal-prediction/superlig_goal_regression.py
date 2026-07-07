"""
SUPER LIG GOL TAHMINI - Basit ve Coklu Dogrusal Regresyon
Amac: Bir futbol takiminin sezon istatistiklerinden (isabetli sut, top
      hakimiyeti, pas isabet yuzdesi vb.) attigi gol sayisini tahmin etmek.
      Basit regresyonla tek degiskenin (isabetli sut) etkisi, coklu
      regresyonla birden fazla degiskenin birlikte etkisi karsilastirilir.

Not: Gercek Super Lig verisi bu ortamda bulunmadigi icin, gercekci futbol
     istatistikleri iliskilerini (daha fazla isabetli sut / top hakimiyeti
     -> daha fazla gol) yansitan SENTETIK bir veri seti uretilir. 18 takim
     x 15 sezon = 270 takim-sezon kaydi olusturulur.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Super Lig Gol Tahmini - Veri Uretimi")

n_teams = 18
n_seasons = 15
n = n_teams * n_seasons

isabetli_sut = np.clip(np.random.normal(140, 35, n), 50, 260)
toplam_sut = isabetli_sut + np.clip(np.random.normal(180, 40, n), 60, 320)
pas_isabet_yuzdesi = np.clip(np.random.normal(78, 6, n), 60, 92)
top_hakimiyeti = np.clip(np.random.normal(50, 8, n), 30, 72)
korner = np.clip(np.random.normal(120, 25, n), 50, 200)

atilan_gol = (
    0.32 * isabetli_sut
    + 0.05 * toplam_sut
    + 0.25 * (pas_isabet_yuzdesi - 78)
    + 0.4 * (top_hakimiyeti - 50)
    + 0.03 * korner
    + np.random.normal(0, 6, n)
)
atilan_gol = np.clip(atilan_gol, 15, None).round(0)

teams = [f"Takim_{i+1:02d}" for i in range(n_teams)]
takim_adi = np.tile(teams, n_seasons)

df = pd.DataFrame({
    "Takim_Adi": takim_adi,
    "Isabetli_Sut": isabetli_sut.round(0),
    "Toplam_Sut": toplam_sut.round(0),
    "Pas_Isabet_Yuzdesi": pas_isabet_yuzdesi.round(1),
    "Top_Hakimiyeti": top_hakimiyeti.round(1),
    "Korner": korner.round(0),
    "Atilan_Gol": atilan_gol
})
print(f"Uretilen kayit sayisi: {len(df)} (takim-sezon)")

print("\nKesifci veri analizi - korelasyon matrisi hesaplaniyor...")
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=150)
plt.close()

print("\nBasit Dogrusal Regresyon (tek degisken: Isabetli_Sut)")
x = df[["Isabetli_Sut"]].values
y = df["Atilan_Gol"].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

model_simple = LinearRegression()
model_simple.fit(x_train, y_train)
y_pred_test = model_simple.predict(x_test)

r2_simple = r2_score(y_test, y_pred_test)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_simple = mean_absolute_error(y_test, y_pred_test)
print(f"    Kesisim noktasi: {model_simple.intercept_:.3f}")
print(f"    Egim (Isabetli_Sut): {model_simple.coef_[0]:.3f}")
print(f"    R2: {r2_simple:.4f} | RMSE: {rmse_simple:.2f} | MAE: {mae_simple:.2f}")

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color="#2563eb", alpha=0.5, label="Gercek Veri", s=60)
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = model_simple.predict(x_line)
plt.plot(x_line, y_line, color="#dc2626", linewidth=2, label="Regresyon Dogrusu")
plt.xlabel("Isabetli Sut (sezon)")
plt.ylabel("Atilan Gol (sezon)")
plt.title("Basit Dogrusal Regresyon: Isabetli Sut -> Atilan Gol")
plt.legend()
plt.tight_layout()
plt.savefig("figures/simple_regression.png", dpi=150)
plt.close()

print("\nCoklu Dogrusal Regresyon (tum degiskenler)")
feature_cols = ["Isabetli_Sut", "Toplam_Sut", "Pas_Isabet_Yuzdesi", "Top_Hakimiyeti", "Korner"]
X_multi = df[feature_cols].values
y_multi = df["Atilan_Gol"].values

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=RANDOM_STATE
)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_test_m = model_multi.predict(X_test_m)

r2_multi = r2_score(y_test_m, y_pred_test_m)
rmse_multi = np.sqrt(mean_squared_error(y_test_m, y_pred_test_m))
mae_multi = mean_absolute_error(y_test_m, y_pred_test_m)
print(f"    R2: {r2_multi:.4f} | RMSE: {rmse_multi:.2f} | MAE: {mae_multi:.2f}")

coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model_multi.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nCoklu regresyon katsayilari:")
print(coef_df.to_string(index=False))
coef_df.to_csv("figures/coefficients.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=coef_df, x="Coefficient", y="Feature",
            hue="Feature", palette="Blues_r", legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Coklu Regresyon Katsayilari (Gole Etki)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 6))
plt.scatter(y_test_m, y_pred_test_m, alpha=0.5, color="#059669", s=60)
lims = [min(y_test_m.min(), y_pred_test_m.min()), max(y_test_m.max(), y_pred_test_m.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Gercek Gol")
plt.ylabel("Tahmin Edilen Gol")
plt.title("Coklu Regresyon: Gercek vs Tahmin")
plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", dpi=150)
plt.close()

print("\nKarsilastirma:")
print(f"    Basit regresyon  R2: {r2_simple:.4f}")
print(f"    Coklu regresyon  R2: {r2_multi:.4f}")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

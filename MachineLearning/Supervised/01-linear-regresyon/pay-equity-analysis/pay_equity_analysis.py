"""
UCRET ESITLIGI ANALIZI (Pay Equity Analysis) - Linear Regression
Amac: Bir sirkette cinsiyete gore HAM maas farkinin ne kadarinin mesru
      faktorlerle (deneyim, pozisyon, departman, egitim, performans)
      ACIKLANABILDIGINI, ne kadarinin ACIKLANAMADIGINI (potansiyel
      esitsizlik sinyali) tespit etmek.

      Bu tur analizler HR/compliance ekipleri tarafindan sirketin kendi
      ucret yapisini denetlemek icin kullanilir (ozellikle AB Pay
      Transparency Directive gibi duzenlemelerle onemi artmistir).

ONEMLI ETIK CERCEVE: Bu proje cinsiyet ayrimciligini SAVUNMAZ, tam tersine
      ONA KARSI bir denetim/seffaflik araci olarak kurulmustur. Amac
      esitsizligi normallestirmek degil, varsa ORTAYA CIKARMAKTIR.

Not: Gercek bir sirket verisi kullanilmadigi icin, literatuerde bilinen
     "ham fark var ama kontrol edilince küçülüyor / küçülmuyor" oruntu
     ussune sadik SENTETIK bir veri seti uretilir.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)


# 1) SENTETIK CALISAN/MAAS VERISI URETIMI

print("UCRET ESITLIGI ANALIZI - VERI URETIMI")
print("=" * 60)

n = 3000
gender = np.random.choice(["Kadin", "Erkek"], n, p=[0.48, 0.52])
experience_years = np.clip(np.random.normal(8, 5, n), 0, 35)
department = np.random.choice(
    ["Muhendislik", "Satis", "Pazarlama", "Operasyon", "Finans"], n,
    p=[0.35, 0.2, 0.15, 0.15, 0.15]
)
education = np.random.choice(["Lisans", "Yuksek Lisans", "Doktora"], n, p=[0.55, 0.35, 0.10])
performance_score = np.clip(np.random.normal(75, 12, n), 30, 100)

dept_base = {
    "Muhendislik": 45000, "Satis": 38000, "Pazarlama": 36000,
    "Operasyon": 34000, "Finans": 40000
}
edu_bonus = {"Lisans": 0, "Yuksek Lisans": 6000, "Doktora": 14000}

base_salary = np.array([dept_base[d] for d in department])
edu_add = np.array([edu_bonus[e] for e in education])

# Mesru faktorlerin maasa etkisi
legitimate_salary = (
    base_salary
    + edu_add
    + experience_years * 1800
    + (performance_score - 75) * 300
)

# KASITLI OLARAK eklenen, "aciklanamayan" cinsiyet etkisi
# (bu satir gercek hayatta ORTAYA CIKARILMASI gereken sorunu temsil eder)
unexplained_gender_penalty = np.where(gender == "Kadin", -4200, 0)

noise = np.random.normal(0, 3500, n)
salary = legitimate_salary + unexplained_gender_penalty + noise
salary = np.clip(salary, 28000, None).round(0)

df = pd.DataFrame({
    "gender": gender,
    "experience_years": experience_years.round(1),
    "department": department,
    "education": education,
    "performance_score": performance_score.round(1),
    "salary": salary
})
print(f"Uretilen kayit sayisi: {len(df)}")

raw_gap = df.groupby("gender")["salary"].mean()
raw_gap_pct = (1 - raw_gap["Kadin"] / raw_gap["Erkek"]) * 100
print(f"Ham ortalama maas (Erkek) : {raw_gap['Erkek']:.0f}")
print(f"Ham ortalama maas (Kadin) : {raw_gap['Kadin']:.0f}")
print(f"HAM (kontrolsuz) fark     : %{raw_gap_pct:.1f}")


# 2) ON ISLEME + MODEL (kontrol degiskenleriyle)

print("\n[2] Kategorik degiskenler encode ediliyor...")
df["gender_enc"] = (df["gender"] == "Kadin").astype(int)  # 1=Kadin, 0=Erkek

le_dept = LabelEncoder()
df["department_enc"] = le_dept.fit_transform(df["department"])
le_edu = LabelEncoder()
df["education_enc"] = le_edu.fit_transform(df["education"])

feature_cols = [
    "experience_years", "performance_score",
    "department_enc", "education_enc", "gender_enc"
]
X = df[feature_cols]
y = df["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print("\n[3] Linear Regression egitiliyor (kontrol degiskenleriyle)...")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"    R2 : {r2:.4f}")
print(f"    MAE: {mae:.0f}")

gender_coef = model.coef_[feature_cols.index("gender_enc")]
print(f"\n>>> KONTROL EDILMIS (aciklanamayan) cinsiyet etkisi: {gender_coef:.0f} TL")
print(f"    (Deneyim, departman, egitim, performans SABIT tutulduğunda,")
print(f"     kadin calisanlarin maasi erkeklere gore ortalama {gender_coef:.0f} TL farkli.)")

adjusted_gap_pct = abs(gender_coef) / raw_gap["Erkek"] * 100


# 3) GORSELLESTIRME

print("\n[4] Gorseller kaydediliyor...")

# --- Ham vs Kontrol Edilmis Fark ---
plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["Ham Fark\n(kontrolsuz)", "Kontrol Edilmis Fark\n(aciklanamayan)"],
    [raw_gap_pct, adjusted_gap_pct],
    color=["#f59e0b", "#dc2626"]
)
for b in bars:
    plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
              f"%{b.get_height():.1f}", ha="center", fontweight="bold")
plt.ylabel("Maas Farki (%)")
plt.title("Ham Fark vs Kontrol Edilmis (Aciklanamayan) Fark")
plt.tight_layout()
plt.savefig("figures/raw_vs_adjusted_gap.png", dpi=150)
plt.close()

# --- Departmana gore ortalama maas, cinsiyete gore ---
plt.figure(figsize=(9, 5))
dept_gender = df.groupby(["department", "gender"])["salary"].mean().unstack()
dept_gender.plot(kind="bar", ax=plt.gca(), color=["#f59e0b", "#2563eb"])
plt.ylabel("Ortalama Maas (TL)")
plt.xlabel("Departman")
plt.title("Departmana ve Cinsiyete Gore Ortalama Maas")
plt.xticks(rotation=20)
plt.legend(title="Cinsiyet")
plt.tight_layout()
plt.savefig("figures/salary_by_department_gender.png", dpi=150)
plt.close()

# Deneyim vs Maas, cinsiyete gore renklendirilmis 
plt.figure(figsize=(8, 6))
for g, c in [("Erkek", "#2563eb"), ("Kadin", "#f59e0b")]:
    subset = df[df["gender"] == g]
    plt.scatter(subset["experience_years"], subset["salary"],
                alpha=0.35, label=g, color=c, s=18)
plt.xlabel("Deneyim (Yil)")
plt.ylabel("Maas (TL)")
plt.title("Deneyime Gore Maas Dagilimi (Cinsiyete Gore)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/experience_vs_salary.png", dpi=150)
plt.close()

#  Model katsayilari 
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
coef_df["Feature"] = coef_df["Feature"].replace({"gender_enc": "gender_enc (Kadin=1)"})

plt.figure(figsize=(8, 5))
colors = ["#dc2626" if f.startswith("gender") else "#6366f1" for f in coef_df["Feature"]]
sns.barplot(data=coef_df, x="Coefficient", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Regresyon Katsayilari (Maasa Etki, TL)")
plt.tight_layout()
plt.savefig("figures/coefficients.png", dpi=150)
plt.close()
coef_df.to_csv("figures/coefficients.csv", index=False)

print("    Kaydedildi: figures/raw_vs_adjusted_gap.png")
print("    Kaydedildi: figures/salary_by_department_gender.png")
print("    Kaydedildi: figures/experience_vs_salary.png")
print("    Kaydedildi: figures/coefficients.png")

print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)

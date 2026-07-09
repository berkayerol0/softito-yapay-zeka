"""
SIGORTA PAZAR SEGMENTASYONU - K-Means, Hiyerarsik, DBSCAN, GMM Kiyaslamasi
Amac: Sigorta musterilerini davranissal ozelliklerine gore segmentlere
      ayirmak - ama bunu TEK bir algoritmayla degil, 4 FARKLI kumeleme
      algoritmasiyla yapip sonuclari karsilastirmak. Her algoritmanin
      farkli bir varsayimi vardir (K-Means kuresel kumeler bekler, DBSCAN
      yogunluk bazli calisir, GMM olasiliksal/yumusak kumeleme yapar) -
      bu proje "hangi algoritma bu veri icin en uygun" sorusuna veriyle
      cevap arar.

Not: Orijinal not defteri kagglehub uzerinden gercek bir sigorta pazar
     segmentasyonu veri seti indiriyordu; bu Kaggle API credential'i
     gerektirdigi icin, benzer istatistiksel ozellikte SENTETIK bir
     sigorta musteri veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Sigorta Pazar Segmentasyonu - Veri Uretimi")

n = 1200
segment_choice = np.random.choice(
    ["genc_dusuk_risk", "orta_yas_aile", "yuksek_gelir_dusuk_talep", "yuksek_risk"],
    n, p=[0.3, 0.3, 0.2, 0.2]
)

def by_segment(seg, mapping):
    return np.array([mapping[s] for s in seg])

age_mean = by_segment(segment_choice, {
    "genc_dusuk_risk": 27, "orta_yas_aile": 42,
    "yuksek_gelir_dusuk_talep": 50, "yuksek_risk": 45
})
income_mean = by_segment(segment_choice, {
    "genc_dusuk_risk": 32000, "orta_yas_aile": 55000,
    "yuksek_gelir_dusuk_talep": 110000, "yuksek_risk": 40000
})
claims_mean = by_segment(segment_choice, {
    "genc_dusuk_risk": 0.4, "orta_yas_aile": 1.2,
    "yuksek_gelir_dusuk_talep": 0.3, "yuksek_risk": 3.5
})
policies_mean = by_segment(segment_choice, {
    "genc_dusuk_risk": 1.2, "orta_yas_aile": 2.8,
    "yuksek_gelir_dusuk_talep": 3.5, "yuksek_risk": 1.8
})

age = np.clip(np.random.normal(age_mean, 6, n), 18, 80)
income = np.clip(np.random.normal(income_mean, income_mean * 0.25, n), 15000, None)
claims_filed = np.clip(np.random.poisson(claims_mean), 0, None)
policy_count = np.clip(np.random.normal(policies_mean, 0.8, n), 1, 8)

# Kasitli olarak birkac eksik deger ekle (gercek hayatta oldugu gibi)
missing_idx = np.random.choice(n, size=int(n * 0.03), replace=False)
income_with_na = income.copy()
income_with_na[missing_idx] = np.nan

df = pd.DataFrame({
    "CustomerID": [f"POL-{i:05d}" for i in range(n)],
    "Age": age.round(0),
    "Annual_Income": income_with_na.round(0),
    "Claims_Filed": claims_filed,
    "Policy_Count": policy_count.round(1)
})
print(f"Uretilen musteri sayisi: {len(df)}")
print(f"\nVeri seti bilgisi:")
print(f"Sutun sayisi: {len(df.columns)}")
print(f"Satir sayisi: {len(df)}")

print("\nIstatistiksel ozet:")
print(df.describe().T.round(2).to_string())

print("\nEksik deger raporu:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
print(f"Toplam eksik deger: {df.isnull().sum().sum()}")

feature_cols = ["Age", "Annual_Income", "Claims_Filed", "Policy_Count"]
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(df[feature_cols])
df_imputed = pd.DataFrame(X_imputed, columns=feature_cols)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df_imputed)

print("\n--- K-MEANS ---")
inertia, silhouette_scores = [], []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(x_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertia, "bo-")
axes[0].set_title("Elbow Yontemi")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), silhouette_scores, "ro-")
axes[1].set_title("Silhouette Skoru")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Skor")
plt.tight_layout()
plt.savefig("figures/optimal_k.png", dpi=150)
plt.close()

optimal_k = 4   # veri uretimindeki 4 gizli segmente denk gelecek sekilde secildi
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(x_scaled)
kmeans_sil = silhouette_score(x_scaled, kmeans_labels)
print(f"K-Means (K={optimal_k}) Silhouette Skoru: {kmeans_sil:.4f}")

print("\n--- HIYERARSIK (AGGLOMERATIVE) ---")
agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
agg_labels = agg.fit_predict(x_scaled)
agg_sil = silhouette_score(x_scaled, agg_labels)
print(f"Agglomerative (K={optimal_k}) Silhouette Skoru: {agg_sil:.4f}")

print("\n--- DBSCAN ---")
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(x_scaled)
distances, _ = neighbors_fit.kneighbors(x_scaled)
distances = np.sort(distances[:, 4], axis=0)

plt.figure(figsize=(9, 5))
plt.plot(distances, "b-")
plt.xlabel("Nokta Sirasi")
plt.ylabel("5. En Yakin Komsu Mesafesi")
plt.title("DBSCAN - Eps Secimi icin K-Distance Grafigi")
plt.tight_layout()
plt.savefig("figures/dbscan_kdistance.png", dpi=150)
plt.close()

dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(x_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"DBSCAN kume sayisi: {n_clusters_db}, gurultu noktasi: {n_noise}")
if n_clusters_db > 1:
    mask = dbscan_labels != -1
    dbscan_sil = silhouette_score(x_scaled[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette Skoru (gurultu haric): {dbscan_sil:.4f}")
else:
    dbscan_sil = np.nan
    print("DBSCAN tek kume/gurultu uretti, silhouette hesaplanamadi.")

print("\n--- GAUSSIAN MIXTURE MODEL ---")
gmm = GaussianMixture(n_components=optimal_k, covariance_type="full",
                       max_iter=200, random_state=RANDOM_STATE)
gmm_labels = gmm.fit_predict(x_scaled)
gmm_sil = silhouette_score(x_scaled, gmm_labels)
print(f"GMM (K={optimal_k}) Silhouette Skoru: {gmm_sil:.4f}")

print("\n--- ALGORITMA KARSILASTIRMASI ---")
comparison = pd.DataFrame({
    "Algoritma": ["K-Means", "Hiyerarsik (Agglomerative)", "DBSCAN", "Gaussian Mixture Model"],
    "Silhouette_Skoru": [kmeans_sil, agg_sil, dbscan_sil, gmm_sil],
    "Kume_Sayisi": [optimal_k, optimal_k, n_clusters_db, optimal_k]
})
print(comparison.to_string(index=False))
comparison.to_csv("figures/algorithm_comparison.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=comparison, x="Silhouette_Skoru", y="Algoritma",
            hue="Algoritma", palette="viridis", legend=False)
plt.title("Algoritma Karsilastirmasi (Silhouette Skoru)")
plt.tight_layout()
plt.savefig("figures/algorithm_comparison.png", dpi=150)
plt.close()

print("\nPCA ile gorsellestirme (tum algoritmalar)...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(x_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
configs = [
    (kmeans_labels, "K-Means", "viridis", axes[0, 0]),
    (agg_labels, "Hiyerarsik (Agglomerative)", "plasma", axes[0, 1]),
    (dbscan_labels, "DBSCAN (-1 = Gurultu)", "tab10", axes[1, 0]),
    (gmm_labels, "Gaussian Mixture Model", "coolwarm", axes[1, 1]),
]
for labels, title, cmap, ax in configs:
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap,
                     alpha=0.7, s=35, edgecolors="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Kume")
plt.tight_layout()
plt.savefig("figures/all_algorithms_pca.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

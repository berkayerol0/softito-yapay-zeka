"""
E-TICARET MUSTERI SEGMENTASYONU - K-Means + PCA
Amac: Musterileri satin alma davranislarina (ortalama miktar, ortalama
      birim fiyat, toplam harcama, siparis sikligi) gore anlamli
      segmentlere ayirmak. Optimal kume sayisi Elbow + Silhouette
      yontemiyle veriden secilir, sonuc PCA ile 2 boyuta indirgenip
      gorsellestirilir.

Not: Orijinal not defteri kagglehub uzerinden gercek bir e-ticaret veri
     seti (carrie1/ecommerce-data) indiriyordu; bu Kaggle API credential'i
     gerektirdigi ve ortam disina bagimlilik yarattigi icin, benzer
     istatistiksel ozellikte SENTETIK bir musteri veri seti uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("E-ticaret Musteri Segmentasyonu - Veri Uretimi")

n_customers = 1500

# 3 dogal musteri "arketipi" karistirarak gercekci kumeler olustur
segment_choice = np.random.choice(["dusuk_hacim", "orta_hacim", "vip"], n_customers, p=[0.5, 0.35, 0.15])

def gen(seg, low, mid, vip):
    return np.where(seg == "dusuk_hacim", low,
           np.where(seg == "orta_hacim", mid, vip))

avg_quantity = np.array([
    np.random.normal(gen(s, 3, 8, 18), 2) for s in segment_choice
]).clip(1, None)

avg_unit_price = np.array([
    np.random.normal(gen(s, 8, 15, 35), 5) for s in segment_choice
]).clip(1, None)

order_count = np.array([
    np.random.poisson(gen(s, 2, 6, 15)) for s in segment_choice
]).clip(1, None)

total_spent = avg_quantity * avg_unit_price * order_count

df = pd.DataFrame({
    "CustomerID": [f"CUST-{i:05d}" for i in range(n_customers)],
    "Avg_Quantity": avg_quantity.round(2),
    "Avg_UnitPrice": avg_unit_price.round(2),
    "Order_Count": order_count,
    "Total_Spent": total_spent.round(2)
})
print(f"Uretilen musteri sayisi: {len(df)}")
print(df.describe().round(2).to_string())

feature_cols = ["Avg_Quantity", "Avg_UnitPrice", "Order_Count"]
profile_cols = feature_cols + ["Total_Spent"]

# Total_Spent, diger 3 degiskenin carpimi oldugu icin kumelemeye DAHIL EDILMEZ
# (dahil edilirse tek bir asiri carpik degisken mesafe hesabina hakim olur,
# kumeleme yapisini bozar). Log-donusum carpik dagilimlari normallestirir.
X = df[feature_cols].apply(lambda col: np.log1p(col))

print("\nEksik deger kontrolu...")
print(f"Eksik deger: {X.isnull().sum().sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nOptimal kume sayisi araniyor (Elbow + Silhouette)...")
inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"    K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_range), inertias, "bo-")
axes[0].set_title("Elbow Yontemi")
axes[0].set_xlabel("Kume Sayisi (K)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(K_range), silhouettes, "ro-")
axes[1].set_title("Silhouette Skoru")
axes[1].set_xlabel("Kume Sayisi (K)")
axes[1].set_ylabel("Skor")
plt.tight_layout()
plt.savefig("figures/optimal_k.png", dpi=150)
plt.close()

optimal_k_statistical = list(K_range)[int(np.argmax(silhouettes))]
print(f"\nIstatistiksel optimal K (Silhouette'e gore): {optimal_k_statistical}")

# Silhouette skoru istatistiksel olarak K=2'yi one cikarir, ancak bu sadece
# "az harcayan / cok harcayan" gibi kaba, is acisindan az aksiyon alinabilir
# bir ayrim uretir. Is baglaminda (dusuk/orta/VIP musteri katmanlari) K=3
# daha anlamli ve pazarlama ekibinin kullanabilecegi bir segmentasyon sunar.
# Bu, istatistiksel metrigin degil is ihtiyacinin K secimine yon verdigi
# bilincli bir karardir.
optimal_k = 3
print(f"Secilen K (is baglami nedeniyle): {optimal_k}")
print("Gerekce: K=2 sadece kaba bir 'az/cok harcayan' ayrimi verirken, K=3")
print("pazarlama ekibinin dogrudan aksiyon alabilecegi (dusuk/orta/VIP)")
print("katmanlari ortaya cikarir.")

print(f"\nK-Means egitiliyor (K={optimal_k})...")
kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Skoru: {final_silhouette:.4f}")

df["KMeans_Cluster"] = kmeans_labels
cluster_summary = df.groupby("KMeans_Cluster")[profile_cols].mean().round(2)
cluster_summary["Musteri_Sayisi"] = df["KMeans_Cluster"].value_counts().sort_index()
print("\nKume Profilleri:")
print(cluster_summary.to_string())
cluster_summary.to_csv("figures/cluster_profiles.csv")

print("\nPCA ile 2 boyutlu gorsellestirme...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"Aciklanan varyans orani: {pca.explained_variance_ratio_.sum():.3f}")

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis",
                       alpha=0.7, s=40, edgecolors="k", linewidth=0.3)
plt.colorbar(scatter, label="Kume")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"K-Means Kumeleme Sonuclari (PCA ile, K={optimal_k})")
plt.tight_layout()
plt.savefig("figures/kmeans_pca.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
for c in sorted(df["KMeans_Cluster"].unique()):
    subset = df[df["KMeans_Cluster"] == c]
    plt.scatter(subset["Order_Count"], subset["Total_Spent"], alpha=0.5, s=40, label=f"Kume {c}")
plt.xlabel("Siparis Sayisi")
plt.ylabel("Toplam Harcama")
plt.title("Kumelere Gore Siparis Sayisi vs Toplam Harcama")
plt.legend()
plt.tight_layout()
plt.savefig("figures/order_vs_spending.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# ============================ CONFIG ============================
DATA_PATH = 'insurance.csv'                 # None yaparsan klasorde otomatik aranir
CLUSTER_FEATURES = ['age', 'bmi', 'charges']  # kumeleme bu sayisal kolonlara gore
RANDOM_STATE = 42
# ===============================================================

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("INSURANCE - CLUSTERING (KMEANS)")
print("=" * 60)

# ----------------------------------------------------------------
print("\n[1] Loading data...")
if DATA_PATH is None:
    cands = [c for c in glob.glob('*.csv') + glob.glob('**/*.csv', recursive=True)
             if not c.startswith('output')]
    if not cands:
        raise FileNotFoundError("Klasorde .csv bulunamadi.")
    DATA_PATH = cands[0]
    print(f"    Otomatik bulunan dosya: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")

# ----------------------------------------------------------------
print("\n[2] Cleaning data...")
df = df.dropna(how='all').dropna(axis=1, how='all')

missing = df[CLUSTER_FEATURES].isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    print("    Eksik deger iceren kolonlar:")
    for col, n in missing.items():
        print(f"        {col}: {n}")
    for col in CLUSTER_FEATURES:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
else:
    print("    Kumeleme kolonlarinda eksik deger yok.")

dups = df.duplicated().sum()
if dups > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"    {dups} yinelenen satir atildi.")
print(f"    Temizlik sonrasi shape: {df.shape}")

# ----------------------------------------------------------------
print("\n[3] Scaling features...")
print(f"    Kullanilan ozellikler: {CLUSTER_FEATURES}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])

# ----------------------------------------------------------------
print("\n[4] Finding optimal K (Elbow + Silhouette)...")
inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouettes.append(sil)
    print(f"    K={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('Elbow Method'); axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')
axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('K'); axes[1].set_ylabel('Score')
plt.tight_layout()
plt.savefig('output/clu_optimal_k.png', dpi=150)
plt.close()
print("    Saved: output/clu_optimal_k.png")

best_k = list(K_range)[int(np.argmax(silhouettes))]
print(f"\n    Best K (Silhouette): {best_k}")

# ----------------------------------------------------------------
print(f"\n[5] Running K-Means with K={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------------------------------------------
print("\n[6] Cluster profiles:")
agg = {f: 'mean' for f in CLUSTER_FEATURES}
profile = df.groupby('Cluster').agg(Count=('Cluster', 'size'), **{f + '_mean': (f, 'mean') for f in CLUSTER_FEATURES})
profile = profile.round(2)
profile['Pct'] = (profile['Count'] / profile['Count'].sum() * 100).round(1)
print(profile.to_string())

# ----------------------------------------------------------------
print("\n[7] Boxplots by cluster...")
fig, axes = plt.subplots(1, len(CLUSTER_FEATURES), figsize=(6 * len(CLUSTER_FEATURES), 5))
if len(CLUSTER_FEATURES) == 1:
    axes = [axes]
for ax, feat in zip(axes, CLUSTER_FEATURES):
    sns.boxplot(data=df, x='Cluster', y=feat, ax=ax)
    ax.set_title(f'{feat} by Cluster')
plt.tight_layout()
plt.savefig('output/clu_profiles.png', dpi=150)
plt.close()
print("    Saved: output/clu_profiles.png")

# ----------------------------------------------------------------
print("\n[8] Segment labels...")
segment_names = {}
charges_med = profile['charges_mean'].median() if 'charges_mean' in profile else None
age_med = profile['age_mean'].median() if 'age_mean' in profile else None
for c in sorted(df['Cluster'].unique()):
    row = profile.loc[c]
    if charges_med is not None and age_med is not None:
        if row['charges_mean'] > charges_med and row['age_mean'] > age_med:
            label = "Yasli - Yuksek Masraf"
        elif row['charges_mean'] > charges_med:
            label = "Genc - Yuksek Masraf (riskli)"
        elif row['age_mean'] > age_med:
            label = "Yasli - Dusuk Masraf"
        else:
            label = "Genc - Dusuk Masraf"
    else:
        label = f"Cluster {c}"
    segment_names[c] = label
    print(f"    Cluster {c}: {label} ({int(row['Count'])} kisi, {row['Pct']}%)")

df['Segment'] = df['Cluster'].map(segment_names)

# ----------------------------------------------------------------
print("\n[9] Saving results...")
df.to_csv('output/clu_segments_labeled.csv', index=False)
print("    Saved: output/clu_segments_labeled.csv")

print("\n" + "=" * 60)
print("CLUSTERING COMPLETED SUCCESSFULLY")
print("=" * 60)
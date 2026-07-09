"""
URUN ONERI SISTEMI - K-Nearest Neighbors (KNN) ile Item-Based Collaborative Filtering
Amac: "Bu urunu begenen kullanicilar, su urunleri de begendi" mantigiyla
      calisan bir oneri sistemi kurmak. Item-based collaborative filtering,
      urunler arasindaki benzerligi kullanici davranis kaliplarindan (puan/
      satin alma) ogrenir, KNN ile her urune en yakin (en benzer) K urunu
      bulur. Bu, Amazon/Netflix tarzi "bunu alanlar sunu da aldi" bolumlerinin
      temelini olusturan klasik bir yontemdir.

Not: Gercek bir e-ticaret kullanici-urun etkilesim verisi bu ortamda
     bulunmadigi icin, gercekci kullanici tercih oruntulerini (benzer
     kategori urunlere benzer puanlar verilmesi) yansitan SENTETIK bir
     kullanici-urun puanlama matrisi uretilir.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Urun Oneri Sistemi (KNN) - Veri Uretimi")

n_users = 700
n_products = 60
categories = ["Elektronik", "Giyim", "Kitap", "Ev-Yasam", "Spor", "Kozmetik"]
products_per_cat = n_products // len(categories)

product_ids = [f"P{i:03d}" for i in range(n_products)]
product_category = np.repeat(categories, products_per_cat)
product_df = pd.DataFrame({"product_id": product_ids, "category": product_category})

# Her kullanicinin 1-3 favori kategorisi olsun (gercekci tercih oruntusu)
user_fav_categories = {
    u: np.random.choice(categories, size=np.random.randint(1, 3), replace=False)
    for u in range(n_users)
}

records = []
for u in range(n_users):
    fav_cats = user_fav_categories[u]
    n_ratings = np.random.randint(15, 35)
    rated_products = np.random.choice(product_ids, size=n_ratings, replace=False)
    for p in rated_products:
        cat = product_df.loc[product_df["product_id"] == p, "category"].values[0]
        if cat in fav_cats:
            rating = np.clip(np.random.normal(4.6, 0.4), 1, 5)
        else:
            rating = np.clip(np.random.normal(2.0, 0.8), 1, 5)
        records.append((u, p, round(rating)))

ratings_df = pd.DataFrame(records, columns=["user_id", "product_id", "rating"])
print(f"Toplam puanlama sayisi: {len(ratings_df)}")
print(f"Kullanici sayisi: {n_users} | Urun sayisi: {n_products}")

print("\nKullanici-urun puanlama matrisi olusturuluyor...")
user_item_matrix = ratings_df.pivot_table(
    index="user_id", columns="product_id", values="rating", fill_value=0
)
print(f"Matris boyutu: {user_item_matrix.shape}")

item_user_matrix = user_item_matrix.T  # KNN urunler arasi benzerlik icin transpoze

print("\nKNN modeli (item-based, cosine benzerligi) egitiliyor...")
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
knn_model.fit(item_user_matrix.values)


def get_similar_products(product_id, k=5):
    idx = item_user_matrix.index.get_loc(product_id)
    distances, indices = knn_model.kneighbors(
        item_user_matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=k + 1
    )
    similar = []
    for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
        similar.append((item_user_matrix.index[i], round(1 - dist, 3)))
    return similar


print("\nOrnek oneri: P000 urunune en benzer 5 urun")
example_recs = get_similar_products("P000", k=5)
for pid, sim in example_recs:
    cat = product_df.loc[product_df["product_id"] == pid, "category"].values[0]
    print(f"    {pid} ({cat}) - benzerlik: {sim}")

print("\nOneri kalitesi degerlendiriliyor: onerilen urunler ayni kategoriden mi?")
correct = 0
total = 0
for pid in item_user_matrix.index[:40]:
    true_cat = product_df.loc[product_df["product_id"] == pid, "category"].values[0]
    recs = get_similar_products(pid, k=5)
    for rec_pid, _ in recs:
        rec_cat = product_df.loc[product_df["product_id"] == rec_pid, "category"].values[0]
        total += 1
        if rec_cat == true_cat:
            correct += 1
category_precision = correct / total
print(f"Kategori tutarliligi (Precision@5): %{100*category_precision:.1f}")
print("(Yuksek oran, modelin ayni kategori urunleri dogru sekilde birbirine")
print(" yakin buldugunu gosterir - kullanici davranisindan ogrendigi icin")
print(" kategori bilgisini hic gormeden bunu basarir.)")

print("\nKullanici-bazli puan tahmini (user-based KNN) test ediliyor...")
train_ratings, test_ratings = train_test_split(
    ratings_df, test_size=0.2, random_state=RANDOM_STATE
)
train_matrix = train_ratings.pivot_table(
    index="user_id", columns="product_id", values="rating", fill_value=0
)

user_knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
user_knn.fit(train_matrix.values)

predictions, actuals = [], []
for _, row in test_ratings.iterrows():
    u, p, actual_rating = row["user_id"], row["product_id"], row["rating"]
    if u not in train_matrix.index or p not in train_matrix.columns:
        continue
    idx = train_matrix.index.get_loc(u)
    distances, indices = user_knn.kneighbors(
        train_matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=11
    )
    neighbor_ratings = []
    for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
        neighbor_id = train_matrix.index[i]
        r = train_matrix.loc[neighbor_id, p]
        if r > 0:
            neighbor_ratings.append(r)
    if neighbor_ratings:
        pred = np.mean(neighbor_ratings)
        predictions.append(pred)
        actuals.append(actual_rating)

mae = mean_absolute_error(actuals, predictions)
print(f"Test seti Puan Tahmini MAE: {mae:.3f} (1-5 puan olceginde)")

print("\nGorseller kaydediliyor...")

plt.figure(figsize=(10, 7))
sample_matrix = user_item_matrix.iloc[:40, :30]
sns.heatmap(sample_matrix, cmap="YlOrRd", cbar_kws={"label": "Puan (0=puanlanmadi)"})
plt.title("Kullanici-Urun Puanlama Matrisi (ornek 40x30 kesit)")
plt.xlabel("Urun")
plt.ylabel("Kullanici")
plt.tight_layout()
plt.savefig("figures/rating_matrix_heatmap.png", dpi=150)
plt.close()

cat_counts = product_df["category"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=cat_counts.index, y=cat_counts.values, hue=cat_counts.index,
            palette="viridis", legend=False)
plt.ylabel("Urun Sayisi")
plt.title("Kategori Basina Urun Sayisi")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/category_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5))
plt.bar(["Kategori Tutarliligi\n(Precision@5)"], [category_precision * 100], color="#0ea5e9")
plt.ylim(0, 100)
plt.ylabel("Yuzde (%)")
plt.title("Oneri Sistemi Kalite Metrigi")
plt.text(0, category_precision * 100 + 2, f"%{100*category_precision:.1f}",
          ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/recommendation_quality.png", dpi=150)
plt.close()

example_df = pd.DataFrame(example_recs, columns=["product_id", "similarity"])
example_df["category"] = example_df["product_id"].map(
    product_df.set_index("product_id")["category"]
)
example_df.to_csv("figures/example_recommendations_P000.csv", index=False)

print("Kaydedildi: figures/rating_matrix_heatmap.png")
print("Kaydedildi: figures/category_distribution.png")
print("Kaydedildi: figures/recommendation_quality.png")
print("Kaydedildi: figures/example_recommendations_P000.csv")

print("\nTamamlandi.")

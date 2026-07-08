"""
MOBIL CIHAZ OZELLIKLERINE GORE FIYAT SINIFLANDIRMASI - Decision Tree
Amac: Mobil cihazlarin teknik donanim ozelliklerine (RAM, batarya, kamera,
      ekran cozunurlugu vb.) bakarak hangi fiyat segmentinde (Dusuk, Orta,
      Yuksek, Cok Yuksek) yer alacagini tahmin etmek. Decision Tree,
      sonucu "hangi ozellik esik degerine gore karar verildigini" acikca
      gosterdigi icin tercih edilmistir.

Not: Orijinal not defteri Kaggle'in unlu "Mobile Price Classification"
     veri setini (train.csv, 2000 telefon, 20 ozellik) kullaniyordu. Bu
     dosya bu ortamda bulunmadigi icin, ayni kolon yapisini ve gercekci
     ozellik-fiyat iliskilerini (yuksek RAM/batarya/kamera -> yuksek fiyat)
     yansitan SENTETIK bir veri seti uretilir.
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

print("Mobil Fiyat Siniflandirmasi - Veri Uretimi")

n = 2000

battery_power = np.random.randint(500, 2000, n)
blue = np.random.choice([0, 1], n)
clock_speed = np.round(np.random.uniform(0.5, 3.0, n), 1)
dual_sim = np.random.choice([0, 1], n)
fc = np.random.randint(0, 20, n)
four_g = np.random.choice([0, 1], n)
int_memory = np.random.randint(2, 65, n)
m_dep = np.round(np.random.uniform(0.1, 1.0, n), 1)
mobile_wt = np.random.randint(80, 200, n)
n_cores = np.random.randint(1, 9, n)
pc = np.random.randint(0, 20, n)
px_height = np.random.randint(0, 1960, n)
px_width = np.random.randint(500, 2000, n)
ram = np.random.randint(256, 4000, n)
sc_h = np.random.randint(5, 19, n)
sc_w = np.random.randint(0, 18, n)
talk_time = np.random.randint(2, 20, n)
three_g = np.random.choice([0, 1], n)
touch_screen = np.random.choice([0, 1], n)
wifi = np.random.choice([0, 1], n)

# Gercek veri setindeki bilinen en guclu iliski: RAM, batarya, piksel
# cozunurlugu ve dahili hafiza fiyat segmentini belirleyen ana etkenler.
price_score = (
    0.55 * (ram / ram.max())
    + 0.15 * (battery_power / battery_power.max())
    + 0.10 * ((px_height * px_width) / (px_height.max() * px_width.max()))
    + 0.10 * (int_memory / int_memory.max())
    + 0.05 * (pc / pc.max())
    + 0.05 * (n_cores / n_cores.max())
    + np.random.normal(0, 0.03, n)
)
price_range = pd.qcut(price_score, q=4, labels=[0, 1, 2, 3]).astype(int)

df = pd.DataFrame({
    "battery_power": battery_power, "blue": blue, "clock_speed": clock_speed,
    "dual_sim": dual_sim, "fc": fc, "four_g": four_g, "int_memory": int_memory,
    "m_dep": m_dep, "mobile_wt": mobile_wt, "n_cores": n_cores, "pc": pc,
    "px_height": px_height, "px_width": px_width, "ram": ram, "sc_h": sc_h,
    "sc_w": sc_w, "talk_time": talk_time, "three_g": three_g,
    "touch_screen": touch_screen, "wifi": wifi, "price_range": price_range
})

print(f"Veri seti boyutu: {df.shape}")
print(f"Sinif dagilimi:\n{df['price_range'].value_counts().sort_index()}")

X = df.drop("price_range", axis=1)
y = df["price_range"]
print(f"\nOzellik matrisi boyutu: {X.shape}")
print(f"Hedef vektoru boyutu: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"\nEgitim seti: {X_train.shape}")
print(f"Test seti: {X_test.shape}")

class_names = ["Dusuk (0)", "Orta (1)", "Yuksek (2)", "Cok Yuksek (3)"]

print("\nDecision Tree modeli egitiliyor...")
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dogruluk (Accuracy): {accuracy:.4f}")
print("\nSiniflandirma Raporu:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gercek")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("Karar agaci gorsellestiriliyor...")
plt.figure(figsize=(28, 14))
plot_tree(
    model, feature_names=list(X.columns), class_names=class_names,
    filled=True, rounded=True, fontsize=8, max_depth=3
)
plt.title("Decision Tree Gorsellestirmesi (ilk 3 seviye)", fontsize=20)
plt.tight_layout()
plt.savefig("figures/decision_tree.png", dpi=150)
plt.close()

tree_rules = export_text(model, feature_names=list(X.columns), spacing=3)
with open("figures/decision_rules.txt", "w") as f:
    f.write(tree_rules)
print("Karar kurallari kaydedildi: figures/decision_rules.txt")

print("\nOzellik onem siralamasi hesaplaniyor...")
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print("En Onemli 10 Ozellik:")
print(feature_importance.head(10).to_string(index=False))
feature_importance.to_csv("figures/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x="importance", y="feature",
            hue="feature", palette="viridis", legend=False)
plt.title("Feature Importance (En Onemli 10 Ozellik)")
plt.xlabel("Onem Skoru")
plt.ylabel("Ozellik")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()

print("\nOverfitting kontrolu...")
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Egitim Dogrulugu: {train_accuracy:.4f}")
print(f"Test Dogrulugu:   {test_accuracy:.4f}")
if train_accuracy - test_accuracy > 0.1:
    print("Uyari: Model overfitting yapiyor olabilir (fark > %10).")
else:
    print("Egitim-test farki makul seviyede, belirgin overfitting yok.")

print("\nFarkli max_depth degerleriyle model karsilastirmasi...")
depths = range(1, 16)
train_scores, test_scores = [], []
for depth in depths:
    dt = DecisionTreeClassifier(
        max_depth=depth, criterion="gini",
        min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(list(depths), train_scores, "b-o", label="Egitim Dogrulugu")
plt.plot(list(depths), test_scores, "r-s", label="Test Dogrulugu")
plt.xlabel("Max Depth")
plt.ylabel("Dogruluk")
plt.title("Derinlik - Dogruluk Iliskisi")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(list(depths))
plt.tight_layout()
plt.savefig("figures/depth_accuracy_curve.png", dpi=150)
plt.close()

best_depth = list(depths)[int(np.argmax(test_scores))]
print(f"En iyi test dogrulugunu veren derinlik: {best_depth} (test acc={max(test_scores):.4f})")

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

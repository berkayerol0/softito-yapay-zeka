"""
URUN YORUMU DUYGU ANALIZI - Multinomial Naive Bayes (3 Sinif)
Amac: Bir e-ticaret urun yorumunun POZITIF, NOTR veya NEGATIF oldugunu
      metin icinden otomatik tespit etmek. Gercek e-ticaret platformlarinda
      (Trendyol, Hepsiburada, Amazon) yorum moderasyon ve analiz sistemleri
      genelde ikili degil UC SINIFLI calisir, cunku "fena degil ama..."
      turu yorumlar isletme acisindan pozitif veya negatiften FARKLI bir
      aksiyon gerektirir (orn. urun gelistirme sinyali olarak izlenir).

      Naive Bayes, metin siniflandirmanin en klasik ve hala en cok
      kullanilan algoritmalarindan biridir - spam filtreleme, duygu
      analizi, haber kategorileme gibi alanlarda hizli ve az veriyle
      bile etkili sonuc verir.

Not: Gercek bir e-ticaret yorum veri seti bu ortamda bulunmadigi icin,
     pozitif/notr/negatif kelime havuzlari ve cumle sablonlari
     birlestirilerek gercekci TURKCE urun yorumlari SENTETIK olarak
     uretilir.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
os.makedirs('figures', exist_ok=True)

print("Urun Yorumu Duygu Analizi (3 Sinif) - Sentetik Yorum Uretimi")

products = ["telefon", "kulaklık", "ayakkabı", "kitap", "laptop", "çanta",
            "saat", "tişört", "kahve makinesi", "kamera"]

positive_templates = [
    "Bu {p} gerçekten harika, çok memnun kaldım.",
    "{p} beklediğimden çok daha kaliteli çıktı, kesinlikle tavsiye ederim.",
    "Fiyatına göre mükemmel bir {p}, herkese öneririm.",
    "{p} hızlı geldi ve paketleme çok özenliydi, teşekkürler.",
    "Uzun zamandır böyle iyi bir {p} kullanmamıştım, çok başarılı.",
    "{p} tam beklediğim gibiydi, hiç sorun yaşamadım.",
    "Kaliteli malzeme, şık tasarım, bu {p}'a bayıldım.",
    "{p} kullanışlı ve dayanıklı, paramın karşılığını aldım.",
    "Harika bir alışverişti, {p} çok kaliteli.",
    "Bu fiyata bu kalitede bir {p} bulmak zor, çok beğendim.",
]

negative_templates = [
    "Bu {p} tam bir hayal kırıklığı, hiç memnun kalmadım.",
    "{p} beklediğimin çok altında, param boşa gitti.",
    "Kalitesiz malzeme, {p} birkaç günde bozuldu.",
    "{p} geç geldi ve kutusu hasarlıydı, çok kötü deneyim.",
    "Hiç tavsiye etmiyorum, bu {p} paranıza değmez.",
    "{p} açıklamada yazandan çok farklı çıktı, kandırıldım.",
    "Berbat bir kalite, {p} ilk kullanımda arıza verdi.",
    "{p} çok pahalı ve düşük kaliteli, pişman oldum.",
    "Müşteri hizmetleri de kötüydü, {p} de kalitesizdi.",
    "Bu {p}'ı asla tekrar almam, tam bir zaman kaybı.",
]

# Gercek bir NOTR sinif: ne belirgin pozitif ne belirgin negatif duygu iceren,
# tanimlayici/olgusal veya "ortalama" ifadeler - gercek e-ticaret
# platformlarinda 3 yildizli yorumlarin tipik dili.
neutral_templates = [
    "{p} açıklamada belirtildiği gibi geldi, standart bir ürün.",
    "Fiyatı ortalama, {p} performansı da ortalama seviyede.",
    "{p} fena değil ama beklediğim kadar iyi de değildi.",
    "Ne çok iyi ne çok kötü, sıradan bir {p}.",
    "{p} güzel ama kargo çok geç geldi, karışık duygular içindeyim.",
    "Bazı yönleri iyi ama {p} genel olarak vasat kaldı.",
    "{p} idare eder, fiyatına göre normal.",
    "Kutudan çıktığı haliyle {p} beklediğim gibiydi, ne fazla ne eksik.",
    "{p} hakkında henüz kesin bir fikrim yok, kullandıkça göreceğiz.",
    "Ortalama bir {p}, ne öve öve bitiremem ne de şikayetçiyim.",
]

records = []
n_per_class = 700

for _ in range(n_per_class):
    p = random.choice(products)
    text = random.choice(positive_templates).format(p=p)
    records.append((text, "Pozitif"))

for _ in range(n_per_class):
    p = random.choice(products)
    text = random.choice(negative_templates).format(p=p)
    records.append((text, "Negatif"))

for _ in range(n_per_class):
    p = random.choice(products)
    text = random.choice(neutral_templates).format(p=p)
    records.append((text, "Nötr"))

df = pd.DataFrame(records, columns=["review", "sentiment"])

# Gercek yorum etiketleme surecinde siniflar arasi (ozellikle komsu
# sinifllar arasi) belirsizlik/anlasmazlik olur - bunu simule etmek icin
# bilincli bir etiket gurultusu eklenir (orn. hafif olumlu bir yorum bazen
# "Notr" bazen "Pozitif" olarak etiketlenebilir).
def add_boundary_noise(label, p=0.15):
    if random.random() > p:
        return label
    if label == "Pozitif":
        return random.choice(["Pozitif", "Nötr"])
    if label == "Negatif":
        return random.choice(["Negatif", "Nötr"])
    return random.choice(["Nötr", "Pozitif", "Negatif"])

df["sentiment"] = df["sentiment"].apply(add_boundary_noise)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Toplam yorum sayisi: {len(df)}")
print(f"Sinif dagilimi:\n{df['sentiment'].value_counts().to_string()}")
print(f"\nOrnek yorumlar:")
print(df.sample(6, random_state=RANDOM_STATE).to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["sentiment"]
)
print(f"\nEgitim: {len(X_train)} | Test: {len(X_test)}")

print("\nTF-IDF vektorlestirme uygulaniyor...")
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Ozellik (kelime/n-gram) sayisi: {X_train_tfidf.shape[1]}")

print("\nMultinomial Naive Bayes egitiliyor (3 sinif)...")
model = MultinomialNB(alpha=0.5)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nSiniflandirma Raporu:")
class_labels = ["Negatif", "Nötr", "Pozitif"]
print(classification_report(y_test, y_pred, labels=class_labels))

print("\nConfusion matrix kaydediliyor...")
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Tahmin")
plt.ylabel("Gercek")
plt.title("Confusion Matrix - Duygu Analizi (3 Sinif)")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
plt.close()

print("En ayirt edici kelimeler hesaplaniyor (her sinif icin)...")
feature_names = np.array(vectorizer.get_feature_names_out())
class_order = list(model.classes_)  # alfabetik: Negatif, Notr, Pozitif

top_words_per_class = {}
for i, cls in enumerate(class_order):
    other_idx = [j for j in range(len(class_order)) if j != i]
    other_mean = model.feature_log_prob_[other_idx].mean(axis=0)
    diff = model.feature_log_prob_[i] - other_mean
    top_idx = np.argsort(diff)[-12:][::-1]
    top_words_per_class[cls] = pd.DataFrame({
        "word": feature_names[top_idx],
        "score": diff[top_idx]
    })
    print(f"\nEn ayirt edici 8 kelime/ifade - {cls}:")
    print(top_words_per_class[cls].head(8).to_string(index=False))
    top_words_per_class[cls].to_csv(f"figures/top_words_{cls.lower().replace('ö','o')}.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = {"Negatif": "#dc2626", "Nötr": "#f59e0b", "Pozitif": "#059669"}
for ax, cls in zip(axes, ["Negatif", "Nötr", "Pozitif"]):
    data = top_words_per_class[cls].sort_values("score")
    ax.barh(data["word"], data["score"], color=colors[cls])
    ax.set_title(f"{cls} Sinifi icin En Ayirt Edici Ifadeler")
    ax.set_xlabel("Ayirt Edicilik Skoru")
plt.tight_layout()
plt.savefig("figures/top_words_by_sentiment.png", dpi=150)
plt.close()

print("\nGorseller kaydedildi: figures/")
print("Tamamlandi.")

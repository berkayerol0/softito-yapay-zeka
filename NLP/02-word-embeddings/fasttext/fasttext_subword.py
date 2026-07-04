"""
==============================================================================
 FastText — Subword (Karakter n-gram) Tabanlı Kelime Vektörleri
==============================================================================

Tek bir Python dosyasında uçtan uca bir proje:
  1) text8 korpusunu indirir (Word2Vec/GloVe projeleriyle AYNI veri)
  2) FastText'in çekirdek fikrini — kelimeyi karakter n-gram'larına bölmeyi —
     sıfırdan Python'da gösterir (subword çıkarımı + hash bucket)
  3) Facebook'un resmi `fasttext` kütüphanesiyle skip-gram modeli eğitir
     (sektörde kullanılan yol; subword eğitimi CPU'da sıfırdan çok yavaş
     olduğu için endüstri standardı kütüphane tercih edildi)
  4) Değerlendirir: en yakın komşular, resmi analoji testi, t-SNE
  5) FastText'in İMZA ÖZELLİĞİNİ gösterir: daha önce HİÇ görülmemiş (OOV)
     kelimelere bile subword'lerden anlamlı vektör üretebilme
  6) Sonuçları Word2Vec ve GloVe projelerimizle karşılaştırır

Veri seti: text8 — ~2M kelimelik alt küme (diğer iki projeyle aynı).
Değerlendirme: Mikolov'un resmi analoji test seti (questions-words.txt).
==============================================================================
"""

import os
import time
import random
import urllib.request
import tarfile

import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.manifold import TSNE
from collections import Counter

import fasttext

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TEXT8_TAR_URL = "https://codeload.github.com/lacteolus/word2vec/tar.gz/refs/heads/main"
QUESTIONS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
TEXT8_PATH = os.path.join(DATA_DIR, "text8_full.txt")
SUBSET_PATH = os.path.join(DATA_DIR, "text8_subset.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions-words.txt")

SUBSET_CHARS = 12_000_000     # diğer projelerle AYNI alt küme (adil kıyas)
MIN_COUNT = 5
WINDOW_SIZE = 5
EMB_DIM = 100
MIN_N = 3                     # en kısa karakter n-gram
MAX_N = 5                     # en uzun karakter n-gram
K_NEGATIVE = 5
EPOCHS = 5

N_ANALOGY_PER_CATEGORY = 150

print(f"CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Veriyi indir
# ──────────────────────────────────────────────────────────────────────────
def download_text8():
    if os.path.exists(TEXT8_PATH):
        print(f"text8 zaten mevcut: {TEXT8_PATH}"); return
    print("text8 indiriliyor...")
    tar_path = os.path.join(DATA_DIR, "_repo.tar.gz")
    urllib.request.urlretrieve(TEXT8_TAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = [m for m in tar.getmembers() if m.name.endswith("dataset/text8.txt")][0]
        member.name = os.path.basename(TEXT8_PATH)
        tar.extract(member, path=DATA_DIR)
    os.remove(tar_path)


def download_questions():
    if os.path.exists(QUESTIONS_PATH):
        print(f"Analoji test seti zaten mevcut: {QUESTIONS_PATH}"); return
    print("Google analoji test seti indiriliyor...")
    urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)


download_text8()
download_questions()

with open(TEXT8_PATH) as f:
    raw_text = f.read(SUBSET_CHARS)
with open(SUBSET_PATH, "w") as f:
    f.write(raw_text)
tokens = raw_text.split()
print(f"\nAlt küme boyutu: {len(raw_text):,} karakter → {len(tokens):,} kelime (token)")


# ──────────────────────────────────────────────────────────────────────────
# 2) FastText'in ÇEKİRDEK FİKRİ — subword (karakter n-gram) çıkarımı (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
# FastText'in Word2Vec'ten temel farkı: kelimeyi bölünmez bir bütün olarak
# görmez, karakter parçalarından (n-gram) oluşan bir bileşim olarak görür.
# "<where>" kelimesi → <wh, whe, her, ere, re>, <whe, wher, here, ere> ...
# Kelime vektörü = kendi vektörü + tüm subword vektörlerinin toplamı.
# Bu sayede daha önce hiç görmediği kelimelere bile (parçalarını tanıdığı için)
# vektör üretebilir — Word2Vec bunu YAPAMAZ.
def get_subwords(word, nmin=MIN_N, nmax=MAX_N):
    w = "<" + word + ">"
    ngrams = []
    for n in range(nmin, nmax + 1):
        for i in range(len(w) - n + 1):
            ngrams.append(w[i:i + n])
    return ngrams


print("\n── FastText Çekirdek Fikri: Subword Çıkarımı (örnek) ──")
for demo_word in ["playing", "where", "unhappiness"]:
    sw = get_subwords(demo_word)
    print(f"  '{demo_word}' → {sw[:8]}{' ...' if len(sw) > 8 else ''}  (toplam {len(sw)} subword)")


# ──────────────────────────────────────────────────────────────────────────
# 3) Resmi FastText kütüphanesiyle eğitim (skip-gram + subword)
# ──────────────────────────────────────────────────────────────────────────
print("\n── FastText Modeli Eğitiliyor (resmi kütüphane, skip-gram) ──")
t0 = time.time()
model = fasttext.train_unsupervised(
    SUBSET_PATH,
    model="skipgram",
    dim=EMB_DIM,
    ws=WINDOW_SIZE,
    minCount=MIN_COUNT,
    minn=MIN_N,
    maxn=MAX_N,
    neg=K_NEGATIVE,
    epoch=EPOCHS,
    thread=1,
)
print(f"Eğitim süresi: {time.time()-t0:.0f}s | vocab: {len(model.words):,}")

vocab_words = model.words
word2id = {w: i for i, w in enumerate(vocab_words)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab_words)

word_vectors = np.array([model.get_word_vector(w) for w in vocab_words])
norm_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 4) En yakın komşu sorguları
# ──────────────────────────────────────────────────────────────────────────
query_words = ["king", "computer", "france", "music", "water", "good", "run", "science"]
print("\n── En Yakın Komşular ──")
neighbor_results = {}
for w in query_words:
    neighbors = model.get_nearest_neighbors(w, k=6)
    neighbor_results[w] = [(word, score) for score, word in neighbors]
    print(f"  {w:<12} → " + ", ".join(f"{word}({score:.2f})" for score, word in neighbors))


# ──────────────────────────────────────────────────────────────────────────
# 5) FastText'in İMZA ÖZELLİĞİ — OOV (görülmemiş kelime) vektörü üretebilme
# ──────────────────────────────────────────────────────────────────────────
# Aşağıdaki kelimeler ya vocab'da yok ya da nadir; FastText yine de subword'lerden
# vektör üretip anlamlı komşular bulabiliyor. Word2Vec/GloVe bunları hiç
# tanımadığı için KeyError verir / sıfır vektör döndürür.
print("\n── OOV (Görülmemiş Kelime) Testi — FastText'in İmza Özelliği ──")
oov_words = ["playfulness", "unhappily", "reconfigure", "hyperconnectivity"]
oov_results = {}
for w in oov_words:
    in_vocab = w in word2id
    neighbors = model.get_nearest_neighbors(w, k=5)
    oov_results[w] = [(word, score) for score, word in neighbors]
    tag = "(vocab'da)" if in_vocab else "(vocab DIŞI - sadece subword'lerden!)"
    print(f"  '{w}' {tag}")
    print(f"      → " + ", ".join(f"{word}({score:.2f})" for score, word in neighbors))


# ──────────────────────────────────────────────────────────────────────────
# 6) Analoji görevi
# ──────────────────────────────────────────────────────────────────────────
def load_analogy_questions(path):
    categories, cur_cat = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cur_cat = line[1:].strip(); categories[cur_cat] = []; continue
            words = line.lower().split()
            if len(words) == 4:
                categories[cur_cat].append(words)
    return categories


def evaluate_analogies(vectors_norm, w2id, categories, per_category=N_ANALOGY_PER_CATEGORY):
    results = {}
    vt = torch.tensor(vectors_norm)
    for cat, questions in categories.items():
        valid = [q for q in questions if all(w in w2id for w in q)]
        if not valid:
            results[cat] = (0, 0); continue
        sample = random.sample(valid, min(per_category, len(valid)))
        correct = 0
        for a, b, c, d in sample:
            ia, ib, ic, id_ = w2id[a], w2id[b], w2id[c], w2id[d]
            query = vt[ib] - vt[ia] + vt[ic]
            query = query / (query.norm() + 1e-9)
            sims = vt @ query
            sims[ia] = -1; sims[ib] = -1; sims[ic] = -1
            pred = int(torch.argmax(sims))
            if pred == id_:
                correct += 1
        results[cat] = (correct, len(sample))
    return results


categories = load_analogy_questions(QUESTIONS_PATH)
print(f"\n── Analoji Değerlendirmesi ──")
t0 = time.time()
ft_results = evaluate_analogies(norm_vectors, word2id, categories)
total_correct = sum(c for c, n in ft_results.values())
total_n = sum(n for c, n in ft_results.values())
for cat, (c, n) in ft_results.items():
    acc = 100 * c / n if n else 0
    print(f"  {cat:<32} {c:4d}/{n:<4d} (%{acc:5.1f})")
print(f"  {'TOPLAM':<32} {total_correct:4d}/{total_n:<4d} (%{100*total_correct/max(total_n,1):5.1f})  [{time.time()-t0:.1f}s]")


# ──────────────────────────────────────────────────────────────────────────
# 7) Word2Vec ve GloVe projeleriyle kıyaslama (üçlü karşılaştırma)
# ──────────────────────────────────────────────────────────────────────────
def eval_checkpoint(path, key_in):
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, weights_only=False)
    if key_in == "sum":  # glove: w + wc
        vecs = (ckpt["w"] + ckpt["wc"]).numpy()
    else:
        vecs = ckpt["in_embed"].numpy()
    w2id = ckpt["word2id"]
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    res = evaluate_analogies(norm, w2id, categories)
    return res


print("\n── Üçlü Karşılaştırma (aynı korpus) ──")
w2v_res = eval_checkpoint("reference_word2vec_model.pt", "in_embed")
glove_res = eval_checkpoint("reference_glove_model.pt", "sum")

comparison = {"FastText": ft_results}
if w2v_res:
    comparison["Word2Vec"] = w2v_res
if glove_res:
    comparison["GloVe"] = glove_res

for name, res in comparison.items():
    tc = sum(c for c, n in res.values()); tn = sum(n for c, n in res.values())
    print(f"  {name:<10} analoji doğruluğu: {tc}/{tn} (%{100*tc/max(tn,1):.1f})")


# ──────────────────────────────────────────────────────────────────────────
# 8) Görseller
# ──────────────────────────────────────────────────────────────────────────
# 8a) Üçlü analoji karşılaştırması
cats_sorted = sorted(ft_results.keys())
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(cats_sorted))
colors = {"FastText": "#E24B4A", "Word2Vec": "#534AB7", "GloVe": "#1D9E75"}
n_models = len(comparison)
width = 0.8 / n_models
for mi, (name, res) in enumerate(comparison.items()):
    acc = [100 * res[c][0] / max(res[c][1], 1) for c in cats_sorted]
    ax.bar(x + (mi - (n_models - 1) / 2) * width, acc, width, label=name, color=colors.get(name, "#888"))
ax.set_xticks(x); ax.set_xticklabels(cats_sorted, rotation=40, ha="right")
ax.set_ylabel("Doğruluk (%)")
ax.set_title("Analoji Görevi — FastText vs Word2Vec vs GloVe (Kategori Bazlı)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_analogy_accuracy_comparison.png"), dpi=130)
plt.close(fig)

# 8b) t-SNE
tsne_groups = {
    "Ülkeler": ["france", "germany", "italy", "spain", "china", "japan", "russia", "england", "canada", "india"],
    "Başkentler": ["paris", "berlin", "rome", "madrid", "beijing", "tokyo", "moscow", "london", "ottawa", "delhi"],
    "Sayılar": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "Meslekler": ["doctor", "teacher", "engineer", "lawyer", "writer", "artist", "scientist", "musician"],
    "Zaman": ["day", "week", "month", "year", "morning", "evening", "night", "hour"],
    "Duygu/Sıfat": ["good", "bad", "happy", "sad", "large", "small", "fast", "slow"],
}
plot_words, plot_labels = [], []
for group, words in tsne_groups.items():
    for w in words:
        if w in word2id:
            plot_words.append(w); plot_labels.append(group)
plot_vecs = np.array([norm_vectors[word2id[w]] for w in plot_words])
tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca", max_iter=1000)
coords = tsne.fit_transform(plot_vecs)

fig, ax = plt.subplots(figsize=(11, 8.5))
palette = ["#534AB7", "#E24B4A", "#1D9E75", "#BA7517", "#3178C6", "#C2338B"]
for idx, group in enumerate(tsne_groups.keys()):
    mask = [j for j, l in enumerate(plot_labels) if l == group]
    ax.scatter(coords[mask, 0], coords[mask, 1], color=palette[idx % len(palette)], label=group, s=60)
    for j in mask:
        ax.annotate(plot_words[j], (coords[j, 0], coords[j, 1]), fontsize=9, alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")
ax.set_title("t-SNE ile FastText Kelime Vektörü Görselleştirmesi")
ax.legend(loc="best"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_tsne_word_clusters.png"), dpi=130)
plt.close(fig)

# 8c) En yakın komşular
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, w in zip(axes.flat, query_words):
    nn_list = neighbor_results.get(w)
    if not nn_list:
        ax.axis("off"); continue
    names = [n for n, s in nn_list[:6]][::-1]
    scores = [s for n, s in nn_list[:6]][::-1]
    ax.barh(names, scores, color="#E24B4A")
    ax.set_title(f'"{w}"', fontsize=12)
    ax.set_xlim(0, 1)
fig.suptitle("En Yakın Komşu Kelimeler (kosinüs benzerliği) — FastText", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_nearest_neighbors.png"), dpi=130)
plt.close(fig)

# 8d) OOV özel görseli — FastText'in imza özelliği
fig, axes = plt.subplots(1, len(oov_words), figsize=(16, 4.5))
for ax, w in zip(axes, oov_words):
    nn_list = oov_results.get(w, [])
    names = [n for n, s in nn_list[:5]][::-1]
    scores = [s for n, s in nn_list[:5]][::-1]
    ax.barh(names, scores, color="#BA7517")
    ax.set_title(f'OOV: "{w}"', fontsize=11)
    ax.set_xlim(0, 1)
fig.suptitle("Görülmemiş (OOV) Kelimeler İçin Bile Anlamlı Komşular — FastText'in Farkı", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_oov_capability.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 9) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
model.save_model(os.path.join(DATA_DIR, "fasttext_model.bin"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,}")
print(f"FastText analoji doğruluğu: %{100*total_correct/max(total_n,1):.1f}")
print("FastText, OOV kelimelere bile subword'lerden vektör üretebilir (imza özellik).")
print("Tamamlandı.")

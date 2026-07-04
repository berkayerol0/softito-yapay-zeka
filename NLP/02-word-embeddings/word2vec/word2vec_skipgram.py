"""
==============================================================================
 Word2Vec (Skip-gram + Negative Sampling) — Sıfırdan PyTorch İmplementasyonu
==============================================================================

Tek bir Python dosyasında uçtan uca bir proje:
  1) text8 korpusunu indirir (Wikipedia'dan temizlenmiş klasik word2vec verisi)
  2) Kelime dağarcığı (vocab) kurar, sık kelimeleri alt örnekler (subsampling)
  3) Skip-gram çiftlerini üretir, negatif örnekleme (negative sampling) ile
     PyTorch'ta SIFIRDAN bir Word2Vec modeli eğitir (Mikolov ve ark., 2013)
  4) Öğrenilen vektörleri değerlendirir:
       - en yakın komşu sorguları
       - resmi Google analoji test seti (questions-words.txt) doğruluğu
       - t-SNE ile 2 boyutlu görselleştirme
  5) Sonuçları, aynı veriyle eğitilmiş bir `gensim` Word2Vec modeliyle kıyaslar

Veri seti: text8 — Wikipedia'dan temizlenmiş ~17M kelimelik metin (bu projede
hız için ilk ~5M kelimelik alt küme kullanılır). Değerlendirme için Mikolov'un
resmi analoji test seti (questions-words.txt, 19.544 soru) kullanılır.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TEXT8_TAR_URL = "https://codeload.github.com/lacteolus/word2vec/tar.gz/refs/heads/main"
QUESTIONS_URL = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"
TEXT8_PATH = os.path.join(DATA_DIR, "text8_full.txt")
QUESTIONS_PATH = os.path.join(DATA_DIR, "questions-words.txt")

SUBSET_CHARS = 12_000_000     # ~2M kelimelik alt küme — hız/kalite dengesi
MIN_COUNT = 5
SUBSAMPLE_T = 1e-5
WINDOW_SIZE = 5
EMB_DIM = 100
K_NEGATIVE = 5
BATCH_SIZE = 65536
EPOCHS = 4
LR_START = 0.01
LR_END = 0.0005

N_ANALOGY_PER_CATEGORY = 150   # kategori başına en fazla bu kadar soru değerlendirilir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Veriyi indir
# ──────────────────────────────────────────────────────────────────────────
def download_text8():
    if os.path.exists(TEXT8_PATH):
        print(f"text8 zaten mevcut: {TEXT8_PATH}")
        return
    print("text8 indiriliyor (GitHub mirror üzerinden, ~100MB)...")
    tar_path = os.path.join(DATA_DIR, "_repo.tar.gz")
    urllib.request.urlretrieve(TEXT8_TAR_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        member = [m for m in tar.getmembers() if m.name.endswith("dataset/text8.txt")][0]
        member.name = os.path.basename(TEXT8_PATH)
        tar.extract(member, path=DATA_DIR)
    os.remove(tar_path)


def download_questions():
    if os.path.exists(QUESTIONS_PATH):
        print(f"Analoji test seti zaten mevcut: {QUESTIONS_PATH}")
        return
    print("Google analoji test seti indiriliyor...")
    urllib.request.urlretrieve(QUESTIONS_URL, QUESTIONS_PATH)


download_text8()
download_questions()

with open(TEXT8_PATH) as f:
    raw_text = f.read(SUBSET_CHARS)
tokens = raw_text.split()
print(f"\nAlt küme boyutu: {len(raw_text):,} karakter → {len(tokens):,} kelime (token)")


# ──────────────────────────────────────────────────────────────────────────
# 2) Kelime dağarcığı (vocab) + alt örnekleme (subsampling)
# ──────────────────────────────────────────────────────────────────────────
counts = Counter(tokens)
vocab_words = [w for w, c in counts.items() if c >= MIN_COUNT]
vocab_words.sort(key=lambda w: -counts[w])   # en sık kelime id=0

word2id = {w: i for i, w in enumerate(vocab_words)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab_words)
print(f"Vocab boyutu (min_count={MIN_COUNT}): {VOCAB_SIZE:,}")

token_ids = np.array([word2id[w] for w in tokens if w in word2id], dtype=np.int64)
print(f"Kelime dağarcığındaki toplam token: {len(token_ids):,}")

# Sık kelimeleri (the, of, a...) olasılıksal olarak seyrekleştir — Mikolov ve ark. (2013)
freqs = np.zeros(VOCAB_SIZE, dtype=np.float64)
for i, w in id2word.items():
    freqs[i] = counts[w]
freq_ratio = freqs / freqs.sum()
keep_prob = (np.sqrt(freq_ratio / SUBSAMPLE_T) + 1) * (SUBSAMPLE_T / freq_ratio)
keep_prob = np.clip(keep_prob, 0, 1)

rand_vals = np.random.rand(len(token_ids))
train_ids = token_ids[rand_vals < keep_prob[token_ids]]
print(f"Alt örnekleme sonrası token sayısı: {len(train_ids):,}")

# Negatif örnekleme için gürültü (noise) dağılımı: unigram^0.75 (orijinal makale)
noise_dist = torch.tensor(freqs ** 0.75)
noise_dist = noise_dist / noise_dist.sum()


# ──────────────────────────────────────────────────────────────────────────
# 3) Skip-gram çiftlerini üret (vektörize, kayan pencere)
# ──────────────────────────────────────────────────────────────────────────
def build_skipgram_pairs(ids, window):
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers.append(ids[:-offset]);  contexts.append(ids[offset:])   # sağa bakan bağlam
        centers.append(ids[offset:]);   contexts.append(ids[:-offset])  # sola bakan bağlam
    return np.concatenate(centers), np.concatenate(contexts)


t0 = time.time()
center_ids, context_ids = build_skipgram_pairs(train_ids, WINDOW_SIZE)
print(f"Skip-gram çift sayısı: {len(center_ids):,} ({time.time()-t0:.1f}s)")


class SkipGramDataset(Dataset):
    def __init__(self, centers, contexts):
        self.centers = torch.from_numpy(centers)
        self.contexts = torch.from_numpy(contexts)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


train_loader = DataLoader(
    SkipGramDataset(center_ids, context_ids),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
)


# ──────────────────────────────────────────────────────────────────────────
# 4) Model — Skip-gram + Negative Sampling (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class SkipGramNegativeSampling(nn.Module):
    """
    Her kelime için İKİ ayrı vektör öğrenilir:
      - in_embed  : kelime "merkez" (center) rolündeyken kullanılan vektör
      - out_embed : kelime "bağlam" (context) rolündeyken kullanılan vektör
    Eğitim sonunda genelde in_embed nihai kelime vektörü olarak kullanılır.
    """
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, emb_dim)
        self.out_embed = nn.Embedding(vocab_size, emb_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)                                   # (B, D)
        v_o = self.out_embed(context)                                  # (B, D)
        v_neg = self.out_embed(negatives)                               # (B, K, D)

        pos_score = torch.sum(v_c * v_o, dim=1)                         # (B,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        neg_score = torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)        # (B, K)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()


model = SkipGramNegativeSampling(VOCAB_SIZE, EMB_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)
noise_dist = noise_dist.to(device)

# Negatif örnekleme darboğazını azaltmak için önceden büyük bir negatif havuzu çek
NEG_POOL_SIZE = 20_000_000
neg_pool = torch.multinomial(noise_dist, NEG_POOL_SIZE, replacement=True)
neg_pool_ptr = 0

n_total_steps = EPOCHS * len(train_loader)
print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
print(f"Epoch başına adım: {len(train_loader):,} | Toplam adım: {n_total_steps:,}")


# ──────────────────────────────────────────────────────────────────────────
# 5) Eğitim döngüsü (doğrusal öğrenme oranı azaltma — orijinal word2vec gibi)
# ──────────────────────────────────────────────────────────────────────────
print("\n── Eğitim başlıyor ──")
step = 0
loss_history = []
t_train_start = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_loss, n_batches = 0.0, 0
    for center, context in train_loader:
        center, context = center.to(device), context.to(device)
        n_needed = len(center) * K_NEGATIVE
        if neg_pool_ptr + n_needed > NEG_POOL_SIZE:
            neg_pool_ptr = 0
        negatives = neg_pool[neg_pool_ptr: neg_pool_ptr + n_needed].view(len(center), K_NEGATIVE)
        neg_pool_ptr += n_needed

        lr = LR_START - (LR_START - LR_END) * (step / n_total_steps)
        for g in optimizer.param_groups:
            g["lr"] = max(lr, LR_END)

        optimizer.zero_grad()
        loss = model(center, context, negatives)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
        step += 1

    avg_loss = epoch_loss / n_batches
    loss_history.append(avg_loss)
    elapsed = time.time() - t_train_start
    print(f"Epoch {epoch}/{EPOCHS} | ortalama kayıp: {avg_loss:.4f} | geçen süre: {elapsed:.0f}s")

print(f"Eğitim tamamlandı. Toplam süre: {time.time()-t_train_start:.0f}s")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, EPOCHS + 1), loss_history, marker="o", color="#534AB7")
ax.set_xlabel("Epoch")
ax.set_ylabel("Ortalama Kayıp (Negative Sampling)")
ax.set_title("Eğitim Kaybı — Skip-gram + Negative Sampling")
ax.set_xticks(range(1, EPOCHS + 1))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_loss.png"), dpi=130)
plt.close(fig)

# Nihai kelime vektörleri: in_embed katmanı (merkez kelime vektörleri)
word_vectors = model.in_embed.weight.detach().cpu().numpy()
norm_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + 1e-9)


# ──────────────────────────────────────────────────────────────────────────
# 6) En yakın komşu sorguları
# ──────────────────────────────────────────────────────────────────────────
def nearest_neighbors(word, k=8):
    if word not in word2id:
        return None
    idx = word2id[word]
    sims = norm_vectors @ norm_vectors[idx]
    top_idx = np.argsort(-sims)[1:k + 1]   # kendisini hariç tut
    return [(id2word[i], float(sims[i])) for i in top_idx]


query_words = ["king", "computer", "france", "music", "water", "good", "run", "science"]
print("\n── En Yakın Komşular ──")
neighbor_results = {}
for w in query_words:
    nn_list = nearest_neighbors(w)
    neighbor_results[w] = nn_list
    if nn_list:
        formatted = ", ".join(f"{n}({s:.2f})" for n, s in nn_list[:6])
        print(f"  {w:<12} → {formatted}")
    else:
        print(f"  {w:<12} → [vocab dışında]")


# ──────────────────────────────────────────────────────────────────────────
# 7) Analoji görevi — resmi Google analoji test seti (3CosAdd yöntemi)
# ──────────────────────────────────────────────────────────────────────────
def load_analogy_questions(path):
    categories = {}
    cur_cat = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                cur_cat = line[1:].strip()
                categories[cur_cat] = []
                continue
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
            results[cat] = (0, 0)
            continue
        sample = random.sample(valid, min(per_category, len(valid)))

        correct = 0
        for a, b, c, d in sample:
            ia, ib, ic, id_ = w2id[a], w2id[b], w2id[c], w2id[d]
            query = vt[ib] - vt[ia] + vt[ic]
            query = query / (query.norm() + 1e-9)
            sims = vt @ query
            sims[ia] = -1; sims[ib] = -1; sims[ic] = -1   # girdi kelimelerini hariç tut
            pred = int(torch.argmax(sims))
            if pred == id_:
                correct += 1
        results[cat] = (correct, len(sample))
    return results


categories = load_analogy_questions(QUESTIONS_PATH)
print(f"\n── Analoji Değerlendirmesi (kategori başına en fazla {N_ANALOGY_PER_CATEGORY} soru) ──")
t0 = time.time()
custom_results = evaluate_analogies(norm_vectors, word2id, categories)

total_correct = sum(c for c, n in custom_results.values())
total_n = sum(n for c, n in custom_results.values())
for cat, (c, n) in custom_results.items():
    acc = 100 * c / n if n else 0
    print(f"  {cat:<32} {c:4d}/{n:<4d} (%{acc:5.1f})")
print(f"  {'TOPLAM':<32} {total_correct:4d}/{total_n:<4d} (%{100*total_correct/max(total_n,1):5.1f})  [{time.time()-t0:.1f}s]")


# ──────────────────────────────────────────────────────────────────────────
# 8) gensim ile kıyaslama — endüstri standardı kütüphaneye karşı
# ──────────────────────────────────────────────────────────────────────────
print("\n── Gensim Word2Vec ile Kıyaslama ──")
from gensim.models import Word2Vec as GensimWord2Vec

t0 = time.time()
GENSIM_CHUNK = 1000  # gensim, çok uzun tek bir "cümle" ile beslenirse doğru eğitmiyor;
                       # gerçekçi cümle uzunluklarına böl
sentences = [tokens[i:i + GENSIM_CHUNK] for i in range(0, len(tokens), GENSIM_CHUNK)]
gensim_model = GensimWord2Vec(
    sentences=sentences,
    vector_size=EMB_DIM,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    sg=1,                      # skip-gram
    negative=K_NEGATIVE,
    sample=SUBSAMPLE_T,
    epochs=EPOCHS,
    workers=1,
    seed=SEED,
)
print(f"Gensim eğitim süresi: {time.time()-t0:.0f}s | vocab: {len(gensim_model.wv):,}")

gensim_vectors = np.zeros((VOCAB_SIZE, EMB_DIM), dtype=np.float32)
gensim_word2id = {}
for i, w in enumerate(vocab_words):
    if w in gensim_model.wv:
        gensim_vectors[i] = gensim_model.wv[w]
        gensim_word2id[w] = i
gensim_norm = gensim_vectors / (np.linalg.norm(gensim_vectors, axis=1, keepdims=True) + 1e-9)

gensim_results = evaluate_analogies(gensim_norm, gensim_word2id, categories)
gensim_correct = sum(c for c, n in gensim_results.values())
gensim_n = sum(n for c, n in gensim_results.values())
print(f"Gensim analoji doğruluğu: {gensim_correct}/{gensim_n} (%{100*gensim_correct/max(gensim_n,1):.1f})")
print(f"Bizim modelin analoji doğruluğu: {total_correct}/{total_n} (%{100*total_correct/max(total_n,1):.1f})")


# ──────────────────────────────────────────────────────────────────────────
# 9) Görseller
# ──────────────────────────────────────────────────────────────────────────
# 9a) Kategori bazlı analoji doğruluğu karşılaştırması
cats_sorted = sorted(custom_results.keys())
our_acc = [100 * custom_results[c][0] / max(custom_results[c][1], 1) for c in cats_sorted]
gs_acc = [100 * gensim_results[c][0] / max(gensim_results[c][1], 1) for c in cats_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cats_sorted))
width = 0.38
ax.bar(x - width / 2, our_acc, width, label="Bizim Model (sıfırdan)", color="#534AB7")
ax.bar(x + width / 2, gs_acc, width, label="Gensim Word2Vec", color="#1D9E75")
ax.set_xticks(x)
ax.set_xticklabels(cats_sorted, rotation=40, ha="right")
ax.set_ylabel("Doğruluk (%)")
ax.set_title("Analoji Görevi — Kategori Bazlı Doğruluk Karşılaştırması")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_analogy_accuracy_comparison.png"), dpi=130)
plt.close(fig)

# 9b) t-SNE görselleştirmesi — kategorilere göre renklendirilmiş kelime kümeleri
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
            plot_words.append(w)
            plot_labels.append(group)

plot_vecs = np.array([norm_vectors[word2id[w]] for w in plot_words])
tsne = TSNE(n_components=2, perplexity=15, random_state=SEED, init="pca", max_iter=1500)
coords = tsne.fit_transform(plot_vecs)

fig, ax = plt.subplots(figsize=(11, 8.5))
palette = ["#534AB7", "#E24B4A", "#1D9E75", "#BA7517", "#3178C6", "#C2338B"]
for i, group in enumerate(tsne_groups.keys()):
    mask = [j for j, l in enumerate(plot_labels) if l == group]
    ax.scatter(coords[mask, 0], coords[mask, 1], color=palette[i % len(palette)], label=group, s=60)
    for j in mask:
        ax.annotate(plot_words[j], (coords[j, 0], coords[j, 1]), fontsize=9, alpha=0.85,
                     xytext=(4, 4), textcoords="offset points")
ax.set_title("t-SNE ile Kelime Vektörü Görselleştirmesi (kategoriye göre renklendirilmiş)")
ax.legend(loc="best")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_tsne_word_clusters.png"), dpi=130)
plt.close(fig)

# 9c) En yakın komşu örnekleri — görsel özet
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, w in zip(axes.flat, query_words):
    nn_list = neighbor_results.get(w)
    if not nn_list:
        ax.axis("off")
        continue
    names = [n for n, s in nn_list[:6]][::-1]
    scores = [s for n, s in nn_list[:6]][::-1]
    ax.barh(names, scores, color="#534AB7")
    ax.set_title(f'"{w}"', fontsize=12)
    ax.set_xlim(0, 1)
fig.suptitle("En Yakın Komşu Kelimeler (kosinüs benzerliği)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_nearest_neighbors.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")


# ──────────────────────────────────────────────────────────────────────────
# 10) Modeli ve kelime dağarcığını kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "in_embed": model.in_embed.weight.detach().cpu(),
    "out_embed": model.out_embed.weight.detach().cpu(),
    "word2id": word2id,
    "id2word": id2word,
    "emb_dim": EMB_DIM,
}, os.path.join(DATA_DIR, "word2vec_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE:,} | Eğitim çifti: {len(center_ids):,}")
print(f"Bizim model analoji doğruluğu : %{100*total_correct/max(total_n,1):.1f}")
print(f"Gensim analoji doğruluğu       : %{100*gensim_correct/max(gensim_n,1):.1f}")
print("Tamamlandı.")

"""
==============================================================================
 Self-Attention / Transformer — Karakter Düzeyinde Mini-GPT
==============================================================================

Tek bir Python dosyasında uçtan uca bir proje:
  1) Tiny Shakespeare veri setini indirir (Karpathy'nin ünlü char-rnn/nanoGPT
     veri seti — ~1.1M karakter, Shakespeare'in eserlerinden)
  2) Karakter düzeyinde bir vocab kurar (65 benzersiz karakter)
  3) "Attention is All You Need" (Vaswani ve ark., 2017) mimarisinin
     decoder-only halini — tıpkı GPT'nin kendisi gibi — PyTorch'ta SIFIRDAN
     kurar: multi-head self-attention, positional embedding, causal mask,
     feed-forward, residual bağlantılar, layer normalization
  4) Modeli sonraki-karakter tahmini ile eğitir
  5) Eğitilen modelle YENİ Shakespeare-vari metin üretir (otoregresif örnekleme)

RNN/LSTM/Attention-BiLSTM projelerinden temel farkı: bu modelde HİÇ tekrarlayan
(recurrent) katman yok. Bütün bir dizi, sadece self-attention ile paralel
olarak işleniyor — "Attention is All You Need" makalesinin ana iddiası tam
olarak bu: recurrence'a hiç ihtiyacınız yok.

Veri seti: Tiny Shakespeare (~1.1M karakter, 65 benzersiz karakter)
==============================================================================
"""

import os
import time
import urllib.request

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)

DATA_DIR = "data"; FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "shakespeare.txt")

BLOCK_SIZE = 64        # modelin bir seferde göreceği maksimum bağlam uzunluğu
N_EMBD = 64            # gömme (embedding) boyutu
N_HEAD = 4             # multi-head attention'daki kafa sayısı
N_LAYER = 3            # üst üste dizilen Transformer blok sayısı
DROPOUT = 0.1
BATCH_SIZE = 64
MAX_ITERS = 2500
EVAL_INTERVAL = 250
LR = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device} | CPU çekirdek: {torch.get_num_threads()}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Veriyi indir ve karakter düzeyinde tokenize et
# ──────────────────────────────────────────────────────────────────────────
def download_shakespeare():
    if os.path.exists(SHAKESPEARE_PATH):
        print(f"Veri zaten mevcut: {SHAKESPEARE_PATH}"); return
    print("Tiny Shakespeare indiriliyor...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)


download_shakespeare()
with open(SHAKESPEARE_PATH) as f:
    text = f.read()
print(f"\nToplam karakter: {len(text):,}")

chars = sorted(set(text))
VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(f"Benzersiz karakter (vocab): {VOCAB_SIZE} → {''.join(chars)!r}")


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Eğitim: {len(train_data):,} karakter | Doğrulama: {len(val_data):,} karakter")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# ──────────────────────────────────────────────────────────────────────────
# 2) Model — Decoder-only Transformer (sıfırdan)
# ──────────────────────────────────────────────────────────────────────────
class SelfAttentionHead(nn.Module):
    """
    Tek bir self-attention kafası. Her pozisyon kendi Query'sini, tüm
    pozisyonlar Key ve Value üretir. Bir pozisyonun diğerlerine "ne kadar
    dikkat edeceği", Query-Key benzerliğinden (skorundan) hesaplanır.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    CAUSAL MASK: dil modelleme geleceği göremez — bir karakter, kendisinden
    SONRA gelen karakterlere bakamaz (alt üçgen maske ile bu engellenir).
    Bu, encoder'daki self-attention'dan (BERT gibi) temel farkı.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        weights = q @ k.transpose(-2, -1) * (C ** -0.5)               # (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v                                             # (B,T,head_size)


class MultiHeadAttention(nn.Module):
    """Birden çok attention kafasını paralel çalıştırıp birleştirir — her
    kafa farklı bir ilişki türünü öğrenebilir (biri sözdizimi, biri anlam...)."""
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bir Transformer bloğu: self-attention + feed-forward, ikisi de
    residual (artık) bağlantı ve pre-layernorm ile sarılı."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # residual bağlantı 1
        x = x + self.ffwd(self.ln2(x))    # residual bağlantı 2
        return x


class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                              # (B,T,C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb    # RNN'in aksine, sıra bilgisi (pozisyon) AYRICA eklenmeli
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                          # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]                # sadece son BLOCK_SIZE karakteri kullan
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # son zaman adımının tahmini
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(20)
        for k in range(20):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3) Eğitim döngüsü
# ──────────────────────────────────────────────────────────────────────────
print("\n── Eğitim başlıyor ──")
history = {"iter": [], "train_loss": [], "val_loss": []}
t_start = time.time()

for it in range(MAX_ITERS + 1):
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        losses = estimate_loss()
        history["iter"].append(it)
        history["train_loss"].append(losses["train"])
        history["val_loss"].append(losses["val"])
        elapsed = time.time() - t_start
        print(f"Adım {it:5d}/{MAX_ITERS} | train_loss={losses['train']:.4f} | "
              f"val_loss={losses['val']:.4f} | {elapsed:.0f}s")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Eğitim tamamlandı: {time.time()-t_start:.0f}s")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history["iter"], history["train_loss"], label="Eğitim", color="#534AB7")
ax.plot(history["iter"], history["val_loss"], label="Doğrulama", color="#E24B4A")
ax.set_xlabel("Adım"); ax.set_ylabel("Cross-Entropy Kaybı")
ax.set_title("Mini-GPT Eğitim Kaybı (Karakter Düzeyinde Tahmin)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_training_loss.png"), dpi=130)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 4) Metin Üretimi — Eğitilen Modelle Yeni "Shakespeare" Yaz
# ──────────────────────────────────────────────────────────────────────────
print("\n── Üretilen Metin Örnekleri ──")
model.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)   # boş bağlamla başla
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
generated_text = decode(generated_ids)
print(generated_text)

# Belirli bir başlangıç ile devam ettirme (prompt tamamlama)
prompt = "ROMEO:"
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
continued_ids = model.generate(prompt_ids, max_new_tokens=300)[0].tolist()
continued_text = decode(continued_ids)
print("\n--- Prompt tamamlama ---")
print(continued_text)

with open(os.path.join(DATA_DIR, "generated_samples.txt"), "w") as f:
    f.write("=== Boş bağlamdan üretim ===\n")
    f.write(generated_text)
    f.write("\n\n=== 'ROMEO:' prompt'undan devam ===\n")
    f.write(continued_text)


# ──────────────────────────────────────────────────────────────────────────
# 5) Attention Görselleştirmesi — Bir Kafa Neye Bakıyor?
# ──────────────────────────────────────────────────────────────────────────
# Eğitilen modelin ilk katmanındaki ilk attention kafasının, bir örnek
# dizi üzerindeki dikkat matrisini çıkarıp görselleştiriyoruz.
sample_text = "ROMEO: But soft, what light"
sample_ids = torch.tensor([encode(sample_text)], dtype=torch.long, device=device)

captured = {}
def hook(module, input, output):
    pass

# Attention ağırlıklarını manuel olarak yeniden hesapla (ilk blok, ilk kafa)
with torch.no_grad():
    x = model.token_embedding(sample_ids) + model.position_embedding(
        torch.arange(sample_ids.shape[1], device=device))
    x_norm = model.blocks[0].ln1(x)
    head = model.blocks[0].sa.heads[0]
    k = head.key(x_norm); q = head.query(x_norm)
    T = sample_ids.shape[1]
    weights = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
    weights = weights.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    weights = F.softmax(weights, dim=-1)[0].numpy()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(weights, cmap="Purples")
ax.set_xticks(range(len(sample_text))); ax.set_xticklabels(list(sample_text), fontsize=9)
ax.set_yticks(range(len(sample_text))); ax.set_yticklabels(list(sample_text), fontsize=9)
ax.set_xlabel("Dikkat Edilen Karakter (Key)"); ax.set_ylabel("Sorgulayan Karakter (Query)")
ax.set_title("Self-Attention Matrisi (1. Blok, 1. Kafa)\nAlt üçgen: nedensel (causal) maske")
plt.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_self_attention_matrix.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 6) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "stoi": stoi, "itos": itos,
    "block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
}, os.path.join(DATA_DIR, "minigpt_model.pt"))

print("\n── Özet ──")
print(f"Vocab: {VOCAB_SIZE} | Parametre: {sum(p.numel() for p in model.parameters()):,}")
print(f"Son eğitim kaybı: {history['train_loss'][-1]:.4f} | Son doğrulama kaybı: {history['val_loss'][-1]:.4f}")
print("Tamamlandı.")

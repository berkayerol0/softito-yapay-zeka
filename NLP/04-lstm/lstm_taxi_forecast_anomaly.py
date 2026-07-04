"""
==============================================================================
 NYC Taksi Talebi — LSTM ile Zaman Serisi Tahmini ve Olay Tabanlı Anomali Tespiti
==============================================================================

Tek bir Python dosyasında uçtan uca bir proje:
  1) Veriyi indirir (Numenta Anomaly Benchmark — NYC Taxi)
  2) Keşifsel görseller üretir
  3) Sadece "normal" döneme LSTM tahminci (forecaster) eğitir
  4) Modeli tüm zaman çizelgesine uygulayıp tahmin hatasından (residual)
     bir anomali skoru türetir
  5) Bulunan anomalileri NAB'in gerçek etiketli olaylarıyla (maraton,
     Şükran Günü, Noel, Yılbaşı, kar fırtınası) karşılaştırır
  6) Sonuç görsellerini ve metrikleri raporlar

Veri seti: Numenta Anomaly Benchmark (NAB) — nyc_taxi.csv
  30 dakikalık aralıklarla NYC taksi yolculuk sayısı, 2014-07-01 / 2015-01-31
  5 bilinen anomali penceresi: NYC Maratonu, Şükran Günü, Noel, Yılbaşı, kar fırtınası
==============================================================================
"""

import os
import json
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────────────────────
# 0) Sabitler / Hiperparametreler
# ──────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "data"
FIG_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv"
LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"
DATA_PATH = os.path.join(DATA_DIR, "nyc_taxi.csv")
LABELS_PATH = os.path.join(DATA_DIR, "combined_windows.json")

SEQ_LEN = 48          # geçmiş 48 adım = 1 gün (30 dk aralıklarla)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
PATIENCE = 8           # early stopping sabrı
VAL_FRACTION = 0.15    # normal verinin son %15'i doğrulama seti
Z_THRESHOLD = 3.0       # anomali skoru eşiği
ROLLING_WINDOW = 96     # anomali skoru için yuvarlanan pencere (2 gün)

EVENT_NAMES = [
    "NYC Maratonu",
    "Şükran Günü",
    "Noel",
    "Yılbaşı",
    "Kar Fırtınası",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {device}")


# ──────────────────────────────────────────────────────────────────────────
# 1) Veriyi indir ve yükle
# ──────────────────────────────────────────────────────────────────────────
def download_if_missing(url: str, path: str):
    if not os.path.exists(path):
        print(f"İndiriliyor: {url}")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"Zaten mevcut: {path}")


download_if_missing(DATA_URL, DATA_PATH)
download_if_missing(LABELS_URL, LABELS_PATH)

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

with open(LABELS_PATH) as f:
    all_windows = json.load(f)
raw_windows = all_windows["realKnownCause/nyc_taxi.csv"]
anomaly_windows = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in raw_windows]

print(f"\nVeri seti: {len(df)} satır | {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"Bilinen anomali penceresi sayısı: {len(anomaly_windows)}")
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    print(f"  - {name:<16} {s} → {e}")


# ──────────────────────────────────────────────────────────────────────────
# 2) Özellik mühendisliği
# ──────────────────────────────────────────────────────────────────────────
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
df["dow"] = df["timestamp"].dt.dayofweek

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

# Her satırın bilinen bir anomali penceresine düşüp düşmediğini işaretle
def in_any_window(ts):
    return any(s <= ts <= e for s, e in anomaly_windows)

df["is_known_anomaly"] = df["timestamp"].apply(in_any_window)

# İlk anomali penceresinden önceki kısım = güvenli "normal" eğitim verisi
first_anomaly_start = anomaly_windows[0][0]
normal_cutoff_idx = int((df["timestamp"] < first_anomaly_start).sum())
print(f"\nİlk anomaliden önceki temiz satır sayısı: {normal_cutoff_idx} / {len(df)}")

# Güvenlik payı: pencereye çok yaklaşmamak için son SEQ_LEN kadar geri çekil
safe_normal_end = normal_cutoff_idx - SEQ_LEN
val_start = int(safe_normal_end * (1 - VAL_FRACTION))

print(f"Eğitim aralığı : 0 → {val_start}")
print(f"Doğrulama aralığı: {val_start} → {safe_normal_end}")
print(f"Test/izleme aralığı (tüm anomalileri içerir): {safe_normal_end} → {len(df)}")


# ──────────────────────────────────────────────────────────────────────────
# 3) Keşifsel görsel
# ──────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["timestamp"], df["value"], color="#534AB7", lw=0.8, alpha=0.85, label="Taksi yolcu sayısı")
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    ax.axvspan(s, e, color="#E24B4A", alpha=0.25)
ax.axvline(df["timestamp"].iloc[normal_cutoff_idx], color="#1D9E75", ls="--", lw=1.5, label="Eğitim/izleme sınırı")
ax.set_title("NYC Taksi Talebi — Bilinen Anomali Pencereleri (kırmızı)", fontsize=13)
ax.set_xlabel("Tarih")
ax.set_ylabel("30 dk'lık yolcu sayısı")
ax.legend(loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "01_eda_overview.png"), dpi=130)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 4) Ölçekleme (sadece eğitim verisiyle fit edilir — veri sızıntısı yok)
# ──────────────────────────────────────────────────────────────────────────
value_scaler = MinMaxScaler()
value_scaler.fit(df["value"].values[:val_start].reshape(-1, 1))
df["value_scaled"] = value_scaler.transform(df["value"].values.reshape(-1, 1))

FEATURE_COLS = ["value_scaled", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
feature_matrix = df[FEATURE_COLS].values.astype(np.float32)
target_vector = df["value_scaled"].values.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# 5) Kayan pencere veri seti (Sliding Window)
# ──────────────────────────────────────────────────────────────────────────
class SlidingWindowDataset(Dataset):
    """
    X[i] = feature_matrix[i : i+SEQ_LEN]           (geçmiş SEQ_LEN adım)
    y[i] = target_vector[i+SEQ_LEN]                 (bir sonraki adımın değeri)
    Sadece [start, end) satır aralığındaki pencereler kullanılır.
    """
    def __init__(self, features, targets, start, end, seq_len=SEQ_LEN):
        self.features = features
        self.targets = targets
        self.seq_len = seq_len
        # i + seq_len, [start, end) aralığında olmalı
        self.indices = list(range(max(start, 0), min(end, len(targets)) - seq_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.features[i: i + self.seq_len]
        y = self.targets[i + self.seq_len]
        return torch.tensor(x), torch.tensor(y)


train_ds = SlidingWindowDataset(feature_matrix, target_vector, 0, val_start)
val_ds = SlidingWindowDataset(feature_matrix, target_vector, val_start, safe_normal_end)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nEğitim pencere sayısı: {len(train_ds)} | Doğrulama pencere sayısı: {len(val_ds)}")


# ──────────────────────────────────────────────────────────────────────────
# 6) Model — LSTM Tahminci
# ──────────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    Çok değişkenli girdi (değer + zaman özellikleri) → tek adım ileri tahmin.
    nn.RNN yerine nn.LSTM: unutma kapısı sayesinde uzun vadeli (günlük/haftalık)
    döngüleri kaybolan gradyan olmadan öğrenebilir.
    """
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # output: (batch, seq, hidden) | h_n, c_n: (num_layers, batch, hidden)
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]          # son katmanın son zaman adımı
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


model = LSTMForecaster(input_size=len(FEATURE_COLS)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

print(f"\nModel parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")


# ──────────────────────────────────────────────────────────────────────────
# 7) Eğitim döngüsü (early stopping + gradient clipping)
# ──────────────────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


history = {"train_loss": [], "val_loss": []}
best_val = float("inf")
best_state = None
patience_counter = 0

print("\n── Eğitim başlıyor ──")
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    improved = val_loss < best_val - 1e-6
    if improved:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch % 5 == 0 or epoch == 1 or improved:
        flag = " ✓" if improved else ""
        print(f"Epoch {epoch:3d}/{EPOCHS} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}{flag}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping: {PATIENCE} epoch boyunca iyileşme yok (epoch {epoch}).")
        break

model.load_state_dict(best_state)
print(f"En iyi doğrulama kaybı: {best_val:.6f}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(history["train_loss"], label="Eğitim kaybı", color="#534AB7")
ax.plot(history["val_loss"], label="Doğrulama kaybı", color="#E24B4A")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")
ax.set_title("Eğitim Geçmişi")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "02_training_history.png"), dpi=130)
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# 8) Tüm zaman çizelgesinde tahmin üret (tek adımlı, gerçek geçmişle)
# ──────────────────────────────────────────────────────────────────────────
model.eval()
full_ds = SlidingWindowDataset(feature_matrix, target_vector, 0, len(df))
full_loader = DataLoader(full_ds, batch_size=256, shuffle=False)

all_preds = []
with torch.no_grad():
    for xb, _ in full_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        all_preds.append(preds)
all_preds = np.concatenate(all_preds)

# tahminler indeks SEQ_LEN'den itibaren başlar
pred_index = np.arange(SEQ_LEN, SEQ_LEN + len(all_preds))
df["pred_scaled"] = np.nan
df.loc[pred_index, "pred_scaled"] = all_preds

df["pred_value"] = value_scaler.inverse_transform(
    df["pred_scaled"].fillna(0).values.reshape(-1, 1)
).flatten()
df.loc[df["pred_scaled"].isna(), "pred_value"] = np.nan

# Tahmin kalite metrikleri — sadece eğitimde hiç görülmemiş NORMAL test bölgesinde
eval_mask = (df.index >= safe_normal_end) & (~df["is_known_anomaly"]) & df["pred_value"].notna()
mae = np.mean(np.abs(df.loc[eval_mask, "value"] - df.loc[eval_mask, "pred_value"]))
rmse = np.sqrt(np.mean((df.loc[eval_mask, "value"] - df.loc[eval_mask, "pred_value"]) ** 2))
mape = np.mean(np.abs((df.loc[eval_mask, "value"] - df.loc[eval_mask, "pred_value"]) / df.loc[eval_mask, "value"])) * 100
print(f"\n── Tahmin kalitesi (görülmemiş normal bölge) ──")
print(f"MAE  : {mae:.1f} yolcu")
print(f"RMSE : {rmse:.1f} yolcu")
print(f"MAPE : %{mape:.2f}")


# ──────────────────────────────────────────────────────────────────────────
# 9) Anomali skoru — tahmin hatasının yuvarlanan z-skoru
# ──────────────────────────────────────────────────────────────────────────
df["residual"] = df["value"] - df["pred_value"]
roll_mean = df["residual"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).mean()
roll_std = df["residual"].rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2).std()
df["anomaly_score"] = (df["residual"] - roll_mean) / (roll_std + 1e-6)
df["flagged"] = df["anomaly_score"].abs() > Z_THRESHOLD

# yalnızca izleme bölgesini değerlendiriyoruz (eğitim/doğrulama "görüldü")
monitor_mask = df.index >= safe_normal_end
flagged_points = df.loc[monitor_mask & df["flagged"], "timestamp"]


# ──────────────────────────────────────────────────────────────────────────
# 10) Olay bazlı değerlendirme — bilinen pencerelerle karşılaştır
# ──────────────────────────────────────────────────────────────────────────
print("\n── Anomali Tespiti Sonuçları (Olay Bazlı) ──")
detected_count = 0
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    hits = flagged_points[(flagged_points >= s) & (flagged_points <= e)]
    detected = len(hits) > 0
    detected_count += int(detected)
    status = "✅ YAKALANDI" if detected else "❌ KAÇIRILDI"
    print(f"  {name:<16} {status}  ({len(hits)} işaretli nokta)")

recall = detected_count / len(anomaly_windows)
print(f"\nOlay bazlı yakalama oranı (recall): {detected_count}/{len(anomaly_windows)} = %{recall*100:.0f}")

# Yanlış alarm oranı: bilinen pencerelerin dışında işaretlenen nokta oranı
outside_mask = monitor_mask & (~df["is_known_anomaly"])
false_positive_rate = df.loc[outside_mask, "flagged"].mean() * 100
print(f"Normal bölgede yanlış alarm oranı: %{false_positive_rate:.2f}")


# ──────────────────────────────────────────────────────────────────────────
# 11) Görseller — tahmin vs gerçek, anomali skoru zaman çizelgesi
# ──────────────────────────────────────────────────────────────────────────
zoom_start = df["timestamp"].iloc[safe_normal_end]
plot_df = df[df["timestamp"] >= zoom_start]

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1, 1.2]})

axes[0].plot(plot_df["timestamp"], plot_df["value"], color="0.3", lw=0.8, label="Gerçek")
axes[0].plot(plot_df["timestamp"], plot_df["pred_value"], color="#534AB7", lw=0.9, alpha=0.85, label="LSTM Tahmini")
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    axes[0].axvspan(s, e, color="#E24B4A", alpha=0.20)
axes[0].set_ylabel("Yolcu sayısı")
axes[0].set_title("İzleme Bölgesi: Gerçek vs LSTM Tahmini (kırmızı = bilinen anomali)", fontsize=12)
axes[0].legend(loc="upper left")

axes[1].plot(plot_df["timestamp"], plot_df["residual"], color="#BA7517", lw=0.8)
axes[1].axhline(0, color="black", lw=0.6)
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    axes[1].axvspan(s, e, color="#E24B4A", alpha=0.20)
axes[1].set_ylabel("Hata (gerçek - tahmin)")

axes[2].plot(plot_df["timestamp"], plot_df["anomaly_score"], color="#1D9E75", lw=0.8)
axes[2].axhline(Z_THRESHOLD, color="#E24B4A", ls="--", lw=1.2, label=f"Eşik (z={Z_THRESHOLD})")
axes[2].axhline(-Z_THRESHOLD, color="#E24B4A", ls="--", lw=1.2)
for (s, e), name in zip(anomaly_windows, EVENT_NAMES):
    axes[2].axvspan(s, e, color="#E24B4A", alpha=0.20)
flagged_in_zoom = plot_df[plot_df["flagged"]]
axes[2].scatter(flagged_in_zoom["timestamp"], flagged_in_zoom["anomaly_score"],
                 color="#E24B4A", s=14, zorder=5, label="İşaretlenen nokta")
axes[2].set_ylabel("Anomali Skoru (z)")
axes[2].set_xlabel("Tarih")
axes[2].legend(loc="upper left")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "03_forecast_and_anomaly_score.png"), dpi=130)
plt.close(fig)

# Olay bazlı özet bar grafiği
fig, ax = plt.subplots(figsize=(8, 4.5))
counts = []
for (s, e) in anomaly_windows:
    hits = flagged_points[(flagged_points >= s) & (flagged_points <= e)]
    counts.append(len(hits))
colors_bar = ["#1D9E75" if c > 0 else "#E24B4A" for c in counts]
ax.bar(EVENT_NAMES, counts, color=colors_bar)
ax.set_ylabel("İşaretlenen nokta sayısı")
ax.set_title(f"Olay Bazlı Tespit Özeti — Yakalama Oranı: %{recall*100:.0f}")
plt.xticks(rotation=20, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "04_event_detection_summary.png"), dpi=130)
plt.close(fig)

print(f"\nGörseller '{FIG_DIR}/' klasörüne kaydedildi.")

# ──────────────────────────────────────────────────────────────────────────
# 12) Modeli kaydet
# ──────────────────────────────────────────────────────────────────────────
torch.save({
    "model_state": model.state_dict(),
    "feature_cols": FEATURE_COLS,
    "seq_len": SEQ_LEN,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
}, os.path.join(DATA_DIR, "lstm_forecaster.pt"))

print("\n── Özet ──")
print(f"MAE: {mae:.1f} | RMSE: {rmse:.1f} | MAPE: %{mape:.2f}")
print(f"Olay yakalama oranı: %{recall*100:.0f} | Yanlış alarm oranı: %{false_positive_rate:.2f}")
print("Tamamlandı.")

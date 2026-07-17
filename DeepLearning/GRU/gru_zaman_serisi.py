import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  GRU İLE ZAMAN SERİSİ TAHMİNİ (Ürün Talebi Demo)")
print("=" * 55)

# 1. Sentetik Zaman Serisi Üretimi — trend + haftalık mevsimsellik + gürültü
GUN_SAYISI = 500
gunler = np.arange(GUN_SAYISI)
trend = 0.05 * gunler
mevsimsellik = 8 * np.sin(2 * np.pi * gunler / 7)  # haftalık döngü
gurultu = np.random.normal(0, 2, GUN_SAYISI)
seri = 50 + trend + mevsimsellik + gurultu
seri = seri.astype(np.float32)

# 2. Normalize etme (min-max, 0-1 aralığı)
seri_min, seri_max = seri.min(), seri.max()
seri_norm = (seri - seri_min) / (seri_max - seri_min)


def normali_geri_don(deger):
    return deger * (seri_max - seri_min) + seri_min


# 3. Pencereleme — geçmiş PENCERE gün ile bir sonraki günü tahmin etme
PENCERE = 14

def pencerelere_ayir(veri, pencere):
    X, y = [], []
    for i in range(len(veri) - pencere):
        X.append(veri[i:i + pencere])
        y.append(veri[i + pencere])
    return np.array(X), np.array(y)


X, y = pencerelere_ayir(seri_norm, PENCERE)

# 4. Eğitim/Test ayrımı — zaman serisinde rastgele değil, KRONOLOJİK bölünür
egitim_orani = 0.8
bolme_noktasi = int(len(X) * egitim_orani)
X_egitim, X_test = X[:bolme_noktasi], X[bolme_noktasi:]
y_egitim, y_test = y[:bolme_noktasi], y[bolme_noktasi:]

X_egitim_t = torch.tensor(X_egitim).unsqueeze(-1)  # (örnek, pencere, 1)
y_egitim_t = torch.tensor(y_egitim).unsqueeze(-1)
X_test_t = torch.tensor(X_test).unsqueeze(-1)
y_test_t = torch.tensor(y_test).unsqueeze(-1)

print(f"\n[BİLGİ] Toplam {len(X)} pencere: {len(X_egitim)} eğitim, {len(X_test)} test.")


# 5. GRU Modeli
class ZamanSerisiGRU(nn.Module):
    def __init__(self, gizli_boyut=32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=gizli_boyut, batch_first=True)
        self.cikis = nn.Linear(gizli_boyut, 1)

    def forward(self, x):
        _, son_gizli_durum = self.gru(x)
        return self.cikis(son_gizli_durum.squeeze(0))


model = ZamanSerisiGRU()
kayip_fonksiyonu = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Eğitim
EPOCH_SAYISI = 150
kayip_gecmisi = []

for epoch in range(1, EPOCH_SAYISI + 1):
    model.train()
    optimizer.zero_grad()
    tahmin = model(X_egitim_t)
    kayip = kayip_fonksiyonu(tahmin, y_egitim_t)
    kayip.backward()
    optimizer.step()
    kayip_gecmisi.append(kayip.item())
    if epoch % 25 == 0:
        print(f"[EPOCH {epoch}/{EPOCH_SAYISI}] Loss: {kayip.item():.5f}")

# 7. Loss grafiği ve CSV'si
csv_yolu = os.path.join(OUTPUT_DIR, "egitim_loss.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss"])
    for i, l in enumerate(kayip_gecmisi, start=1):
        writer.writerow([i, l])
print(f"\n[SİSTEM] Loss verisi kaydedildi: {csv_yolu}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCH_SAYISI + 1), kayip_gecmisi, color="darkorange")
plt.title("GRU Eğitim Kaybı (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
grafik_yolu = os.path.join(OUTPUT_DIR, "egitim_loss.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Loss grafiği kaydedildi: {grafik_yolu}")

# 8. Test seti üzerinde tahmin ve değerlendirme
model.eval()
with torch.no_grad():
    test_tahmin = model(X_test_t).squeeze().numpy()

gercek_deger = normali_geri_don(y_test)
tahmin_deger = normali_geri_don(test_tahmin)

rmse = float(np.sqrt(np.mean((gercek_deger - tahmin_deger) ** 2)))
mae = float(np.mean(np.abs(gercek_deger - tahmin_deger)))
print(f"\n[SONUÇ] Test RMSE: {rmse:.3f} | Test MAE: {mae:.3f}")

# 9. Gerçek vs Tahmin CSV'si ve grafiği
tahmin_csv_yolu = os.path.join(OUTPUT_DIR, "gercek_vs_tahmin.csv")
with open(tahmin_csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["gun_index", "gercek", "tahmin"])
    for i, (g, t) in enumerate(zip(gercek_deger, tahmin_deger)):
        writer.writerow([i, g, t])
print(f"[SİSTEM] Gerçek/tahmin verisi kaydedildi: {tahmin_csv_yolu}")

plt.figure(figsize=(10, 5))
plt.plot(gercek_deger, label="Gerçek", color="steelblue", linewidth=2)
plt.plot(tahmin_deger, label="GRU Tahmini", color="darkorange", linestyle="--", linewidth=2)
plt.title(f"Test Seti: Gerçek vs Tahmin (RMSE={rmse:.2f}, MAE={mae:.2f})")
plt.xlabel("Test Gün İndeksi")
plt.ylabel("Talep")
plt.legend()
plt.grid(True)
tahmin_grafik_yolu = os.path.join(OUTPUT_DIR, "gercek_vs_tahmin.png")
plt.savefig(tahmin_grafik_yolu)
plt.close()
print(f"[SİSTEM] Gerçek/tahmin grafiği kaydedildi: {tahmin_grafik_yolu}")

print(
    "\n(Not: Veri tamamen sentetik (trend + haftalık mevsimsellik + gürültü) üretildi, "
    "gerçek bir veri kaynağına bağımlılık yok. Eğitim/test ayrımı kronolojiktir — zaman "
    "serilerinde rastgele bölme veri sızıntısına (gelecekten geçmişe bilgi akışı) yol açar, "
    "bu yüzden kasıtlı olarak kullanılmadı.)"
)

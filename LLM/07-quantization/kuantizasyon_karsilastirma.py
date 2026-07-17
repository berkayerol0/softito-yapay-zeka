import os
import io
import csv
import copy
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.pytorch_utils import Conv1D

SEED = 42
set_seed(SEED)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  FP32 / FP16 / INT8 KUANTİZASYON KARŞILAŞTIRMASI")
print("=" * 55)

model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
test_prompt = "Yapay zeka, günümüzde"
girdi = tokenizer(test_prompt, return_tensors="pt")


def model_boyutu_mb(model) -> float:
    """Modelin gerçek serileştirilmiş boyutunu MB cinsinden hesaplar. .parameters()
    toplamı yerine state_dict()'i belleğe kaydedip ölçüyoruz — kuantize edilmiş
    katmanların ağırlıkları normal nn.Parameter değil, özel "packed" tensörler olarak
    saklanır ve .parameters() bunları güvenilir şekilde saymaz."""
    tampon = io.BytesIO()
    torch.save(model.state_dict(), tampon)
    return tampon.getbuffer().nbytes / (1024 ** 2)


def conv1d_katmanlarini_linear_yap(model):
    """GPT-2 ailesi (distilgpt2 dahil) attention/mlp projeksiyonlarında torch.nn.Linear
    DEĞİL, transformers'a özgü Conv1D katmanlarını kullanır. torch.quantization.
    quantize_dynamic yalnızca nn.Linear'ı tanır; bu yüzden Conv1D katmanlarını
    matematiksel olarak eşdeğer nn.Linear'a dönüştürüyoruz (ağırlığı transpoze ederek)."""
    for isim, alt_modul in model.named_children():
        if isinstance(alt_modul, Conv1D):
            girdi_boyutu, cikti_boyutu = alt_modul.weight.shape
            linear = torch.nn.Linear(girdi_boyutu, cikti_boyutu, bias=alt_modul.bias is not None)
            with torch.no_grad():
                linear.weight.copy_(alt_modul.weight.t())
                if alt_modul.bias is not None:
                    linear.bias.copy_(alt_modul.bias)
            setattr(model, isim, linear)
        else:
            conv1d_katmanlarini_linear_yap(alt_modul)
    return model


def cikarim_yap(model, tekrar=5):
    """Aynı prompt'u birkaç kez üretip ortalama çıkarım süresini ve son çıktıyı döner."""
    sureler = []
    metin = ""
    for _ in range(tekrar):
        baslangic = time.time()
        with torch.no_grad():
            cikti = model.generate(
                **girdi, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id, do_sample=False
            )
        sureler.append(time.time() - baslangic)
        metin = tokenizer.decode(cikti[0], skip_special_tokens=True)
    return sum(sureler) / len(sureler), metin


sonuclar = []

# 1. FP32 (varsayılan hassasiyet)
print("\n[BİLGİ] FP32 modeli test ediliyor...")
model_fp32 = AutoModelForCausalLM.from_pretrained(model_id)
model_fp32.eval()
boyut_fp32 = model_boyutu_mb(model_fp32)
sure_fp32, cikti_fp32 = cikarim_yap(model_fp32)
sonuclar.append({"hassasiyet": "FP32", "boyut_mb": round(boyut_fp32, 2),
                  "ortalama_sure_sn": round(sure_fp32, 4), "cikti": cikti_fp32})
print(f"  Boyut: {boyut_fp32:.2f} MB | Süre: {sure_fp32:.4f} sn")

# 2. FP16 (yarı hassasiyet)
print("\n[BİLGİ] FP16 modeli test ediliyor...")
model_fp16 = copy.deepcopy(model_fp32).half()
boyut_fp16 = model_boyutu_mb(model_fp16)
try:
    sure_fp16, cikti_fp16 = cikarim_yap(model_fp16)
except RuntimeError as e:
    # CPU'da bazı işlemler FP16'yı desteklemeyebilir; bu durumda hatayı görünür kılıyoruz.
    print(f"  [UYARI] FP16 çıkarımı CPU'da başarısız oldu: {e}")
    sure_fp16, cikti_fp16 = None, f"HATA: {e}"
sonuclar.append({"hassasiyet": "FP16", "boyut_mb": round(boyut_fp16, 2),
                  "ortalama_sure_sn": round(sure_fp16, 4) if sure_fp16 else None, "cikti": cikti_fp16})
print(f"  Boyut: {boyut_fp16:.2f} MB" + (f" | Süre: {sure_fp16:.4f} sn" if sure_fp16 else ""))

# 3. INT8 (dinamik kuantizasyon — sadece Linear katmanlar, CPU'da native destekli)
print("\n[BİLGİ] INT8 (dinamik kuantizasyon) modeli test ediliyor...")
model_linear = conv1d_katmanlarini_linear_yap(copy.deepcopy(model_fp32))
model_int8 = torch.quantization.quantize_dynamic(
    model_linear, {torch.nn.Linear}, dtype=torch.qint8
)
boyut_int8 = model_boyutu_mb(model_int8)
sure_int8, cikti_int8 = cikarim_yap(model_int8)
sonuclar.append({"hassasiyet": "INT8", "boyut_mb": round(boyut_int8, 2),
                  "ortalama_sure_sn": round(sure_int8, 4), "cikti": cikti_int8})
print(f"  Boyut: {boyut_int8:.2f} MB | Süre: {sure_int8:.4f} sn")

# 4. Sonuçları CSV'ye kaydet
csv_yolu = os.path.join(OUTPUT_DIR, "kuantizasyon_karsilastirma.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["hassasiyet", "boyut_mb", "ortalama_sure_sn", "cikti"])
    writer.writeheader()
    writer.writerows(sonuclar)
print(f"\n[SİSTEM] Karşılaştırma tablosu kaydedildi: {csv_yolu}")

# 5. Boyut ve süre karşılaştırma grafikleri
etiketler = [s["hassasiyet"] for s in sonuclar]
boyutlar = [s["boyut_mb"] for s in sonuclar]
sureler_liste = [s["ortalama_sure_sn"] if s["ortalama_sure_sn"] else 0 for s in sonuclar]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(etiketler, boyutlar, color=["steelblue", "seagreen", "indianred"])
ax1.set_title("Model Boyutu (MB)")
ax1.set_ylabel("MB")
ax1.grid(axis="y")

ax2.bar(etiketler, sureler_liste, color=["steelblue", "seagreen", "indianred"])
ax2.set_title("Ortalama Çıkarım Süresi (5 tekrar)")
ax2.set_ylabel("Saniye")
ax2.grid(axis="y")

plt.tight_layout()
grafik_yolu = os.path.join(OUTPUT_DIR, "kuantizasyon_karsilastirma.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Karşılaştırma grafiği kaydedildi: {grafik_yolu}")

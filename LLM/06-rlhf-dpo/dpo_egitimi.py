import os
import csv
import tempfile
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import DPOConfig, DPOTrainer

# datasets kütüphanesinin disk önbellekleme mekanizması Windows'ta bazen bir yarış
# durumuna (race condition) girip "cache dosyası bulunamadı" hatası veriyor. Önbelleklemeyi
# kapatıp her şeyi bellekte tutmak bu sorunu kökten çözer; bu küçük veri setinde performans
# kaybı da yaşanmaz.
disable_caching()

SEED = 42
set_seed(SEED)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  BASİTLEŞTİRİLMİŞ DPO (Direct Preference Optimization) DEMO")
print("=" * 55)

# 1. Tercih Veri Seti — her satır bir prompt + tercih edilen/edilmeyen cevap çifti
# Amaç: modele "kibar ve yardımcı" cevabı, "kaba/yardımsız" cevaba tercih etmeyi öğretmek.
data = {
    "prompt": [
        "Müşteri: Ürünüm bozuk geldi, ne yapmalıyım?\nDestek:",
        "Müşteri: Param ne zaman iade edilecek?\nDestek:",
        "Müşteri: Siparişim hala gelmedi, çok sinirliyim.\nDestek:",
        "Müşteri: Bu ürünü nasıl kuracağım anlamadım.\nDestek:",
    ],
    "chosen": [
        " Bu durumu yaşadığınız için üzgünüm, hemen ilgileneyim. Ürününüzün iade veya değişim sürecini başlatabilirim.",
        " Talebinizi anlıyorum. İade süreci genellikle 3-5 iş günü içinde tamamlanır, takibini yapıyorum.",
        " Yaşadığınız gecikme için özür dilerim, kargo durumunu hemen kontrol ediyorum.",
        " Elbette, kurulum adımlarını sizinle birlikte adım adım geçelim.",
    ],
    "rejected": [
        " Bilmiyorum, kılavuza bakın.",
        " Sistemde öyle bir bilgi yok, bekleyin işte.",
        " Herkes aynı şeyi söylüyor, elimden bir şey gelmez.",
        " O kadar karışık değil, kendiniz uğraşın.",
    ],
}
dataset = Dataset.from_dict(data)

# 2. Model ve Tokenizer
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. ÖNCESİ TESTİ
test_prompt = "Müşteri: Ürünüm hiç çalışmıyor, çok kızgınım.\nDestek:"
girdi = tokenizer(test_prompt, return_tensors="pt")
with torch.no_grad():
    once_ciktisi = model.generate(
        **girdi, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False
    )
once_cevap = tokenizer.decode(once_ciktisi[0], skip_special_tokens=True)
print(f"\n[ÖNCESİ] {once_cevap}")

# 4. DPO Eğitimi
dpo_config = DPOConfig(
    output_dir=os.path.join(tempfile.gettempdir(), "dpo_checkpoints"),
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=2,  # Hızlı, gerçek bir sonuç almak için minimuma indirildi
    beta=0.1,  # DPO kayıp fonksiyonundaki referans-modelden-sapma katsayısı
    logging_steps=5,
    seed=SEED,
    use_cpu=not torch.cuda.is_available(),  # NVIDIA/CUDA yoksa (ör. AMD kartlar) CPU'yu açıkça belirt
    max_length=96,  # Prompt+cevap toplam token sınırı — gereksiz uzun pad'lemeyi önler
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("\n[BİLGİ] DPO Eğitimi Başlıyor...")
trainer.train()

# 5. Loss Grafiği ve CSV'si
log_history = trainer.state.log_history
loss_degerleri = [log["loss"] for log in log_history if "loss" in log]
adimlar = [log["step"] for log in log_history if "loss" in log]

csv_yolu = os.path.join(OUTPUT_DIR, "dpo_loss_log.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    for s, l in zip(adimlar, loss_degerleri):
        writer.writerow([s, l])
print(f"\n[SİSTEM] Loss verisi kaydedildi: {csv_yolu}")

plt.figure(figsize=(8, 5))
plt.plot(adimlar, loss_degerleri, marker='o', linestyle='-', color='purple')
plt.title("DPO Eğitim Kaybı")
plt.xlabel("Adım (Step)")
plt.ylabel("Kayıp (Loss)")
plt.grid(True)
grafik_yolu = os.path.join(OUTPUT_DIR, "dpo_loss_curve.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Eğitim grafiği kaydedildi: {grafik_yolu}")

# 6. SONRASI TESTİ
with torch.no_grad():
    sonra_ciktisi = model.generate(
        **girdi, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False
    )
sonra_cevap = tokenizer.decode(sonra_ciktisi[0], skip_special_tokens=True)

print("\n" + "=" * 55)
print("  ÖNCESİ / SONRASI KARŞILAŞTIRMA")
print("=" * 55)
print(f"GİRDİ:\n{test_prompt}")
print(f"[ÖNCESİ - DPO'suz]:\n{once_cevap}")
print(f"[SONRASI - DPO'lu]:\n{sonra_cevap}")

karsilastirma_yolu = os.path.join(OUTPUT_DIR, "oncesi_sonrasi_karsilastirma.csv")
with open(karsilastirma_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["asama", "cikti"])
    writer.writerow(["oncesi_dpo_yok", once_cevap])
    writer.writerow(["sonrasi_dpo_uygulandi", sonra_cevap])
print(f"\n[SİSTEM] Öncesi/sonrası karşılaştırma kaydedildi: {karsilastirma_yolu}")

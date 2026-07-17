import os
import csv
import tempfile
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Tekrarlanabilirlik için sabit seed
SEED = 42
set_seed(SEED)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("  TELEKOM DESTEK ASİSTANI LORA FINE-TUNING DEMO")
print("=" * 55)

# 1. Sentetik Veri Seti Hazırlığı
data = {
    "girdi": [
        "İnternetim dünden beri yok bıktım sizin gibi firmadan!",
        "Faturam çok yüksek gelmiş soyguncu musunuz siz?",
        "Müşteri hizmetlerine bağlanmak ölüm resmen, kimse açmıyor.",
        "Sürekli bağlantım kopuyor, böyle hizmet olmaz iptal edeceğim."
    ],
    "cikti": [
        "Yaşadığınız bağlantı sorunu nedeniyle özür dileriz. Teknik ekiplerimiz hattınızı kontrol etmek üzere yönlendirilmiştir.",
        "Fatura tutarınızla ilgili endişenizi anlıyoruz. Detaylı dökümü incelemek ve size yardımcı olmak için buradayız.",
        "Bekleme süreniz için üzgünüz. Yoğunluk nedeniyle oluşan bu durumu telafi etmek adına size hemen yardımcı oluyorum.",
        "Bağlantı kopmaları yaşadığınız için üzgünüz. Altyapı kontrollerini başlatarak sorununuzu kalıcı olarak çözeceğiz."
    ]
}

df = pd.DataFrame(data)
df["text"] = df.apply(lambda x: f"Şikayet: {x['girdi']} \nCevap: {x['cikti']}", axis=1)
dataset = Dataset.from_pandas(df)

# 2. Model ve Tokenizer Yükleme (Küçük boyutlu model)
model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = dataset.map(tokenize_function, batched=True)

base_model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. ÖNCESİ TESTİ — Fine-tuning'den önce base modelin çıktısı
#    (LoRA'nın gerçekten fark yaratıp yaratmadığını göstermek için)
test_prompt = "Şikayet: İnternet o kadar yavaş ki hiçbir şey açılmıyor, rezalet! \nCevap:"
inputs = tokenizer(test_prompt, return_tensors="pt").to(base_model.device)

with torch.no_grad():
    base_outputs = base_model.generate(
        **inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False
    )
base_cevap = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

print("\n[ÖNCESİ] Base model (fine-tuning YOK) çıktısı:")
print(base_cevap)

# 4. LoRA Konfigürasyonu (Sadece adaptörleri eğiteceğiz)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention + projeksiyon katmanları
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# 5. Eğitim Ayarları (Demo için kısa tutulmuştur)
training_args = TrainingArguments(
    output_dir=os.path.join(tempfile.gettempdir(), "telekom_lora_checkpoints"),
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    max_steps=200,  # 4 örneklik veri setinde LoRA'nın gözle görülür fark yaratması için yeterli tekrar
    logging_steps=20,
    seed=SEED,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

print("\n[BİLGİ] LoRA Eğitimi Başlıyor...")
train_result = trainer.train()

# 6. Loss (Hata) Grafiğini ve CSV'sini Kaydetme
log_history = trainer.state.log_history
loss_values = [log["loss"] for log in log_history if "loss" in log]
steps = [log["step"] for log in log_history if "loss" in log]

csv_yolu = os.path.join(OUTPUT_DIR, "loss_log.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "loss"])
    for s, l in zip(steps, loss_values):
        writer.writerow([s, l])
print(f"\n[SİSTEM] Loss verisi kaydedildi: {csv_yolu}")

plt.figure(figsize=(8, 5))
plt.plot(steps, loss_values, marker='o', linestyle='-', color='b')
plt.title("LoRA Fine-Tuning Eğitim Kaybı (Training Loss)")
plt.xlabel("Adım (Step)")
plt.ylabel("Kayıp (Loss)")
plt.grid(True)
grafik_yolu = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Eğitim grafiği kaydedildi: {grafik_yolu}")

# 7. SONRASI TESTİ — Aynı prompt, fine-tuned model
print("\n" + "=" * 55)
print("  ÖNCESİ / SONRASI KARŞILAŞTIRMA")
print("=" * 55)

with torch.no_grad():
    tuned_outputs = peft_model.generate(
        **inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False
    )
tuned_cevap = tokenizer.decode(tuned_outputs[0], skip_special_tokens=True)

print(f"GİRDİ:\n{test_prompt}")
print("-" * 30)
print(f"[ÖNCESİ - base distilgpt2]:\n{base_cevap}")
print("-" * 30)
print(f"[SONRASI - LoRA fine-tuned]:\n{tuned_cevap}")

# Karşılaştırmayı da CSV olarak kaydet (README'de tablo göstermek için)
karsilastirma_yolu = os.path.join(OUTPUT_DIR, "oncesi_sonrasi_karsilastirma.csv")
with open(karsilastirma_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["asama", "cikti"])
    writer.writerow(["oncesi_base_model", base_cevap])
    writer.writerow(["sonrasi_lora_finetuned", tuned_cevap])
print(f"\n[SİSTEM] Öncesi/sonrası karşılaştırma kaydedildi: {karsilastirma_yolu}")

print("\n(Not: Veri seti sadece 4 örnekten oluştuğu için model burada genelleme değil, "
      "ezberleme (memorization) yapar. 200 adım, LoRA'nın davranışı değiştirdiğini net "
      "göstermek için seçilmiştir; gerçek bir üretim modeli çok daha geniş bir veri seti "
      "ve daha ölçülü bir adım sayısı gerektirir.)")

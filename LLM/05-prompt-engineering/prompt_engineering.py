"""Prompt Engineering: zero-shot, few-shot, CoT, system prompt, temperature"""
import os
import time
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.makedirs('figures', exist_ok=True)

log = []  # her cagrinin kaydi buraya dusuyor, sonunda figures/'a yaziliyor


def get_client():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY gerekli (.env dosyasına ekle)")
    return OpenAI(api_key=key)


def ask(client, system, user, **kwargs):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})

    t0 = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs,
        temperature=kwargs.get("temperature", 0),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content
    return text, elapsed


def record(strategy, label, prompt, response, elapsed):
    log.append({
        "strategy": strategy,
        "label": label,
        "prompt": prompt,
        "response": response,
        "response_time_sec": round(elapsed, 3),
        "response_length_char": len(response),
        "word_count": len(response.split()),
    })


def zero_shot(client):
    print("=" * 50)
    print("  ZERO-SHOT")
    print("=" * 50)
    prompt = "Bir e-ticaret sitesinde sepet terk etme oranını azaltmak için 3 öneri ver."
    yanit, t = ask(client, None, prompt)
    record("zero_shot", "Zero-shot", prompt, yanit, t)
    print(f"  Yanıt: {yanit[:150]}...\n")


def few_shot(client):
    print("=" * 50)
    print("  FEW-SHOT (3 örnek)")
    print("=" * 50)
    prompt = """Yorum: "Kargo çok hızlı geldi, ürün de kaliteliydi."
Duygu: Pozitif
Yorum: "Ürün kırık geldi, iade sürecinde de kimse yanıt vermedi."
Duygu: Negatif
Yorum: "Fiyatı uygun ama kutusu biraz ezilmiş gelmiş."
Duygu: Nötr
Yorum: "Beklediğimden çok daha iyi çıktı, herkese tavsiye ederim."
Duygu:"""
    yanit, t = ask(client, None, prompt)
    record("few_shot", "Few-shot", prompt, yanit, t)
    print(f"  Yanıt: {yanit.strip()}\n")


def chain_of_thought(client):
    print("=" * 50)
    print("  CHAIN-OF-THOUGHT (CoT)")
    print("=" * 50)
    prompt = """Bir kafe, günde ortalama 240 kahve satıyor. Kahve başına maliyet 18 TL,
satış fiyatı 45 TL. Kafenin günlük sabit gideri (kira, personel vb.) 3200 TL.
Kafe bu şartlarda günlük kaç TL kâr veya zarar ediyor? Adım adım hesapla."""
    yanit, t = ask(client, None, prompt)
    record("chain_of_thought", "Chain-of-Thought", prompt, yanit, t)
    print(f"  Yanıt: {yanit[:200]}...\n")


def system_prompt_demo(client):
    print("=" * 50)
    print("  SYSTEM PROMPT")
    print("=" * 50)
    prompts = {
        "Kıdemli yazılımcı": "Sen 15 yıllık deneyimli bir kıdemli yazılım mühendisisin. Teknik ve net konuş.",
        "Yeni başlayanlara anlatan eğitmen": "Sen bir programlama eğitmenisin. Hiç bilmeyen birine anlatır gibi basit örneklerle açıkla.",
        "Şüpheci teknik incelemeci": "Sen her teknolojiyi eleştirel gözle değerlendiren bir teknoloji yazarısın. Artılarını değil, önce risklerini/sınırlarını söyle.",
    }
    for isim, sp in prompts.items():
        yanit, t = ask(client, sp, "Mikroservis mimarisine geçmeli miyim?")
        record("system_prompt", isim, "Mikroservis mimarisine geçmeli miyim?", yanit, t)
        print(f"  [{isim}]: {yanit[:100]}...\n")


def temperature_demo(client):
    print("=" * 50)
    print("  TEMPERATURE ETKİSİ")
    print("=" * 50)
    prompt = "Yeni kurulan bir teknoloji şirketi için 5 isim öner."
    for temp in [0.0, 0.5, 1.0, 1.5]:
        yanit, t = ask(client, None, prompt, temperature=temp)
        record("temperature", f"T={temp}", prompt, yanit, t)
        print(f"  T={temp:.1f}: {yanit[:80]}...")
    print()


def save_outputs():
    df = pd.DataFrame(log)
    df.to_csv("figures/tum_cagrilar.csv", index=False)
    with open("figures/tum_yanitlar.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # 1) Strateji basina yanit uzunlugu (kelime sayisi)
    main_strategies = df[df["strategy"].isin(["zero_shot", "few_shot", "chain_of_thought"])]
    plt.figure(figsize=(8, 5))
    plt.bar(main_strategies["label"], main_strategies["word_count"], color="#3b82f6")
    plt.ylabel("Yanıt Uzunluğu (kelime)")
    plt.title("Stratejiye Göre Yanıt Uzunluğu")
    plt.tight_layout()
    plt.savefig("figures/yanit_uzunlugu.png", dpi=150)
    plt.close()

    # 2) Strateji basina yanit suresi
    plt.figure(figsize=(8, 5))
    plt.bar(main_strategies["label"], main_strategies["response_time_sec"], color="#059669")
    plt.ylabel("Yanıt Süresi (saniye)")
    plt.title("Stratejiye Göre API Yanıt Süresi")
    plt.tight_layout()
    plt.savefig("figures/yanit_suresi.png", dpi=150)
    plt.close()

    # 3) System prompt karsilastirmasi (uzunluk)
    sp_df = df[df["strategy"] == "system_prompt"]
    plt.figure(figsize=(8, 5))
    plt.bar(sp_df["label"], sp_df["word_count"], color="#7c3aed")
    plt.ylabel("Yanıt Uzunluğu (kelime)")
    plt.title("System Prompt'a Göre Yanıt Uzunluğu")
    plt.tight_layout()
    plt.savefig("figures/system_prompt_karsilastirma.png", dpi=150)
    plt.close()

    # 4) Temperature - yanit uzunlugu / cesitlilik
    temp_df = df[df["strategy"] == "temperature"]
    plt.figure(figsize=(8, 5))
    plt.plot(temp_df["label"], temp_df["word_count"], marker="o", color="#dc2626", linewidth=2)
    plt.ylabel("Yanıt Uzunluğu (kelime)")
    plt.xlabel("Temperature")
    plt.title("Temperature Arttıkça Yanıt Uzunluğu Değişimi")
    plt.tight_layout()
    plt.savefig("figures/temperature_karsilastirma.png", dpi=150)
    plt.close()

    print("Kaydedildi: figures/tum_cagrilar.csv, tum_yanitlar.json")
    print("Kaydedildi: figures/yanit_uzunlugu.png, yanit_suresi.png,")
    print("            system_prompt_karsilastirma.png, temperature_karsilastirma.png")


if __name__ == "__main__":
    client = get_client()
    zero_shot(client)
    few_shot(client)
    chain_of_thought(client)
    system_prompt_demo(client)
    temperature_demo(client)
    save_outputs()

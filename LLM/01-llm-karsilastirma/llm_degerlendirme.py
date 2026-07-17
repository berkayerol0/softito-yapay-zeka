"""LLM Değerlendirme: sampling stratejileri, perplexity, BLEU/ROUGE, LLM-as-Judge"""
import os
import math
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()
torch.manual_seed(42)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sampling_demo():
    print("Sampling stratejileri calistiriliyor...")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompt = "Yapay zeka gelecekte"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    stratejiler = [
        ("Greedy", dict(do_sample=False, max_new_tokens=25)),
        ("Low Temp (0.3)", dict(do_sample=True, temperature=0.3, max_new_tokens=25)),
        ("High Temp (1.5)", dict(do_sample=True, temperature=1.5, max_new_tokens=25)),
        ("Top-k (10)", dict(do_sample=True, top_k=10, temperature=0.7, max_new_tokens=25)),
        ("Top-p (0.85)", dict(do_sample=True, top_p=0.85, temperature=0.7, max_new_tokens=25)),
    ]

    sonuclar = []
    for isim, kwargs in stratejiler:
        with torch.no_grad():
            out = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        metin = tokenizer.decode(out[0], skip_special_tokens=True)
        uretilen_kisim = metin[len(prompt):].strip()
        print(f"  {isim:<18}: {metin}")
        sonuclar.append({"strateji": isim, "metin": metin, "uretilen_kelime_sayisi": len(uretilen_kisim.split())})

    df = pd.DataFrame(sonuclar)
    df.to_csv(f"{OUTPUT_DIR}/sampling_sonuclari.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(df["strateji"], df["uretilen_kelime_sayisi"], color="#3b82f6")
    plt.ylabel("Üretilen Kelime Sayısı")
    plt.title("Sampling Stratejisine Göre Üretilen Metin Uzunluğu")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sampling_karsilastirma.png", dpi=150)
    plt.close()

    return model, tokenizer


def perplexity_hesapla(model, tokenizer):
    print("Perplexity hesaplaniyor...")

    test_metinleri = [
        ("Anlamlı ve tutarlı", "Yapay zeka, insan benzeri düşünme yeteneği kazandırmayı amaçlayan bir bilgisayar bilimi dalıdır."),
        ("Günlük konuşma", "Bugün hava çok güzel, parka yürüyüşe gideceğim."),
        ("Teknik metin", "Transformer mimarisi self-attention mekanizması ile paralel işleme sağlar."),
        ("Anlamsız", "asdfg hjkl qwert zuıop cvbnm wxyz 123456"),
    ]

    sonuclar = []
    for etiket, metin in test_metinleri:
        inputs = tokenizer(metin, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        ppl = math.exp(outputs.loss.item())
        print(f"  PPL={ppl:8.1f} | {etiket}: {metin[:50]}...")
        sonuclar.append({"kategori": etiket, "metin": metin, "perplexity": round(ppl, 1)})

    df = pd.DataFrame(sonuclar)
    df.to_csv(f"{OUTPUT_DIR}/perplexity_sonuclari.csv", index=False)

    plt.figure(figsize=(9, 5))
    colors = ["#dc2626" if k == "Anlamsız" else "#059669" for k in df["kategori"]]
    plt.bar(df["kategori"], df["perplexity"], color=colors)
    plt.ylabel("Perplexity (düşük = modele göre daha 'doğal')")
    plt.title("Metin Türüne Göre Perplexity")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/perplexity_karsilastirma.png", dpi=150)
    plt.close()


def ngram_metrikleri_hesapla():
    print("BLEU ve ROUGE hesaplaniyor...")

    referans = "Transformer mimarisi 2017 yılında Google araştırmacıları tarafından geliştirildi."

    adaylar = {
        "Çok Yakın": "Transformer mimarisi 2017'de Google araştırmacıları tarafından geliştirilmiştir.",
        "Farklı Dizilim": "Google araştırmacıları 2017 yılında Transformer mimarisini geliştirdi.",
        "Eksik Bilgi": "Transformer Google tarafından geliştirilen bir sistemdir.",
        "Alakasız": "Bugün hava gerçekten çok güzel ve güneşli görünüyor.",
    }

    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    sonuclar = []
    for isim, aday in adaylar.items():
        bleu = sentence_bleu([referans.split()], aday.split(), smoothing_function=smoothing)
        rouge_l = scorer.score(referans, aday)["rougeL"].fmeasure

        print(f"  [{isim}] BLEU={bleu:.4f} | ROUGE-L={rouge_l:.4f}")
        sonuclar.append({"aday": isim, "metin": aday, "bleu": round(bleu, 4), "rouge_l": round(rouge_l, 4)})

    return sonuclar


def llm_as_judge_demo():
    print("LLM-as-Judge calistiriliyor (Gemini API)...")

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("  GEMINI_API_KEY bulunamadi, bu adim atlaniyor.")
        return []

    from google import genai
    client = genai.Client(api_key=api_key)

    sorular = [
        {
            "soru": "Derin öğrenmede Overfitting nedir ve nasıl engellenir?",
            "yanit_a": "Overfitting, modelin eğitim verisini ezberlemesi ve yeni verilere genelleme yapamamasıdır. Dropout, L2 regülarizasyonu ve erken durdurma ile engellenir.",
            "yanit_b": "Overfitting modelin çok yavaş çalışması demektir. Daha büyük ekran kartı kullanarak çözülür.",
        },
        {
            "soru": "Gradient descent nedir?",
            "yanit_a": "Bir modelin kayıp fonksiyonunu minimize etmek için parametrelerin gradyan yönünde güncellendiği optimizasyon algoritmasıdır.",
            "yanit_b": "Gradient descent, verinin bir tablodan diğerine aktarılması işlemidir.",
        },
        {
            "soru": "Attention mekanizması ne işe yarar?",
            "yanit_a": "Modelin bir çıktı üretirken girdinin hangi kısımlarına daha çok odaklanması gerektiğini öğrenmesini sağlar.",
            "yanit_b": "Attention, modelin RAM kullanımını azaltan bir sıkıştırma tekniğidir.",
        },
    ]

    sonuclar = []
    for s in sorular:
        prompt = f"""Aşağıdaki iki yanıtı doğruluk ve teknik netlik açısından değerlendir.
Soru: {s['soru']}
Yanıt A: {s['yanit_a']}
Yanıt B: {s['yanit_b']}
Sadece 'A' veya 'B' yaz, başka bir şey yazma."""

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        kazanan = response.text.strip()
        print(f"  Soru: {s['soru'][:40]}... -> Kazanan: {kazanan}")
        sonuclar.append({"soru": s["soru"], "kazanan": kazanan})

    df = pd.DataFrame(sonuclar)
    df.to_csv(f"{OUTPUT_DIR}/judge_sonuclari.csv", index=False)

    win_counts = df["kazanan"].value_counts()
    plt.figure(figsize=(6, 5))
    plt.bar(win_counts.index, win_counts.values, color=["#059669", "#dc2626"])
    plt.ylabel("Kazanma Sayısı")
    plt.title(f"LLM-as-Judge Sonuçları ({len(df)} soru)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/judge_karsilastirma.png", dpi=150)
    plt.close()

    return sonuclar


def metrikleri_kaydet(n_gram_sonuclari):
    df = pd.DataFrame(n_gram_sonuclari)
    df.to_csv(f"{OUTPUT_DIR}/evaluation_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    x = range(len(df))
    plt.bar([i - 0.2 for i in x], df["bleu"], width=0.4, label="BLEU")
    plt.bar([i + 0.2 for i in x], df["rouge_l"], width=0.4, label="ROUGE-L")
    plt.xticks(x, df["aday"])
    plt.xlabel("Aday Metin")
    plt.ylabel("Skor (0-1)")
    plt.title("BLEU ve ROUGE-L Karşılaştırması")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=150)
    plt.close()

    print(f"Kaydedildi: {OUTPUT_DIR}/evaluation_results.csv, metrics_comparison.png")


if __name__ == "__main__":
    print("LLM Değerlendirme başlıyor.\n")

    model, tokenizer = sampling_demo()
    perplexity_hesapla(model, tokenizer)
    n_gram_skorlari = ngram_metrikleri_hesapla()
    metrikleri_kaydet(n_gram_skorlari)
    llm_as_judge_demo()

    print("\nTamamlandı. Sonuçlar 'figures' klasöründe.")

# LLM

Büyük dil modelleri (LLM) ile ilgili temel kavramları ve teknikleri gösteren bağımsız demo
projelerden oluşan bir koleksiyon. Her proje kendi `.py` dosyası, `README.md`'si,
`figures/` klasörü (çıktı grafikleri/CSV'leri) ve `requirements.txt`'i ile bağımsız
çalıştırılabilir.

## Projeler

| # | Klasör | Konu | Ne Gösteriyor |
|---|---|---|---|
| 01 | [`01-llm-karsilastirma`](01-llm-karsilastirma/) | LLM çıktı değerlendirme | BLEU, ROUGE ve Perplexity metrikleriyle farklı model/prompt çıktılarının objektif karşılaştırılması |
| 02 | [`02-fine-tuning-lora`](02-fine-tuning-lora/) | LoRA fine-tuning | `distilgpt2` üzerinde LoRA adaptörleriyle iki ayrı görev: kurumsal ton öğretimi (telekom asistanı) ve yapılandırılmış JSON çıkarımı (finans logları) |
| 03 | [`03-langchain`](03-langchain/) | Agent, memory, tool calling | LangChain'in güncel `create_agent` API'siyle kurulan, konuşma hafızasına sahip ve ihtiyaç halinde araç çağıran bir destek triyaj botu |
| 04 | [`04-rag`](04-rag/) | Retrieval-Augmented Generation | Türkçe Wikipedia makalelerinin FAISS ile indexlenip, kaynak göstererek soru-cevap yapan bir RAG sistemi |
| 05 | [`05-prompt-engineering`](05-prompt-engineering/) | Prompt stratejileri | Zero-shot, few-shot ve chain-of-thought stratejilerinin aynı görevde sistematik doğruluk karşılaştırması |
| 06 | [`06-rlhf-dpo`](06-rlhf-dpo/) | DPO (Direct Preference Optimization) | `trl` kütüphanesiyle tercih çiftlerinden (chosen/rejected) davranış hizalaması — modele kaba yerine kibar cevabı tercih etmeyi öğretme |
| 07 | [`07-quantization`](07-quantization/) | Kuantizasyon | Aynı modelin FP32/FP16/INT8 hassasiyetlerinde boyut, hız ve çıktı kalitesi karşılaştırması; GPT-2'nin `Conv1D` katmanları yüzünden ortaya çıkan gerçek bir kuantizasyon tuzağının tespiti ve düzeltilmesi |

## Kullanılan Modeller

- **Yerel/ücretsiz projeler (01, 02, 05, 06, 07):** `distilgpt2` — küçük, hızlı, CPU'da
  (AMD dahil herhangi bir donanımda) sorunsuz çalışan açık kaynak bir model. API key
  gerektirmez.
- **API tabanlı projeler (03, 04):** Google **Gemini** (`gemini-2.5-flash` ve
  `gemini-embedding-001`) — `langchain-google-genai` üzerinden. Bu projeler çalıştırmak
  için ücretsiz bir Gemini API key gerektirir (`.env` dosyasına `GEMINI_API_KEY` olarak
  eklenir).

## Ortak Prensipler

- Her proje **öncesi/sonrası karşılaştırma** veya **somut bir çıktı** (grafik, CSV, log)
  üretir; sadece kod değil, gözle görülür bir sonuç sunar.
- README'ler dürüsttür: bir yöntemin sınırlamaları, beklenmeyen bir davranış veya bir
  modelin neden başarısız olduğu varsa bu açıkça belirtilir, sonuçlar abartılmaz.
- Güncel olmayan/deprecated API'ler yerine (ör. `langchain.memory`, `create_react_agent`,
  eski embedding model adları, kaldırılmış `trl` parametreleri) kütüphanelerin güncel
  API'leri kullanılır.

## Çalıştırma

Her alt klasör bağımsızdır:

```bash
cd <klasör-adı>
pip install -r requirements.txt
# Gemini kullanan projeler için: .env dosyasına GEMINI_API_KEY=senin-key-in ekleyin
python <script-adı>.py
```

Detaylı yöntem, sonuç ve sınırlamalar için her alt klasörün kendi `README.md` dosyasına
bakınız.

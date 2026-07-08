# Turkce Kucuk Dil Modeli (SLM) + RAG

Wikipedia'dan toplanan Turkce bilim/teknoloji makaleleri uzerinde egitilen,
karakter seviyeli, Transformer tabanli (decoder-only) bir kucuk dil modeli
(SLM) projesi. Iki fazdan olusur:

- **Faz 1**: Transformer mimarisiyle metin uretimi (bir sonraki karakteri tahmin etme)
- **Faz 2**: Retrieval-Augmented Generation (RAG) ile kaynak gosteren soru-cevap sistemi

## Proje Yapisi

```
turkce-slm-rag/
├── configs/config.yaml       Tum hiperparametreler ve ayarlar
├── data/                     Ham ve islenmis veri (git'e dahil degil)
├── src/                      Kaynak kod
│   ├── data_collection.py    Wikipedia'dan veri cekme
│   ├── preprocessing.py      Temizleme + tokenizer
│   ├── dataset.py            PyTorch Dataset/DataLoader
│   ├── models/transformer.py  Decoder-only Transformer mimarisi
│   ├── train.py               Egitim script'i
│   ├── generate.py            Metin uretim script'i
│   ├── chunking.py            RAG icin pasajlama (Faz 2)
│   ├── embedding.py           TF-IDF embedding (Faz 2)
│   ├── retrieval.py           Pasaj arama (Faz 2)
│   └── rag_pipeline.py        Bul + uret akisi (Faz 2)
├── demo/app.py                Gradio arayuzu
├── tests/                     Unit testler
├── figures/                   Egitim grafikleri (git'e dahil degil)
└── checkpoints/                Egitilmis model dosyalari (git'e dahil degil)
```

## Kurulum

```bash
git clone <repo-url>
cd turkce-slm-rag

# (opsiyonel ama onerilir) sanal ortam
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## Kullanim

Not: `generate.py`, `chunking.py`, `embedding.py`, `rag_pipeline.py` dosyalari
`src` paketinin icini kullandigi icin `python -m src.<dosya_adi>` seklinde,
yani **modul olarak** calistirilmalidir. `data_collection.py`, `preprocessing.py`
ve `train.py` ise direkt `python src/<dosya_adi>.py` ile calistirilabilir.

### Faz 1: Veri Toplama ve Egitim

```bash
# 1) Wikipedia'dan veri cek
python src/data_collection.py --config configs/config.yaml

# 2) Veriyi temizle, tokenizer olustur
python src/preprocessing.py --config configs/config.yaml

# 3) Modeli egit
python src/train.py --config configs/config.yaml

# 4) Metin uret
python -m src.generate --prompt "Yapay zeka" --uzunluk 300
```

### Faz 2: RAG Kurulumu

```bash
# 5) Makaleleri pasajlara bol
python -m src.chunking --config configs/config.yaml

# 6) Pasajlari vektorlestir
python -m src.embedding --config configs/config.yaml
```

RAG pipeline'ini Python icinden kullanmak icin:

```python
from src.rag_pipeline import RagPipeline

pipeline = RagPipeline("configs/config.yaml", "checkpoints/slm_model.pt")
sonuc = pipeline.soru_sor("Yapay zeka nedir?")

print(sonuc["cevap"])
print(sonuc["kaynaklar"])
```

### Demo Arayuzu

```bash
python demo/app.py
```

Tarayicida acilan link (orn. `http://127.0.0.1:7860`) uzerinden iki sekmeli
arayuze erisilir: metin tamamlama (Faz 1) ve RAG soru-cevap (Faz 2).

## Model Mimarisi

Decoder-only, GPT tarzi bir Transformer kullanilir: multi-head self-attention
+ feed-forward bloklari, karakter seviyeli embedding ve pozisyon embedding'i
ile. Katman sayisi, embedding boyutu ve attention head sayisi `config.yaml`
icinde `model` bolumunden ayarlanir.

## Veri Kaynagi

Veri, Wikipedia'nin resmi API'si (`wikipedia-api` kutuphanesi) ile toplanir,
HTML kazima (scraping) yapilmaz. `configs/config.yaml` icindeki kategori/makale
listesi degistirilerek farkli konu alanlarina genisletilebilir. Su an 8
kategori altinda ~40 makale (bilim, teknoloji, matematik, fizik, biyoloji,
uzay, kimya, muhendislik) tanimlidir.

## Testler

```bash
pytest tests/
```

`test_tokenizer.py`: karakter tokenizer'in encode/decode dogrulugunu ve
Turkce karakter destegini test eder.
`test_retrieval.py`: RAG icin metin parcalama (chunking) mantigini test eder.

## Gelistirme Fikirleri

- Karakter seviyesi yerine BPE/subword tokenizer
- TF-IDF yerine sentence-transformers ile daha guclu embedding
- Daha genis Wikipedia korpusu (40 makale -> 500+)
- Instruction-tuning ile daha dogal soru-cevap davranisi

## Lisans

MIT

# SLM (Small Language Model) Calismalari

Bu repo, kucuk dil modelleri (Small Language Model) uzerine yaptigim
calismalari icerir.

## Icerik

### turkce-slm/

Wikipedia'dan toplanan Turkce bilim/teknoloji makaleleri uzerinde egitilen,
karakter seviyeli, Transformer tabanli (decoder-only) bir kucuk dil modeli.
Model, verilen bir baslangic metnine gore devamini uretir (metin tamamlama).

- Veri: Wikipedia API ile 8 kategori altinda ~40 makale (bilim, teknoloji,
  matematik, fizik, biyoloji, uzay, kimya, muhendislik)
- Model: Decoder-only Transformer (multi-head self-attention), karakter
  seviyeli tokenizer
- Egitim: Google Colab (GPU) uzerinde gerceklestirildi, sonuclar ve egitim
  grafigi proje ici README'de paylasilmistir

Detayli kurulum ve kullanim icin: [`turkce-slm/README.md`](turkce-slm/README.md)

## Kullanilan Teknolojiler

- PyTorch (model mimarisi ve egitim)
- Wikipedia API (veri toplama)
- Gradio (demo arayuzu)
- pytest (testler)

## Lisans

MIT

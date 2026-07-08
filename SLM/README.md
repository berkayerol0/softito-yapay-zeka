# SLM (Small Language Model) Calismalari

Bu repo, kucuk dil modelleri (Small Language Model) uzerine yaptigim
calismalari icerir.

## Icerik

### turkce-slm-rag/

Wikipedia'dan toplanan Turkce bilim/teknoloji makaleleri uzerinde egitilen,
karakter seviyeli, Transformer tabanli (decoder-only) bir kucuk dil modeli.
Iki bolumden olusur:

- Transformer mimarisiyle metin uretimi (bir sonraki karakteri tahmin etme)
- Retrieval-Augmented Generation (RAG) ile kaynak gosteren soru-cevap sistemi

Veri, Wikipedia'nin resmi API'si ile 8 kategori altinda ~40 makaleden
(bilim, teknoloji, matematik, fizik, biyoloji, uzay, kimya, muhendislik)
toplanmistir.

Detayli kurulum ve kullanim icin: [`turkce-slm-rag/README.md`](turkce-slm-rag/README.md)

## Kullanilan Teknolojiler

- PyTorch (model mimarisi ve egitim)
- Wikipedia API (veri toplama)
- scikit-learn (TF-IDF tabanli embedding)
- Gradio (demo arayuzu)
- pytest (testler)

## Lisans

MIT

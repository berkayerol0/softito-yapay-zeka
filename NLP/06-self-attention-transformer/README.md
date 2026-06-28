# 06 — Self-Attention & Transformer

🔜 Yakında.

## Bu bölüm ne anlatacak?

RNN/LSTM dizileri sırayla işlediği için paralelleştirilemez ve eğitimi yavaştır. Self-Attention, bir cümledeki her kelimenin **aynı cümledeki diğer tüm kelimelerle** ilişkisini aynı anda (paralel) hesaplar. Transformer mimarisi bunu çoklu-başlı (multi-head) self-attention katmanları üzerine kurar ve bugünün büyük dil modellerinin (BERT, GPT vb.) temelini oluşturur.

## Planlanan içerik

- Query, Key, Value matrislerinin self-attention'daki rolü
- Scaled dot-product attention formülü ve sıfırdan uygulanması
- Multi-head attention: neden tek bir attention yetmiyor?
- Positional encoding: sıra bilgisini paralel işlemde nasıl koruyoruz?
- Transformer encoder bloğunun genel mimarisi
- TF-IDF'ten buraya kadar olan yolculuğun özeti: hangi problem hangi yöntemle çözüldü

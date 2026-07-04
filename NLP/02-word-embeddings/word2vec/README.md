# Word2Vec — Skip-gram + Negative Sampling (Sıfırdan PyTorch İmplementasyonu)

Kelimeleri, anlamca benzer olanların vektör uzayında birbirine yaklaştığı sayısal vektörlere dönüştüren klasik bir temsil öğrenme (representation learning) modeli. Etiketli veri yok — model milyonlarca kez "bu kelimenin etrafında hangi kelimeler geçer?" sorusunu tahmin ederek öğreniyor. Tek bir Python dosyasında; veriyi indirir, kelime dağarcığı kurar, Skip-gram + Negative Sampling ile PyTorch'ta sıfırdan model eğitir, sonuçları resmi analoji test setiyle ve `gensim`'in kendi Word2Vec'iyle karşılaştırarak raporlar.

> Veri seti: [text8](https://github.com/numenta/NAB) (Wikipedia'dan temizlenmiş klasik word2vec korpusu, ~2M kelimelik alt küme) · Değerlendirme: [Google/Mikolov resmi analoji test seti](https://github.com/tmikolov/word2vec) (19.544 soru)

## Proje Hakkında

2013'te Mikolov ve arkadaşlarının Google'da yayınladığı orijinal makalenin ("Efficient Estimation of Word Representations in Vector Space") tam kendisi: iki ayrı gömme (embedding) katmanı — kelimenin *merkez* rolündeki ve *bağlam* rolündeki vektörleri — negatif örnekleme ile eğitiliyor. Hazır bir kütüphaneye sarılmadan (`gensim.Word2Vec(...)` tek satır yazıp geçmek yerine) kayıp fonksiyonundan eğitim döngüsüne kadar her adım elle kuruldu.

**Neden önemli:** Bu proje, RNN/LSTM projelerinden farklı olarak **gözetimsiz (unsupervised) öğrenme** gösteriyor — etiket yok, model sadece ham metinden kendi hedefini üretiyor.

## Yöntem

```
"the king ruled the ancient kingdom"
          │
          ▼  merkez kelime: "ruled"  |  pencere: ±5
          │
   Pozitif çift: (ruled, king), (ruled, ancient), (ruled, kingdom) ...
   Negatif çiftler: (ruled, rastgele_kelime) × 5   [unigram^0.75 dağılımından]
          │
          ▼
   Kayıp = -log σ(v_ruled · v_king) - Σ log σ(-v_ruled · v_negatif)
```

- **Alt örnekleme (subsampling):** "the", "of" gibi çok sık kelimeler olasılıksal olarak seyrekleştirilir (orijinal makaledeki formül) — nadir kelimelere daha fazla ağırlık kalır.
- **Negatif örnekleme dağılımı:** unigram frekansının 0.75. kuvveti — çok sık kelimelerin negatif örnek olarak aşırı seçilmesini engeller.
- **Öğrenme oranı:** eğitim boyunca doğrusal olarak azaltılır (orijinal word2vec gibi).

## Sonuçlar

**En yakın komşular** (kosinüs benzerliği) — model anlamsal ilişkileri gerçekten öğrenmiş:

| Sorgu | En yakın komşular |
|---|---|
| king | regent, throne, son, burgundy, maria, toulouse |
| computer | computers, macintosh, chipset, interface, hardware, gui |
| music | musicians, musical, musicals, pop, instrumental, songs |
| water | evaporation, moisture, warm, moist, erosion |
| science | fiction, scientific, observational, theoretical, foundational |

**Analoji görevi (resmi Google test seti, 3CosAdd yöntemi):**

| Model | Doğruluk |
|---|---|
| Bizim implementasyon (sıfırdan) | %5.1 (102/2004) |
| Gensim Word2Vec (aynı veri, aynı hiperparametreler) | %0.6 (13/2004) |

t-SNE görselleştirmesinde ülkeler, başkentler, sayılar, meslekler ve zaman kelimeleri kendi aralarında net kümeler oluşturuyor (`figures/03_tsne_word_clusters.png`).

Görseller `figures/` klasöründe:
- `01_training_loss.png` — eğitim kaybı
- `02_analogy_accuracy_comparison.png` — kategori bazlı doğruluk, gensim kıyaslı
- `03_tsne_word_clusters.png` — 2D kelime kümesi görselleştirmesi
- `04_nearest_neighbors.png` — örnek kelimelerin en yakın komşuları

## Metodolojik Notlar (Dürüst Okuma)

- **Korpus küçük tutuldu (~2M kelime)** — orijinal makale milyarlarca kelimeyle eğitiliyordu, bu yüzden mutlak analoji doğruluğu (%5 civarı) düşük görünür. Bu beklenen bir durumdur; amaç metodolojiyi doğru uygulamak ve endüstri standardıyla adil bir kıyas yapmaktır, state-of-the-art skor üretmek değil.
- **Gensim kıyasında dikkat edilen bir tuzak:** Gensim'e token dizisini tek bir dev "cümle" olarak vermek (`sentences=[tokens]`) modelin neredeyse hiç öğrenmemesine yol açıyor; gerçekçi cümle uzunluklarına bölünmesi (`GENSIM_CHUNK=1000`) gerekiyor. Bu proje bu tuzağı tespit edip düzeltti — kıyasın adil ve doğru olduğundan emin olmak için.
- **Nihai kelime vektörü olarak** `in_embed` (merkez) katmanı kullanıldı; bu, orijinal word2vec uygulamasındaki `syn0` ile aynı yaklaşım.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python word2vec_skipgram.py
```

Veri seti ilk çalıştırmada otomatik indirilir. Çıktılar `figures/` klasöründe, eğitilmiş model `data/word2vec_model.pt` olarak kaydedilir.

## Dosya Yapısı

```
├── word2vec_skipgram.py   # Tüm proje — veri, model, eğitim, değerlendirme, gensim kıyası
├── requirements.txt
├── data/                  # İndirilen veri + eğitilmiş model (otomatik oluşur)
└── figures/                # Üretilen görseller (otomatik oluşur)
```

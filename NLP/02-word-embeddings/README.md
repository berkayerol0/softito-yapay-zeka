# Word Embeddings — Word2Vec, GloVe ve FastText Karşılaştırması

Kelimeleri anlamlarını yakalayan sayısal vektörlere dönüştüren üç klasik yöntemin sıfırdan (veya endüstri standardı kütüphaneyle) uygulanması ve **aynı veri, aynı test seti** üzerinde adil bir kıyası. Her alt klasör bağımsız, kendi başına çalışabilen bir mini projedir; birlikte ele alındıklarında ise kelime temsili öğreniminin üç farklı felsefesini somut biçimde ortaya koyarlar.

> Ortak veri seti: [text8](https://github.com/numenta/NAB) (Wikipedia'dan temizlenmiş, ~2M kelimelik alt küme) · Ortak değerlendirme: [Google/Mikolov resmi analoji test seti](https://github.com/tmikolov/word2vec) (19.544 soru)

## Neden Üç Yöntem Birlikte?

Kelime vektörü öğrenmenin tek bir "doğru" yolu yok — üç farklı felsefe var:

| | [Word2Vec](./word2vec) | [GloVe](./glove) | [FastText](./fasttext) |
|---|---|---|---|
| **Yıl / Kaynak** | 2013, Google | 2014, Stanford | 2016, Facebook |
| **Yaklaşım** | Yerel bağlamı tahmin et (prediction-based) | Global co-occurrence istatistiğini çarpanlarına ayır (count-based) | Kelimeyi karakter parçalarından (subword) inşa et |
| **Temel birim** | Kelime | Kelime | Karakter n-gram |
| **Görülmemiş (OOV) kelime** | ❌ Tanıyamaz | ❌ Tanıyamaz | ✅ Subword'lerden vektör üretir |
| **Bu projede en güçlü olduğu yer** | Anlamsal ilişkiler | Anlamsal ilişkiler (küçük veride en iyisi) | Dilbilgisel ilişkiler (ekler, çekimler) |

Her üçü de **aynı text8 alt kümesiyle** ve **aynı analoji test setiyle** değerlendirildiği için sonuçlar doğrudan karşılaştırılabilir.

## Sonuçların Özeti

**Analoji görevinde toplam doğruluk** (aynı korpus, aynı 2004 soru):

| Model | Doğruluk |
|---|---|
| FastText | %51.9 |
| GloVe | %5.7 |
| Word2Vec | %5.1 |

Toplam skor FastText'i açık ara önde gösteriyor gibi görünse de, **asıl hikaye kategori kırılımında**:

- **Dilbilgisel kategoriler** (sıfat→zarf, çoğul, karşıtlık): FastText ezici üstün (%70-93), çünkü bu ilişkiler ekler (-ly, -s, un-) üzerinden kuruluyor ve FastText tam da bunu karakter düzeyinde yakalıyor.
- **Anlamsal kategoriler** (başkent-ülke, aile ilişkileri): Word2Vec ve GloVe daha iyi, çünkü bu ilişkiler kelimenin yazılışından değil kullanıldığı bağlamdan çıkarılıyor.

Yani "hangisi daha iyi" sorusunun tek bir cevabı yok — **hangi görev için hangi yöntem** sorusunun cevabı var. Detaylı grafikler her alt klasörün `figures/` dizininde.

## Alt Projeler

### [`word2vec/`](./word2vec) — Skip-gram + Negative Sampling
Mikolov'un orijinal makalesi, PyTorch'ta sıfırdan. İki ayrı embedding katmanı (merkez/bağlam), negatif örnekleme, alt örnekleme (subsampling). Gensim ile kıyaslı.

### [`glove/`](./glove) — Global Vectors
Stanford'un count-based yöntemi, PyTorch'ta sıfırdan. Global co-occurrence matrisi + ağırlıklı en küçük kareler regresyonu, AdaGrad ile optimize edilir. Word2Vec ile aynı korpusta kıyaslı.

### [`fasttext/`](./fasttext) — Subword Vektörleri
Facebook'un karakter n-gram tabanlı yöntemi. Subword çıkarım mantığı sıfırdan gösteriliyor, eğitim resmi `fasttext` kütüphanesiyle yapılıyor. OOV (görülmemiş kelime) demosu ve üçlü karşılaştırma bu projenin öne çıkanı.

## Ortak Metodolojik Notlar

- **Adil kıyas için:** Üçü de aynı `text8` alt kümesini (~2M kelime, `min_count=5`) ve aynı analoji test setini kullanıyor.
- **Neden küçük korpus?** Bu projeler sınırlı hesaplama kaynağıyla (tek CPU çekirdeği) geliştirildi; amaç state-of-the-art skor üretmek değil, üç yöntemin mekaniğini doğru uygulayıp adil bir şekilde kıyaslamaktır. Daha güçlü donanımda `SUBSET_CHARS` değerini artırarak (tam text8 veya daha büyük bir korpusla) çok daha yüksek mutlak skorlar elde edilebilir.
- **Çapraz kıyaslar otomatik:** GloVe ve FastText script'leri, aynı klasörde Word2Vec/GloVe kontrol noktası (`reference_*.pt`) bulurlarsa otomatik olarak çapraz karşılaştırma yapar; bulamazlarsa tek başlarına çalışırlar.

## Her Birini Çalıştırma

```bash
cd word2vec && pip install -r requirements.txt && python word2vec_skipgram.py
cd ../glove && pip install -r requirements.txt && python glove_scratch.py
cd ../fasttext && pip install -r requirements.txt && python fasttext_subword.py
```

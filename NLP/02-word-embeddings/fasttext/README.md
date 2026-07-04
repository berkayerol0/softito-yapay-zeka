# FastText — Subword (Karakter n-gram) Tabanlı Kelime Vektörleri

Word2Vec ve GloVe kelimeyi bölünmez bir bütün olarak görürken, FastText her kelimeyi karakter parçalarından (n-gram) oluşan bir bileşim olarak ele alır. Bu tek fark, çok büyük bir yetenek getirir: **daha önce hiç görülmemiş kelimelere bile vektör üretebilme**. Tek bir Python dosyasında; veriyi indirir, subword mekanizmasını gösterir, Facebook'un resmi `fasttext` kütüphanesiyle model eğitir, OOV yeteneğini kanıtlar ve sonuçları Word2Vec + GloVe projeleriyle üçlü karşılaştırır.

> Veri seti: text8 (Word2Vec/GloVe projeleriyle **aynı** ~2M kelimelik alt küme) · Değerlendirme: [Google/Mikolov resmi analoji test seti](https://github.com/tmikolov/word2vec)

## Proje Hakkında

Bojanowski, Grave, Joulin ve Mikolov'un Facebook AI'da 2016'da yayınladığı "Enriching Word Vectors with Subword Information" makalesinin uygulaması. Temel fikir:

```
"where" kelimesi → subword'ler: <wh, whe, her, ere, re>, <whe, wher, here, ere>, ...
Kelime vektörü = kelimenin kendi vektörü + TÜM subword vektörlerinin toplamı
```

**Word2Vec/GloVe'dan temel farkı:** O modeller kelime dağarcığında olmayan bir kelime için hiçbir şey döndüremez (KeyError). FastText, kelimeyi hiç görmemiş olsa bile parçalarını (ör. "-ly", "un-", "connect") tanıdığı için anlamlı bir vektör oluşturabilir. Bu, özellikle ekli/çekimli dillerde (Türkçe gibi) ve yazım hatalarında büyük avantajdır.

## Yöntem

- **Subword çıkarımı:** Her kelime `<` ve `>` sınır işaretleriyle çevrelenir, 3–5 karakterlik tüm n-gram'lara bölünür. Bunlar hash bucket'lara atanır (bellek verimliliği için).
- **Eğitim:** Skip-gram + negative sampling (Word2Vec'le aynı iskelet), ama merkez kelime vektörü subword'lerinin toplamından oluşur.
- **Neden resmi kütüphane?** Subword toplama, her kelime için ~20 vektör toplamayı gerektirir; bu, tek çekirdekli CPU'da sıfırdan çok yavaştır. Bu yüzden sektörde kullanılan Facebook'un optimize `fasttext` kütüphanesi tercih edildi; subword çekirdek fikri ise kod içinde sıfırdan gösterildi.

## Sonuçlar

### FastText'in İmza Özelliği: OOV (Görülmemiş Kelime) Vektörleri

Hiçbiri eğitim vocab'ında olmayan kelimeler için bile anlamlı komşular:

| Görülmemiş kelime | En yakın komşular |
|---|---|
| playfulness | fondness, fullness, fitness |
| unhappily | unruly, unhappy, boldly, oddly |
| reconfigure | configured, reconcile, recoil |
| hyperconnectivity | connectivity, sensitivity, conductivity |

Word2Vec ve GloVe bu kelimelerin hiçbirine cevap veremez — FastText subword'lerden üretir.

### Üçlü Analoji Karşılaştırması (aynı korpus, aynı test)

| Model | Toplam Doğruluk |
|---|---|
| **FastText** | **%51.9** (1040/2004) |
| GloVe | %5.7 (115/2004) |
| Word2Vec | %5.1 (103/2004) |

**Ama toplam skor yanıltıcı — asıl hikaye kategori dağılımında** (`figures/02`):

| Kategori tipi | FastText | Word2Vec | GloVe |
|---|---|---|---|
| Dilbilgisel: sıfat→zarf | %91 | %0 | %0 |
| Dilbilgisel: karşıtlık | %93 | %1 | %0 |
| Dilbilgisel: çoğul | %73 | %8 | %2 |
| Anlamsal: başkent-ülke | %0 | %6 | %17 |
| Anlamsal: şehir-eyalet | %8 | %9 | %21 |

FastText, ek-tabanlı (morphological) dilbilgisel ilişkilerde ezici üstün çünkü subword'ler tam da bu ekleri (-ly, -er, -s, un-) yakalıyor. Buna karşılık saf anlamsal ilişkilerde (başkent-ülke) Word2Vec/GloVe daha iyi, çünkü orada kelimenin karakterleri değil bağlamı önemli. **Bu, üç yöntemin farklı şeyler öğrendiğinin somut kanıtıdır.**

Görseller `figures/` klasöründe:
- `01_oov_capability.png` — OOV kelimeler için komşular (FastText'in imza özelliği)
- `02_analogy_accuracy_comparison.png` — FastText vs Word2Vec vs GloVe, kategori bazlı
- `03_tsne_word_clusters.png` — 2D kelime kümesi görselleştirmesi
- `04_nearest_neighbors.png` — örnek kelimelerin en yakın komşuları

## Üç Yöntemin Karşılaştırması (Özet)

| | Word2Vec | GloVe | FastText |
|---|---|---|---|
| Yaklaşım | Yerel bağlam, tahmin-temelli | Global istatistik, sayım-temelli | Subword, tahmin-temelli |
| Temel birim | Kelime | Kelime | Karakter n-gram |
| OOV kelime | ❌ | ❌ | ✅ |
| Güçlü olduğu yer | Anlamsal | Anlamsal (küçük veride en iyi) | Dilbilgisel / ekli diller |
| Yıl / Kaynak | 2013 / Google | 2014 / Stanford | 2016 / Facebook |

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python fasttext_subword.py
```

Word2Vec ve GloVe kontrol noktaları (`reference_word2vec_model.pt`, `reference_glove_model.pt`) mevcutsa üçlü karşılaştırma otomatik yapılır; yoksa FastText tek başına çalışır.

## Dosya Yapısı

```
├── fasttext_subword.py    # Tüm proje — subword, eğitim, OOV demo, değerlendirme, üçlü kıyas
├── requirements.txt
├── data/                  # İndirilen veri + eğitilmiş model (otomatik oluşur)
└── figures/                # Üretilen görseller (otomatik oluşur)
```

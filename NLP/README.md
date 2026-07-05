# NLP Anatomisi: Klasik Yöntemlerden Transformer'a

## Proje Haritası

| # | Klasör | Yöntem | Bu Projede Ne Yaptık |
|---|---|---|---|
| 01 | [`01-tf-idf`](./01-tf-idf) | TF-IDF | Kadın giyim e-ticaret yorumlarından (23.486 yorum) ürünün tavsiye edilip edilmeyeceğini tahmin ettik (TF-IDF + Logistic Regression, %88.7 doğruluk); SVD ile boyutu 18.573'ten 300'e indirip (%98 sıkışma) doğruluktan sadece ~2 puan kaybettiğimizi gösterdik |
| 02 | [`02-word-embeddings`](./02-word-embeddings) | Word2Vec, GloVe, FastText | Üçünü de sıfırdan PyTorch'ta yazıp aynı text8 korpusunda eğittik; Google'ın analoji test setinde birbirleriyle kıyasladık, FastText'e OOV (görülmemiş kelime) testi yaptırdık |
| 03 | [`03-rnn`](./03-rnn) | Vanilla RNN | HDFS sistem loglarında anomali tespiti — bir log dizisinin normal mi anormal mi olduğunu sınıflandırdık |
| 04 | [`04-lstm`](./04-lstm) | LSTM | NYC taksi talebini zaman serisi olarak tahmin ettik, tahmin hatasından anomali skoru türetip 5 gerçek olayı (maraton, bayramlar, kar fırtınası) yakaladık |
| 05 | [`05-attention`](./05-attention) | BiLSTM + Bahdanau Attention | IMDB film yorumlarında duygu analizi yaptık (%80.6 doğruluk) ve modelin kararını verirken hangi kelimelere baktığını ısı haritasıyla gösterdik |
| 06 | [`06-self-attention-transformer`](./06-self-attention-transformer) | Transformer (Self-Attention) | Karakter düzeyinde bir mini-GPT kurup Shakespeare metinleriyle eğittik, modele yeni "Shakespeare tarzı" metin ürettirdik |

Doğal dil işlemenin temel taşlarını, en basit istatistiksel yöntemden bugünün büyük dil modellerinin temelini oluşturan Transformer mimarisine kadar, **her adımı sıfırdan uygulayarak** öğrenen bir proje serisi. Her klasör bağımsız çalışabilen bir mini proje: gerçek bir veri seti, tek bir Python dosyası, görseller ve sonuçları anlatan bir README içeriyor.

Amaç, hazır kütüphaneleri çağırıp geçmek değil — her mimarinin **neden** var olduğunu, bir öncekinin hangi sorununu çözdüğünü göstermek. Bu yüzden proje serisi kronolojik ve kümülatif bir hikaye anlatıyor:

```
İstatistiksel temsil  →  Öğrenilmiş kelime vektörleri  →  Diziyi anlama  →  Uzun bağımlılık  →  Seçici odaklanma  →  Paralel, recurrence'sız mimari
     (TF-IDF)              (Word2Vec/GloVe/FastText)        (RNN)           (LSTM)            (Attention)          (Transformer)
```

## Neden Bu Sıra?

Her proje, kendinden öncekinin **somut bir sınırlamasını** çözüyor:

- **TF-IDF → Word Embeddings:** TF-IDF bir kelimenin önemini sayar ama anlamını bilmez ("king" ve "queen" için TF-IDF'in gözünde hiçbir ortak nokta yoktur). Word2Vec/GloVe/FastText, kelimeleri anlamca yakın olanların birbirine yakın durduğu bir vektör uzayına yerleştirir.
- **Word Embeddings → RNN:** Kelime vektörleri kelimeleri temsil eder ama bir *cümledeki sırayı* bilmez. RNN, kelimeleri sırayla işleyip bir gizli durum (hidden state) taşıyarak bağlamı biriktirir.
- **RNN → LSTM:** Vanilla RNN, uzun dizilerde erken bilgiyi unutur (kaybolan gradyan problemi). LSTM, unutma/girdi/çıktı kapılarıyla hangi bilginin taşınacağını, hangisinin atılacağını öğrenir.
- **LSTM → Attention:** LSTM/RNN, tüm cümleyi yine de tek bir sabit vektöre sıkıştırır. Attention, modelin karar anında cümledeki HER kelimeye geri dönüp bakmasına ve önem ağırlığı vermesine izin verir.
- **Attention → Transformer:** Attention hâlâ bir RNN/LSTM'in üzerine ekleniyordu — dizi hâlâ sırayla işleniyordu. Transformer, recurrence'ı tamamen kaldırıp SADECE self-attention ile tüm diziyi paralel işler; bu hem çok daha hızlı eğitim hem de çok daha uzun bağımlılıklar demektir.

## Ortak Prensipler

Bütün projeler aynı felsefeyle inşa edildi:

- **Sıfırdan uygulama:** Mimarinin çekirdek fikri (attention skoru, LSTM kapıları, self-attention matrisi vb.) hazır bir kütüphane fonksiyonuna sarılmadan, PyTorch'un temel katmanlarıyla elle kuruldu.
- **Gerçek veri, gerçek sonuç:** Her proje tanınmış, gerçek bir veri setiyle çalışıyor (IMDB, text8, Tiny Shakespeare, NAB, HDFS logları vb.) ve sonuçlar (doğruluk, F1, örnek çıktılar) dürüstçe raporlanıyor — abartılı iddialar yok.
- **Görselleştirme:** Her proje, modelin *ne öğrendiğini* gösteren görseller üretiyor (attention ısı haritaları, t-SNE kümeleri, kayıp eğrileri) — sadece sayı değil, sezgi de veriyor.
- **Bağımsız çalışabilirlik:** Her klasör kendi `requirements.txt` ve verisini indiren kodla birlikte geliyor; herhangi biri tek başına klonlanıp çalıştırılabilir.

## Nasıl Çalıştırılır

Her proje klasörü kendi içinde bağımsızdır:

```bash
cd 01-tf-idf && pip install -r requirements.txt && python <script>.py
cd 02-word-embeddings/word2vec && pip install -r requirements.txt && python word2vec_skipgram.py
cd 03-rnn && pip install -r requirements.txt && python <script>.py
cd 04-lstm && pip install -r requirements.txt && python lstm_taxi_forecast_anomaly.py
cd 05-attention && pip install -r requirements.txt && python attention_sentiment.py
cd 06-self-attention-transformer && pip install -r requirements.txt && python minigpt_transformer.py
```

Veri setleri hiçbir klasörde depoya dahil edilmedi — her script ilgili veriyi ilk çalıştırmada otomatik indirir (`.gitignore` ile `data/` klasörleri hariç tutuluyor).

## Detaylar İçin

Her klasörün kendi `README.md`'si, o projeye özel yöntem açıklaması, sonuç tabloları ve görselleri içeriyor. Bu sayfa sadece genel haritayı ve projeler arasındaki bağı gösteriyor.

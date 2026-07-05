# Yapay Zeka Projeleri

Makine öğrenmesi, derin öğrenme ve doğal dil işleme alanlarında geliştirdiğim projelerin bulunduğu ana repo.

## Yapı

```
softito-yapay-zeka/
├── NLP/                  # Doğal Dil İşleme projeleri
├── MachineLearning/      # Makine öğrenmesi projeleri
├── DeepLearning/         # Derin öğrenme projeleri
├── EDA/                  # Keşifsel veri analizi projeleri
├── LLM/                  # Büyük dil modeli projeleri
├── SLM/                  # Küçük dil modeli projeleri
└── Python/               # Python projeleri
```

## NLP

Doğal dil işlemede temsil yöntemlerini sıfırdan ele alan bir öğrenme yolculuğu. Her bölüm bir önceki yöntemin eksiğini gösterir ve bir sonrakinin neden ortaya çıktığını açıklar.

| # | Proje | Bu Projede Ne Yapıldı | Teknoloji | Durum |
|---|-------|----------|-----------|-------|
| 01 | [TF-IDF](NLP/01-tf-idf) | Kadın giyim e-ticaret yorumlarından (23.486 yorum) ürün tavsiyesi tahmini — TF-IDF + Logistic Regression (%88.7 doğruluk), SVD ile boyut indirgeme kıyası | Scikit-learn | ✅ Hazır |
| 02 | [Kelime Vektörleri](NLP/02-word-embeddings) | Word2Vec, GloVe, FastText'i sıfırdan yazıp aynı korpusta eğittik; analoji testinde birbirleriyle kıyasladık, FastText'e OOV (görülmemiş kelime) testi yaptırdık | PyTorch | ✅ Hazır |
| 03 | [RNN](NLP/03-rnn) | HDFS sistem loglarında anomali tespiti — bir log dizisinin normal mi anormal mi olduğunu sınıflandırdık | PyTorch | ✅ Hazır |
| 04 | [LSTM](NLP/04-lstm) | NYC taksi talebini zaman serisi olarak tahmin ettik, tahmin hatasından anomali skoru türetip 5 gerçek olayı (maraton, bayramlar, kar fırtınası) yakaladık | PyTorch | ✅ Hazır |
| 05 | [Attention](NLP/05-attention) | IMDB film yorumlarında duygu analizi (BiLSTM + Bahdanau Attention, %80.6 doğruluk) — modelin kararını verirken hangi kelimelere baktığını ısı haritasıyla gösterdik | PyTorch | ✅ Hazır |
| 06 | [Transformer](NLP/06-self-attention-transformer) | Karakter düzeyinde bir mini-GPT kurup Shakespeare metinleriyle eğittik, modele yeni "Shakespeare tarzı" metin ürettirdik | PyTorch | ✅ Hazır |

## Machine Learning

| Proje | Açıklama | Teknoloji |
|-------|----------|-----------|
| [Fashion MNIST CNN](MachineLearning/Fashion_MNIST_CNN) | Görüntü sınıflandırma | CNN |
| [ML Comparison](MachineLearning/ML_Comparison) | Model karşılaştırması | Scikit-learn |
| [Mobile Price](MachineLearning/mobile_price) | Fiyat tahmini | Scikit-learn |

## Teknolojiler

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

## Lisans

MIT

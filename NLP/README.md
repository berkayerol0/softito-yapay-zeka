# NLP Anatomisi: Klasik Yöntemlerden Transformer'a

Doğal dil işlemede temsil yöntemlerini sıfırdan ele alan bir öğrenme yolculuğu. Her bölüm bir önceki yöntemin **eksiğini** gösterir ve bir sonrakinin neden ortaya çıktığını açıklar: bag-of-words'ten kelime vektörlerine, oradan dizisel modellere (RNN/LSTM) ve nihayet attention / Transformer'a.

## Yol Haritası

Bölümler öğrenme sırasına göre numaralandırılmıştır. Her klasörün kendi `README.md`'si o konuyu ve çözdüğü problemi anlatır.

| # | Konu | Durum | Çözdüğü problem |
|---|------|-------|-----------------|
| 01 | [TF-IDF](01-tf-idf/) | ✅ Hazır | Metni sayısallaştırma (baseline) |
| 02 | Kelime Vektörleri (Word2Vec, GloVe, FastText) | 🔜 | TF-IDF'in anlamsal körlüğü |
| 03 | RNN | 🔜 | Kelime sırası / dizisel bağlam |
| 04 | LSTM / GRU | 🔜 | Uzun vadeli bağımlılık & vanishing gradient |
| 05 | Attention | 🔜 | Uzak kelimeler arası ilişki |
| 06 | Self-Attention & Transformer | 🔜 | Paralellik + bağlamsal temsil |

## Kurulum

```bash
git clone https://github.com/berkayerol0/NLP.git
cd NLP
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Kullanım

Bölümler ağırlıklı olarak Jupyter / Google Colab not defterleri (veya Colab'dan dışa aktarılmış `.py` dosyaları) olarak tasarlanmıştır. İlgili klasöre girip dosyayı açman yeterli. Colab kullanıyorsan dosyayı doğrudan Colab'da açıp hücreleri sırayla çalıştırabilirsin.

## Lisans

MIT — bkz. [LICENSE](LICENSE).

# 02 — Fine-Tuning (LoRA)

Bu klasör, **LoRA (Low-Rank Adaptation)** yöntemiyle küçük bir açık kaynak dil modelinin
(`distilgpt2`) farklı görevlere nasıl uyarlanabileceğini gösteren iki bağımsız demo proje
içerir. Her iki proje de aynı temel LoRA mekanizmasını kullanır, ama farklı görev türlerinde
test eder — böylece LoRA'nın hem doğal dil üretiminde hem de yapılandırılmış çıktı
(structured output) görevlerinde çalıştığı gösterilir.

## İçindekiler

| Klasör | Görev Türü | Özet |
|---|---|---|
| [`01-telekom-asistan/`](01-telekom-asistan/) | Doğal dil üretimi | Sinirli müşteri şikayetlerine kurumsal, sakinleştirici tonda cevap üreten bir destek asistanı |
| [`02-finans-siniflandirici/`](02-finans-siniflandirici/) | Yapılandırılmış çıktı (JSON) | Serbest metin banka/POS loglarını yapılandırılmış JSON'a dönüştüren bir bilgi çıkarım modeli |

## Ortak Yöntem

Her iki projede de:
- Temel model: `distilgpt2` (küçük, hızlı, CPU'da bile rahatça çalışan bir GPT-2 varyantı)
- LoRA adaptörleri `c_attn` ve `c_proj` katmanlarına eklenmiş, orijinal model ağırlıkları
  dondurulmuş — sadece küçük bir adaptör katmanı eğitiliyor
- 4 örnekten oluşan sentetik, göreve özel bir veri seti kullanılmış
- 200 adım boyunca eğitim yapılmış ve **fine-tuning öncesi/sonrası** aynı prompt üzerinde
  karşılaştırılmış
- Sonuçlar (loss grafiği + öncesi/sonrası çıktı) her projenin kendi `figures/` klasöründe
  CSV ve PNG olarak saklanıyor

## Genel Bulgu

İki proje birlikte şunu gösteriyor: LoRA, çok az veriyle bile modelin **davranışını**
(ton, format) hızla değiştirebiliyor — ama bu değişim genelleme değil, büyük ölçüde
**ezberleme (memorization)**. 01'de model kurumsal bir üslup kalıbına yöneliyor ama
gramer bozuk çıkıyor; 02'de model JSON formatını kusursuz öğreniyor ama içerik değerlerini
yanlış eşleştiriyor. Bu, gerçek dünya kullanımında **veri setinin ölçeğinin ve çeşitliliğinin**
neden bu kadar kritik olduğunu somut biçimde gösteriyor.

## Donanım Notu

Her iki script de CUDA'ya özel bir çağrı içermez (`model.device` ile otomatik cihaz algılar).
NVIDIA GPU olmadan, **AMD ekran kartlarında veya sadece CPU üzerinde** sorunsuz çalışır;
model küçük olduğu için eğitim birkaç dakika içinde tamamlanır.

## Çalıştırma

Her iki alt proje kendi `requirements.txt` ve script'ine sahiptir, birbirinden bağımsız
çalıştırılabilir:

```bash
cd 01-telekom-asistan
pip install -r requirements.txt
python telekom_asistan_lora.py
```

```bash
cd 02-finans-siniflandirici
pip install -r requirements.txt
python finans_lora.py
```

Detaylı sonuçlar ve analiz için her alt projenin kendi README.md dosyasına bakınız.

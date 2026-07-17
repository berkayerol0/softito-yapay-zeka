# Basitleştirilmiş DPO (Direct Preference Optimization) Demo

Bu proje, RLHF'in (Reinforcement Learning from Human Feedback) günümüzde en yaygın
kullanılan alternatiflerinden biri olan **DPO**'yu küçük bir açık kaynak model
(`distilgpt2`) üzerinde gösterir. Klasik RLHF'in ayrı bir ödül modeli (reward model) ve
pekiştirmeli öğrenme aşaması gerektirmesine karşın, DPO tercih verisini doğrudan bir
sınıflandırma benzeri kayıp fonksiyonuyla kullanır — daha basit ve daha stabil bir alternatif.

## Amaç

Modele, aynı soruya verilebilecek **kibar/yardımcı** bir cevabı, **kaba/yardımsız** bir
cevaba tercih etmeyi öğretmek. Bu, müşteri destek asistanlarının "davranış hizalaması"
(behavior alignment) için kullanılan gerçek bir teknik.

## Yöntem

1. 4 adet `prompt / chosen (tercih edilen) / rejected (tercih edilmeyen)` üçlüsünden oluşan
   sentetik bir veri seti hazırlandı.
2. `distilgpt2` üzerinde `trl` kütüphanesinin `DPOTrainer`'ı ile 15 epoch boyunca DPO
   eğitimi yapıldı (`beta=0.1`).
3. Aynı test prompt'u hem **DPO öncesi** hem **DPO sonrası** modele verilerek davranış
   farkı gözlemlendi.

## Sonuçlar

Script çalıştırıldığında `figures/` klasörüne şunlar üretilir:
- `dpo_loss_curve.png` / `dpo_loss_log.csv` — eğitim kaybı
- `oncesi_sonrasi_karsilastirma.csv` — DPO öncesi/sonrası çıktı karşılaştırması

*(Script yerel olarak eğitim çalıştırdığı için burada örnek çıktı henüz yok — çalıştırıp
sonuçları paylaşınca bu bölüm gerçek verilerle güncellenecektir.)*

## Notlar / Sınırlamalar

- Veri seti sadece 4 tercih çiftinden oluştuğu için model, genel bir "kibarlık" kavramını
  öğrenmek yerine bu örneklerdeki yüzeysel kalıpları ezberler. Gerçek bir DPO hattı
  binlerce/on binlerce tercih çiftine ihtiyaç duyar (ör. Anthropic'in HH-RLHF veri seti).
- `beta` parametresi, modelin referans modelden ne kadar sapabileceğini kontrol eder;
  düşük değerler daha muhafazakâr, yüksek değerler daha agresif bir hizalama üretir.
- API key gerekmez, tamamen yerel çalışır (CPU dahil, AMD ekran kartlarında sorunsuz).
- Tekrarlanabilirlik için `seed=42` sabitlenmiştir.

## Çalıştırma

```bash
pip install -r requirements.txt
python dpo_egitimi.py
```

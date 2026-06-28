# 05 — Attention

🔜 Yakında.

## Bu bölüm ne anlatacak?

LSTM uzun dizilerde bile tüm geçmişi tek bir sabit boyutlu gizli duruma (hidden state) sıkıştırmak zorundadır — bu bir darboğazdır. Attention mekanizması, modelin çıktı üretirken girdideki **her kelimeye doğrudan, ağırlıklı olarak bakmasına** izin verir; böylece uzak kelimeler arasındaki ilişkiler daha iyi yakalanır.

## Planlanan içerik

- Encoder-decoder mimarisinin darboğazı (bottleneck problemi)
- Attention skorlarının nasıl hesaplandığı (query, key, value mantığına giriş)
- Bahdanau / Luong attention örnekleri
- Attention ağırlıklarının görselleştirilmesi (hangi kelime hangisine "bakıyor")
- Bir sonraki adım: Self-Attention ve Transformer'a geçiş

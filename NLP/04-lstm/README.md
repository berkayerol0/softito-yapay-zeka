# 04 — LSTM / GRU

🔜 Yakında.

## Bu bölüm ne anlatacak?

RNN, uzun dizilerde geçmiş bilgiyi unutur (vanishing gradient) veya patlar (exploding gradient). LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit), kapı (gate) mekanizmalarıyla hangi bilginin tutulup hangisinin unutulacağını öğrenerek bu problemi büyük ölçüde çözer.

## Planlanan içerik

- RNN'in unutma problemi: somut bir örnekle gösterim
- LSTM hücresinin yapısı: forget gate, input gate, output gate
- GRU ile LSTM'in karşılaştırılması (daha az parametre, benzer performans)
- Uygulamalı örnek: duygu analizi veya sıradaki kelimeyi tahmin etme
- LSTM'in limiti: hâlâ sıralı işlem yapar (paralelleştirilemez), çok uzun dizilerde hâlâ zorlanır → bir sonraki bölüm (Attention) bunu çözer

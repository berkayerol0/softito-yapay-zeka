"""
Tokenize edilmis metni PyTorch Dataset/DataLoader formatina cevirir.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import KarakterTokenizer


class MetinDataset(Dataset):
    """Belirli blok uzunlugunda (giris, hedef) ciftleri ureten dataset.
    Hedef, girisin bir karakter kaydirilmis halidir (next-char prediction)."""

    def __init__(self, veri: torch.Tensor, blok_uzunlugu: int):
        self.veri = veri
        self.blok_uzunlugu = blok_uzunlugu

    def __len__(self) -> int:
        return len(self.veri) - self.blok_uzunlugu

    def __getitem__(self, idx: int):
        giris = self.veri[idx: idx + self.blok_uzunlugu]
        hedef = self.veri[idx + 1: idx + self.blok_uzunlugu + 1]
        return giris, hedef


def dataset_hazirla(
    temiz_metin_yolu: str,
    tokenizer_yolu: str,
    blok_uzunlugu: int,
    batch_boyutu: int,
    egitim_orani: float,
):
    """Metni okuyup egitim/dogrulama DataLoader'larini ve tokenizer'i dondurur."""
    with open(temiz_metin_yolu, "r", encoding="utf-8") as f:
        metin = f.read()

    tokenizer = KarakterTokenizer.yukle(tokenizer_yolu)
    veri = torch.tensor(tokenizer.encode(metin), dtype=torch.long)

    bolme_noktasi = int(len(veri) * egitim_orani)
    egitim_verisi = veri[:bolme_noktasi]
    dogrulama_verisi = veri[bolme_noktasi:]

    egitim_dataset = MetinDataset(egitim_verisi, blok_uzunlugu)
    dogrulama_dataset = MetinDataset(dogrulama_verisi, blok_uzunlugu)

    egitim_loader = DataLoader(egitim_dataset, batch_size=batch_boyutu, shuffle=True)
    dogrulama_loader = DataLoader(dogrulama_dataset, batch_size=batch_boyutu, shuffle=False)

    return egitim_loader, dogrulama_loader, tokenizer

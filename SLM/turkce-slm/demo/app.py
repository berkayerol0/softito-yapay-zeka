"""
Turkce SLM icin Gradio demo arayuzu. Egitilmis Transformer modelini
yukleyip verilen baslangic metnine gore devam metni uretir.

Kullanim:
    python demo/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gradio as gr
import torch

from src.generate import metin_uret, model_yukle
from src.preprocessing import KarakterTokenizer

CHECKPOINT_YOLU = "checkpoints/slm_model.pt"
TOKENIZER_YOLU = "data/processed/tokenizer.json"

_tokenizer = None
_model = None


def _modeli_yukle():
    global _tokenizer, _model
    if _model is None:
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = KarakterTokenizer.yukle(Path(TOKENIZER_YOLU))
        _model = model_yukle(CHECKPOINT_YOLU, _tokenizer.vocab_boyutu, cihaz)
    return _tokenizer, _model


def metin_tamamla(prompt: str, uzunluk: int, sicaklik: float) -> str:
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = _modeli_yukle()
    return metin_uret(model, tokenizer, prompt, uzunluk, sicaklik, cihaz)


with gr.Blocks(title="Turkce SLM Demo") as demo:
    gr.Markdown("# Turkce Kucuk Dil Modeli Demo")

    prompt_girisi = gr.Textbox(label="Baslangic metni", placeholder="Yapay zeka...")
    uzunluk_kaydirici = gr.Slider(50, 500, value=300, step=10, label="Uretilecek karakter sayisi")
    sicaklik_kaydirici = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Sicaklik (yaraticilik)")
    tamamla_buton = gr.Button("Metni Tamamla")
    tamamlama_ciktisi = gr.Textbox(label="Uretilen metin", lines=8)

    tamamla_buton.click(
        fn=metin_tamamla,
        inputs=[prompt_girisi, uzunluk_kaydirici, sicaklik_kaydirici],
        outputs=tamamlama_ciktisi,
    )


if __name__ == "__main__":
    demo.launch()

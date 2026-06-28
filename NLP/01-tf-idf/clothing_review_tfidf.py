"""
clothing_review_tfidf.py
========================
Kadın giyim e-ticaret yorumlarından "müşteri ürünü tavsiye eder mi?" (Recommended IND: 1/0)
tahmini yapan, uçtan uca tek dosyalık bir metin sınıflandırma projesi.

Akış:
  1) Kütüphaneler
  2) Veriyi Kaggle'dan çekme
  3) Veri temizleme (eksik değerler, mükerrerler)
  4) Keşifsel analiz ve görselleştirme
  5) TF-IDF vektörleştirme + Logistic Regression
  6) Değerlendirme (metrikler, confusion matrix, en etkili kelimeler)
  7) SVD (LSA) ile boyut indirgeme ve performans takası analizi
  8) Örnek tahmin

Çalıştırma:
  python clothing_review_tfidf.py
Üretilenler:
  figures/  -> tüm grafikler (PNG)
  models/   -> eğitilmiş model (.joblib)

Veri seti: Women's E-Commerce Clothing Reviews (Kaggle, 23.486 yorum)
"""

# ======================================================================
# 1) KÜTÜPHANELER
# ======================================================================
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # ekran olmadan da grafik üretebilmek için
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import joblib

# --- Sabitler -----------------------------------------------------------
KAGGLE_DATASET = "nicapotato/womens-ecommerce-clothing-reviews"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "Womens Clothing E-Commerce Reviews.csv")
FIG_DIR = "figures"
MODEL_PATH = os.path.join("models", "model.joblib")

TITLE_COL = "Title"
BODY_COL = "Review Text"
TARGET_COL = "Recommended IND"

RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 120


# ======================================================================
# 2) VERİYİ KAGGLE'DAN ÇEKME
# ======================================================================
def ensure_dataset():
    """
    CSV yoksa Kaggle API ile indirir. Kaggle API kurulu/yetkili değilse
    elle indirme talimatı verir.
    (Kaggle API: pip install kaggle + ~/.kaggle/kaggle.json)
    """
    if os.path.exists(CSV_PATH):
        print(f"[veri] Mevcut: {CSV_PATH}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[veri] Bulunamadı, Kaggle'dan indiriliyor: {KAGGLE_DATASET}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", DATA_DIR, "--unzip"],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(
            "\n[HATA] Kaggle'dan otomatik indirme başarısız.\n"
            "Çözüm:\n"
            "  1) pip install kaggle\n"
            "  2) Kaggle > Account > Create New API Token ile kaggle.json indirin\n"
            "     ve ~/.kaggle/kaggle.json konumuna koyun.\n"
            "  3) Ya da elle indirin:\n"
            f"     https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"     ve CSV'yi şu yola koyun: {CSV_PATH}\n"
        )

    if not os.path.exists(CSV_PATH):
        # bazı sürümlerde dosya adı farklı inebilir; klasördeki ilk csv'yi bul
        csvs = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if csvs:
            os.rename(os.path.join(DATA_DIR, csvs[0]), CSV_PATH)
    print(f"[veri] İndirildi: {CSV_PATH}")


# ======================================================================
# 3) VERİ TEMİZLEME
# ======================================================================
_ws = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Tek metni normalize eder: küçük harf + fazla boşlukları sadeleştirme."""
    if not isinstance(text, str):
        return ""
    return _ws.sub(" ", text.lower().strip())


def load_and_clean():
    """Ham CSV'yi yükler, temizler; (df, raw_df, rapor) döndürür."""
    raw = pd.read_csv(CSV_PATH)
    report = {"ham_satir": len(raw), "ham_sutun": raw.shape[1],
              "eksik": raw.isnull().sum().to_dict()}

    df = raw.drop(columns=[c for c in raw.columns if str(c).startswith("Unnamed")],
                  errors="ignore").copy()

    before = len(df)
    df = df.dropna(subset=[BODY_COL])                 # yorum metni yoksa sınıflandıramayız
    report["bos_yorum_atilan"] = before - len(df)

    df[TITLE_COL] = df.get(TITLE_COL, "").fillna("")
    df["text"] = (df[TITLE_COL].astype(str) + " " + df[BODY_COL].astype(str)).map(clean_text)

    before = len(df)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    report["mukerrer_bos_atilan"] = before - len(df)

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    # yorum uzunluğu (EDA için)
    df["review_len"] = df[BODY_COL].astype(str).str.split().map(len)

    report["temiz_satir"] = len(df)
    report["pozitif_oran"] = float(df[TARGET_COL].mean())
    return df.reset_index(drop=True), raw, report


def print_report(r):
    print("=" * 56)
    print("VERİ TEMİZLEME RAPORU")
    print("=" * 56)
    print(f"Ham veri            : {r['ham_satir']} satır, {r['ham_sutun']} sütun")
    print("Eksik değerler (ham):")
    for c, n in r["eksik"].items():
        if n > 0:
            print(f"   - {c:<25}: {n}")
    print(f"Atılan (boş yorum)  : {r['bos_yorum_atilan']}")
    print(f"Atılan (mükerrer/boş): {r['mukerrer_bos_atilan']}")
    print(f"Temiz veri          : {r['temiz_satir']} satır")
    print(f"Pozitif sınıf oranı : %{r['pozitif_oran']*100:.1f} (dengesiz)")
    print("=" * 56)


# ======================================================================
# 4) KEŞİFSEL ANALİZ / GÖRSELLEŞTİRME
# ======================================================================
def plot_eda(df, raw, report):
    """Sınıf dağılımı, eksik değerler ve yorum uzunluğu görselleri."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) Sınıf dağılımı
    counts = df[TARGET_COL].value_counts().sort_index()
    axes[0].bar(["Tavsiye etmez (0)", "Tavsiye eder (1)"], counts.values,
                color=["#d9534f", "#5cb85c"])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}\n%{100*v/counts.sum():.1f}", ha="center", va="bottom")
    axes[0].set_title("Sınıf Dağılımı (dengesiz)")
    axes[0].set_ylabel("Yorum sayısı")
    axes[0].set_ylim(0, counts.max() * 1.18)

    # (b) Eksik değerler (ham)
    miss = {k: v for k, v in report["eksik"].items() if v > 0}
    axes[1].barh(list(miss.keys()), list(miss.values()), color="#f0ad4e")
    for i, v in enumerate(miss.values()):
        axes[1].text(v, i, f" {v}", va="center")
    axes[1].set_title("Ham Veride Eksik Değerler")
    axes[1].invert_yaxis()

    # (c) Yorum uzunluğu dağılımı (sınıfa göre)
    for cls, color, lab in [(1, "#5cb85c", "Tavsiye eder"), (0, "#d9534f", "Tavsiye etmez")]:
        axes[2].hist(df.loc[df[TARGET_COL] == cls, "review_len"], bins=40,
                     alpha=0.6, color=color, label=lab)
    axes[2].set_title("Yorum Uzunluğu (kelime)")
    axes[2].set_xlabel("Kelime sayısı")
    axes[2].set_xlim(0, 120)
    axes[2].legend()

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "01_eda.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[görsel] EDA kaydedildi -> {out}")


# ======================================================================
# 5) TF-IDF + MODEL  (yardımcı)
# ======================================================================
def build_tfidf(max_features):
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                           min_df=5, max_features=max_features, sublinear_tf=True)


# ======================================================================
# 6) DEĞERLENDİRME GÖRSELLERİ
# ======================================================================
def plot_confusion(y_true, y_pred):
    os.makedirs(FIG_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    labs = ["Tavsiye etmez", "Tavsiye eder"]
    ax.set_xticks([0, 1], labels=labs); ax.set_yticks([0, 1], labels=labs)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix (TF-IDF + LogReg)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "02_confusion_matrix.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] Confusion matrix kaydedildi -> {out}")


def plot_top_words(vectorizer, clf, n=15):
    """LogReg katsayılarından en etkili pozitif/negatif kelimeler (yorumlanabilirlik)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos = np.argsort(coefs)[-n:]
    top_neg = np.argsort(coefs)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(names[top_pos], coefs[top_pos], color="#5cb85c")
    axes[0].set_title("Tavsiyeye en çok iten kelimeler")
    axes[1].barh(names[top_neg], coefs[top_neg], color="#d9534f")
    axes[1].set_title("Tavsiyeden en çok caydıran kelimeler")
    for ax in axes:
        ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "03_top_words.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] En etkili kelimeler kaydedildi -> {out}")


# ======================================================================
# 7) SVD ANALİZİ + GÖRSEL
# ======================================================================
def svd_analysis(X_tr, X_te, y_tr, y_te, max_features, components):
    """Farklı SVD bileşen sayıları için doğruluk/F1 ve açıklanan varyansı ölçer."""
    os.makedirs(FIG_DIR, exist_ok=True)
    accs, f1s, evrs = [], [], []
    for k in components:
        pipe = make_pipeline(
            build_tfidf(max_features),
            TruncatedSVD(n_components=k, random_state=RANDOM_STATE),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        accs.append(accuracy_score(y_te, pred))
        f1s.append(f1_score(y_te, pred, average="macro"))
        evrs.append(pipe.named_steps["truncatedsvd"].explained_variance_ratio_.sum())
        print(f"   SVD k={k:<4} -> acc {accs[-1]:.4f} | F1 {f1s[-1]:.4f} | varyans %{100*evrs[-1]:.1f}")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(components, accs, "o-", color="#0275d8", label="Accuracy")
    ax1.plot(components, f1s, "s-", color="#5cb85c", label="F1 (macro)")
    ax1.set_xlabel("SVD bileşen sayısı (boyut)")
    ax1.set_ylabel("Skor")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(components, [100*e for e in evrs], "^--", color="#f0ad4e", label="Açıklanan varyans %")
    ax2.set_ylabel("Açıklanan varyans (%)", color="#f0ad4e")
    ax1.set_title("SVD: Boyut ↓ ↔ Performans Takası")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "04_svd_tradeoff.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"[görsel] SVD takas grafiği kaydedildi -> {out}")
    return accs, f1s, evrs


# ======================================================================
# ANA AKIŞ
# ======================================================================
def main():
    MAX_FEATURES = 20000
    SVD_COMPONENTS = [50, 100, 200, 300, 500]

    # 2) Veri
    ensure_dataset()

    # 3) Temizlik
    df, raw, report = load_and_clean()
    print_report(report)

    # 4) EDA görselleri
    plot_eda(df, raw, report)

    X = df["text"].values
    y = df[TARGET_COL].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"\nEğitim: {len(X_tr)} | Test: {len(X_te)}")

    # 5) TF-IDF + LogReg
    model = make_pipeline(
        build_tfidf(MAX_FEATURES),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    vec = model.named_steps["tfidfvectorizer"]
    clf = model.named_steps["logisticregression"]
    n_feat = len(vec.get_feature_names_out())

    # 6) Değerlendirme
    acc = accuracy_score(y_te, pred)
    f1m = f1_score(y_te, pred, average="macro")
    print("\n" + "=" * 56)
    print("TF-IDF + LOGISTIC REGRESSION")
    print("=" * 56)
    print(f"Öznitelik (kelime) sayısı: {n_feat}")
    print(f"Accuracy : {acc:.4f}   F1 (macro): {f1m:.4f}")
    print(classification_report(y_te, pred,
          target_names=["Tavsiye etmez (0)", "Tavsiye eder (1)"]))
    plot_confusion(y_te, pred)
    plot_top_words(vec, clf)

    # 7) SVD analizi
    print("\n" + "=" * 56)
    print("SVD (LSA) İLE BOYUT İNDİRGEME ANALİZİ")
    print("=" * 56)
    accs, f1s, evrs = svd_analysis(X_tr, X_te, y_tr, y_te, MAX_FEATURES, SVD_COMPONENTS)
    k = SVD_COMPONENTS[len(SVD_COMPONENTS)//2]
    idx = SVD_COMPONENTS.index(k)
    print(f"\nÖrnek: {n_feat} -> {k} boyut (%{100*(1-k/n_feat):.1f} sıkışma) ile "
          f"accuracy {acc:.4f} -> {accs[idx]:.4f}.")
    print("Yorum: Boyut çok büyük oranda düşerken doğrulukta yalnızca küçük bir kayıp var.")

    # 8) Model kaydı + örnek tahmin
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[model] Kaydedildi -> {MODEL_PATH}")

    print("\nÖRNEK TAHMİNLER:")
    for s in ["this dress is gorgeous, fits perfectly and i love the fabric",
              "very cheap material, fell apart after one wash, returned it"]:
        p = model.predict([clean_text(s)])[0]
        prob = model.predict_proba([clean_text(s)])[0][1]
        print(f"   '{s[:45]}...' -> {'TAVSIYE EDER' if p else 'TAVSIYE ETMEZ'} (%{prob*100:.1f})")


if __name__ == "__main__":
    main()

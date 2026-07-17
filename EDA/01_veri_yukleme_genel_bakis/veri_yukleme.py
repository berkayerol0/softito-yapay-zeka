# -*- coding: utf-8 -*-
"""
01 - VERİ YÜKLEME & GENEL BAKIŞ
================================
Bu klasör tamamen BAĞIMSIZDIR — diğer klasörlerdeki hiçbir dosyaya ihtiyaç duymaz.
Kaggle'dan (argonnxx/teen-mental-health) veriyi kagglehub ile OTOMATİK indirir,
genel yapıyı inceler. Tüm görseller bu klasördeki figures/ altına kaydedilir.
"""

import os
import warnings

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
sns.set_style("whitegrid")

KAGGLE_DATASET = "argonnxx/teen-mental-health"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> kaydedildi: {path}")


def load_dataset():
    """Kaggle'dan otomatik indirir ve DataFrame olarak döner."""
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset indirildi:", dataset_path)
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("İndirilen klasörde .csv dosyası bulunamadı.")
    csv_path = os.path.join(dataset_path, csv_files[0])
    print("Okunan dosya:", csv_path)
    return pd.read_csv(csv_path)


def main():
    print("=" * 70)
    print("01 - VERİ YÜKLEME & GENEL BAKIŞ")
    print("=" * 70)

    df = load_dataset()

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nSon 5 satır:")
    print(df.tail())

    print("\nRastgele 5 satır:")
    print(df.sample(5))

    print("\nBoyut (satır, sütun):", df.shape)

    dtype_df = pd.DataFrame({
        "sutun": df.columns,
        "veri_tipi": df.dtypes.values,
        "null_sayisi": df.isnull().sum().values,
        "null_orani(%)": (df.isnull().sum().values / len(df) * 100).round(2),
        "unique": df.nunique().values,
    })
    print("\nSütun özet tablosu:")
    print(dtype_df.to_string(index=False))

    print("\ndf.info():")
    df.info()

    print("\nSayısal değişkenler için istatistik özeti:")
    print(df.describe().T)

    if df.select_dtypes(include=["object", "category"]).shape[1] > 0:
        print("\nKategorik değişkenler için özet:")
        print(df.describe(include=["object", "category"]).T)

    # --- Görsel: eksik değer haritası ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Eksik Değer Haritası (Sarı = Eksik)")
    save_fig("01_eksik_deger_haritasi.png")

    # --- Görsel: sütun bazında eksik değer oranı ---
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind="barh", color="#E67E22")
        plt.title("Sütun Bazında Eksik Değer Sayısı")
        plt.xlabel("Eksik Değer Sayısı")
        save_fig("01_sutun_eksik_deger_sayisi.png")
    else:
        print("\nEksik değer bulunmuyor, eksik değer bar grafiği atlandı.")


if __name__ == "__main__":
    main()

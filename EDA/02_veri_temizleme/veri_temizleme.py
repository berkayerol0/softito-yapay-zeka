# -*- coding: utf-8 -*-
"""
02 - VERİ TEMİZLEME
=====================
Bu klasör tamamen BAĞIMSIZDIR — diğer klasörlerdeki hiçbir dosyaya ihtiyaç duymaz.
Veriyi kagglehub ile kendisi indirir, temizler ve tüm görselleri
bu klasördeki figures/ altına kaydeder.
"""

import os
import warnings

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
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
    print("02 - VERİ TEMİZLEME")
    print("=" * 70)

    df = load_dataset()
    print("Ham veri:", df.shape)

    df_clean = df.copy()

    # --- Duplicate kontrolü ---
    dup_count = df_clean.duplicated().sum()
    print(f"\nTekrarlanan (duplicate) satır sayısı: {dup_count}")
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates()
        print("Duplicate satırlar silindi. Yeni boyut:", df_clean.shape)

    # --- Eksik değer analiz tablosu ---
    empty_values = df_clean.isnull().sum()
    empty_rate = (empty_values / len(df_clean) * 100).round(2)
    empty_df = pd.concat([empty_values, empty_rate], axis=1)
    empty_df.columns = ["eksik_sayisi", "eksik_orani(%)"]
    empty_df = empty_df.sort_values("eksik_orani(%)", ascending=False)
    print("\nEksik değer tablosu:")
    print(empty_df)

    # --- Eksik değer doldurma ---
    num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"'{col}' -> medyan ({median_val:.2f}) ile dolduruldu.")

    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"'{col}' -> mod ('{mode_val}') ile dolduruldu.")

    if df_clean.isnull().sum().sum() == 0:
        print("\nEksik değer kalmadı / zaten yoktu.")

    # --- Aykırı değer: boxplot görselleştirme ---
    if len(num_cols) > 0:
        n = len(num_cols)
        cols_per_row = 3
        rows = int(np.ceil(n / cols_per_row))
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(num_cols):
            sns.boxplot(y=df_clean[col], ax=axes[i], color="#3498DB")
            axes[i].set_title(f"{col} - Aykırı Değer Kontrolü")

        for j in range(len(num_cols), len(axes)):
            axes[j].axis("off")

        save_fig("02_aykiri_deger_boxplot.png")

        print("\nIQR yöntemi ile aykırı değer sayıları:")
        for col in num_cols:
            q1, q3 = df_clean[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
            print(f"  {col}: {len(outliers)} aykırı değer")


if __name__ == "__main__":
    main()

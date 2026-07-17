# -*- coding: utf-8 -*-
"""
04 - ÇİFT DEĞİŞKENLİ ANALİZ (Bivariate)
=========================================
Bu klasör tamamen BAĞIMSIZDIR — diğer klasörlerdeki hiçbir dosyaya ihtiyaç duymaz.
Veriyi kagglehub ile kendisi indirir, kendi içinde temizler ve
çift değişkenli analiz görsellerini bu klasördeki figures/ altına kaydeder.
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


def load_and_clean():
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    csv_path = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(csv_path)

    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def main():
    print("=" * 70)
    print("04 - ÇİFT DEĞİŞKENLİ ANALİZ")
    print("=" * 70)

    df = load_and_clean()
    print("Veri hazır:", df.shape)

    num_cols = list(df.select_dtypes(include=["int64", "float64"]).columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    # --- 1) Korelasyon heatmap (sayısal - sayısal) ---
    if len(num_cols) > 1:
        plt.figure(figsize=(12, 9))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
        plt.title("Sayısal Değişkenler Arası Korelasyon Matrisi")
        save_fig("04_korelasyon_heatmap.png")

    # --- 2) Cinsiyete göre scatter plot ---
    gender_col = next((c for c in cat_cols if c.lower() in ("gender", "cinsiyet")), None)
    if gender_col and len(num_cols) >= 2:
        x_col, y_col = num_cols[0], num_cols[1]
        fig, ax = plt.subplots(figsize=(9, 6))
        for grup, renk in zip(df[gender_col].unique(), ["#3498DB", "#E74C3C", "#2ECC71"]):
            alt = df[df[gender_col] == grup]
            ax.scatter(alt[x_col], alt[y_col], label=str(grup), alpha=0.6, s=30, color=renk)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col} ({gender_col} bazında)")
        ax.legend()
        save_fig("04_scatter_iliski.png")

    # --- 3) Kategorik x Kategorik - crosstab stacked bar ---
    if len(cat_cols) >= 2:
        c1, c2 = cat_cols[0], cat_cols[1]
        ct_pct = pd.crosstab(df[c1], df[c2], normalize="index") * 100
        ct_pct.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
        plt.title(f"{c1} - {c2} İlişkisi (Yüzde)")
        plt.ylabel("Yüzde")
        plt.xticks(rotation=45)
        plt.legend(title=c2, bbox_to_anchor=(1.02, 1), loc="upper left")
        save_fig("04_kategorik_kategorik_iliski.png")

    # --- 4) Kategorik x Sayısal - violin / box plot ---
    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        c1 = cat_cols[0]
        target_num = num_cols[-1]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.violinplot(data=df, x=c1, y=target_num, ax=axes[0], palette="Set2")
        axes[0].set_title(f"{c1} - {target_num} (Violin)")
        axes[0].tick_params(axis="x", rotation=45)

        c2 = cat_cols[1] if len(cat_cols) > 1 else c1
        sns.boxplot(data=df, x=c2, y=num_cols[0], ax=axes[1], palette="Set1")
        axes[1].set_title(f"{c2} - {num_cols[0]} (Box)")
        axes[1].tick_params(axis="x", rotation=45)

        save_fig("04_kategorik_sayisal_iliski.png")


if __name__ == "__main__":
    main()

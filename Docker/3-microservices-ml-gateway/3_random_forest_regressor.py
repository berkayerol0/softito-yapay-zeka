"""
ALGORITMA 3: Random Forest Regressor (Regresyon) - Flask Servisi
Gorev: Ucus fiyatini (price) tahmin etmek
Dataset: Clean_Dataset.csv
"""
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5003))

model = None
encoders = {}
ozellikler = []


def egit():
    global model, encoders, ozellikler
    df = pd.read_csv("Clean_Dataset.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    df = df.dropna()

    for kolon in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[kolon] = le.fit_transform(df[kolon])
        encoders[kolon] = le

    X = df.drop("price", axis=1)
    y = df["price"]
    ozellikler = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 40)
    print("RANDOM FOREST REGRESSOR SONUCLARI")
    print("=" * 40)
    print(f"R2 Skoru : {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")


def hazirla(data):
    satir = []
    for kolon in ozellikler:
        deger = data.get(kolon)
        if kolon in encoders:
            le = encoders[kolon]
            deger = int(le.transform([deger])[0]) if deger in le.classes_ else 0
        satir.append(deger)
    return [satir]


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "rf", "model_hazir": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model egitilmedi"}), 503
    data = request.get_json(force=True)
    tahmin = float(model.predict(hazirla(data))[0])
    return jsonify({"tahmini_fiyat": round(tahmin, 2)})


if __name__ == "__main__":
    egit()
    app.run(host="0.0.0.0", port=PORT)
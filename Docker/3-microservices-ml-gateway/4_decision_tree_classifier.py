"""
ALGORITMA 4: Decision Tree Classifier (Siniflandirma) - Flask Servisi
Gorev: Ucusu "pahali (1)" / "ucuz (0)" olarak siniflandirmak
Dataset: Clean_Dataset.csv
"""
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5004))

model = None
encoders = {}
ozellikler = []
esik = 0


def egit():
    global model, encoders, ozellikler, esik
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

    esik = df["price"].median()
    df["pahali"] = (df["price"] > esik).astype(int)

    X = df.drop(["price", "pahali"], axis=1)
    y = df["pahali"]
    ozellikler = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 40)
    print("DECISION TREE CLASSIFIER SONUCLARI")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ucuz", "Pahali"]))


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
    return jsonify({"status": "ok", "service": "dtree", "model_hazir": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model egitilmedi"}), 503
    data = request.get_json(force=True)
    tahmin = int(model.predict(hazirla(data))[0])
    return jsonify({"pahali": tahmin, "etiket": "pahali" if tahmin == 1 else "ucuz",
                    "esik_fiyat": round(float(esik), 2)})


if __name__ == "__main__":
    egit()
    app.run(host="0.0.0.0", port=PORT)
# 🐳 Docker

Docker ile farklı seviyelerde konteynerizasyon pratiği yaptığım projelerin toplandığı repo. Tek konteynerden çok servisli mikroservis mimarisine kadar artan bir karmaşıklıkla ilerliyor.

![Genel Bakış](assets/overview.png)

---

## 📁 Projeler

### 1️⃣ [1-single-container-xgboost](./1-single-container-xgboost)
Tek bir `Dockerfile` ile paketlenmiş, öğrencilerde yapay zekâ (GenAI) kullanımının **tükenmişlik riskine** etkisini tahmin eden bir XGBoost sınıflandırma modeli.

**Gösterdiği şey:** Tek Dockerfile → tek Python dosyası → tek görev.

`Python` · `Docker` · `XGBoost` · `scikit-learn`

---

### 2️⃣ [2-docker-compose-3ML](./2-docker-compose-3ML)
Sağlık sigortası verisiyle **3 farklı ML görevini** (regresyon, sınıflandırma, kümeleme) tek bir Docker imajından, **Docker Compose** ile paralel çalıştıran bir proje.

**Gösterdiği şey:** Aynı imajdan birden fazla servis türetip farklı komutlarla paralel çalıştırma.

`Python` · `Docker Compose` · `XGBoost` · `scikit-learn`

---

### 3️⃣ [3-microservices-ml-gateway](./3-microservices-ml-gateway)
Uçuş fiyatı verisi üzerinde çalışan **5 bağımsız ML servisi** (Flask API) ve bunları tek noktadan yöneten bir **API Gateway** — gerçek bir mikroservis mimarisi örneği.

**Gösterdiği şey:** Bağımsız servisler + servisler arası HTTP iletişimi + API Gateway. Service discovery, authentication gibi production-seviyesi bileşenler kapsam dışıdır.

`Python` · `Flask` · `Docker Compose` · `Microservices`

---

## 🛠️ Ortak Teknolojiler

`Docker` · `Docker Compose` · `Python` · `scikit-learn` · `XGBoost` · `Flask` · `pandas`

---

<p align="center"><i>Docker öğrenme/pratik amaçlı bir portföy reposudur.</i></p>

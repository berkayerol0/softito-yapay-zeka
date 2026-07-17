import os
import csv
import time
import requests
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google.genai.errors import ClientError
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# FAISS için ayrı bir "langchain-faiss" paketi mevcut ama bakımı durmuş (son sürüm 2024).
# langchain-community'nin FAISS entegrasyonu "sunset" uyarısı veriyor ama hâlâ çalışıyor
# ve şu an için en güvenilir seçenek bu; ileride resmi bir standalone paket çıkarsa
# (langchain-google-community veya benzeri) buraya taşınmalı.
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. API Key Kontrolü
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY bulunamadı. Lütfen .env dosyasına "
        "GEMINI_API_KEY=senin-key-in şeklinde ekleyin."
    )

print("=" * 55)
print("  TÜRKÇE WIKIPEDIA TABANLI RAG SİSTEMİ")
print("=" * 55)


def wikipedia_makale_getir(baslik: str, dil: str = "tr") -> dict:
    """Wikipedia REST API'sine doğrudan istek atar (bakımı zayıf 'wikipedia' pip paketi
    yerine). Yönlendirmeleri (redirect) takip eder, makale bulunamazsa hata fırlatır.
    Her istek için taze bir bağlantı kullanır (Connection: close) — art arda hızlı
    isteklerde bazı ağların/proxy'lerin kalıcı bağlantıyı bozması sorununu önlemek için."""
    url = f"https://{dil}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": True,
        "titles": baslik, "format": "json", "redirects": 1,
    }
    headers = {
        "User-Agent": "softito-yapay-zeka-rag-demo/1.0 (kisisel egitim projesi)",
        "Connection": "close",
    }
    yanit = requests.get(url, params=params, headers=headers, timeout=15)
    if yanit.status_code != 200 or not yanit.text.strip():
        raise ValueError(
            f"HTTP {yanit.status_code} — boş veya hatalı yanıt "
            f"(ilk 200 karakter: {yanit.text[:200]!r})"
        )
    veri = yanit.json()
    sayfalar = veri["query"]["pages"]
    sayfa = next(iter(sayfalar.values()))
    if "missing" in sayfa or not sayfa.get("extract"):
        raise ValueError(f"'{baslik}' başlıklı sayfa bulunamadı veya içerik boş.")
    gercek_baslik = sayfa["title"]
    return {
        "baslik": gercek_baslik,
        "icerik": sayfa["extract"],
        "url": f"https://{dil}.wikipedia.org/wiki/{gercek_baslik.replace(' ', '_')}",
    }


# 2. Wikipedia'dan doküman toplama
KONULAR = ["Yapay zeka", "Makine öğrenmesi", "Derin öğrenme"]

docs = []
for konu in KONULAR:
    try:
        sayfa = wikipedia_makale_getir(konu)
        docs.append(Document(
            page_content=sayfa["icerik"],
            metadata={"kaynak": sayfa["baslik"], "url": sayfa["url"]},
        ))
        print(f"[BİLGİ] '{sayfa['baslik']}' makalesi çekildi ({len(sayfa['icerik'])} karakter).")
    except Exception as e:
        print(f"[UYARI] '{konu}' makalesi çekilemedi: {e}")
    time.sleep(1)  # Wikipedia'yı art arda hızlı isteklerle yormamak için

if not docs:
    raise RuntimeError("Hiçbir Wikipedia makalesi çekilemedi, internet bağlantısını kontrol edin.")

# 3. Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"\n[BİLGİ] {len(docs)} makale, {len(chunks)} parçaya (chunk) bölündü.")

# 4. Embedding + FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.2)


def soru_cevapla(soru: str, maks_deneme: int = 3):
    """Retriever ile en alakalı chunk'ları bulur, bunları bağlam olarak LLM'e verir
    ve cevapla birlikte kullanılan kaynakları döner. Ücretsiz katman kota hatası (429)
    alırsa API'nin önerdiği süre kadar bekleyip otomatik tekrar dener."""
    ilgili_chunklar = retriever.invoke(soru)
    baglam = "\n\n".join(
        f"[Kaynak: {c.metadata['kaynak']}]\n{c.page_content}" for c in ilgili_chunklar
    )
    prompt = (
        "Aşağıdaki bağlamı kullanarak soruyu yanıtla. Bağlamda yer almayan bir bilgiyi "
        "uydurma; bağlam yetersizse bunu açıkça belirt.\n\n"
        f"Bağlam:\n{baglam}\n\nSoru: {soru}\n\nCevap:"
    )
    kaynaklar = [c.metadata["kaynak"] for c in ilgili_chunklar]

    for deneme in range(1, maks_deneme + 1):
        try:
            yanit = llm.invoke(prompt)
            return yanit.content, kaynaklar
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and deneme < maks_deneme:
                bekleme = 60
                print(f"[UYARI] Kota limiti (429) — {bekleme} sn beklenip tekrar denenecek "
                      f"({deneme}/{maks_deneme})...")
                time.sleep(bekleme)
            else:
                raise


# 6. Test Soruları
sorular = [
    "Yapay zeka ile makine öğrenmesi arasındaki fark nedir?",
    "Derin öğrenme hangi tür problemlerde kullanılır?",
    "Makine öğrenmesinin başlıca türleri nelerdir?",
]

sonuclar = []
for i, soru in enumerate(sorular, start=1):
    print(f"\n[SORU {i}] {soru}")
    cevap, kaynaklar = soru_cevapla(soru)
    print(f"[CEVAP {i}] {cevap}")
    print(f"[KAYNAK {i}] {kaynaklar}")
    sonuclar.append({"soru": soru, "cevap": cevap, "kaynaklar": " | ".join(kaynaklar)})
    if i < len(sorular):
        time.sleep(3)  # Ücretsiz katman kota limitini zorlamamak için sorular arasında bekleme

# 7. Sonuçları CSV'ye kaydet
csv_yolu = os.path.join(OUTPUT_DIR, "soru_cevap_log.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["soru", "cevap", "kaynaklar"])
    writer.writeheader()
    writer.writerows(sonuclar)
print(f"\n[SİSTEM] Soru-cevap logu kaydedildi: {csv_yolu}")

# 8. Soru başına kullanılan kaynak sayısı grafiği
soru_kisa = [f"S{i+1}" for i in range(len(sorular))]
kaynak_sayilari = [len(set(s["kaynaklar"].split(" | "))) for s in sonuclar]

plt.figure(figsize=(8, 5))
plt.bar(soru_kisa, kaynak_sayilari, color="darkorange")
plt.title("Soru Başına Kullanılan Farklı Makale Sayısı")
plt.xlabel("Soru")
plt.ylabel("Makale Sayısı")
plt.yticks(range(0, len(KONULAR) + 1))
plt.grid(axis="y")
grafik_yolu = os.path.join(OUTPUT_DIR, "kaynak_kullanimi.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Kaynak kullanım grafiği kaydedildi: {grafik_yolu}")

print(
    "\n(Not: RetrievalQA gibi hazır zincir sınıfları yerine, retriever + LLM adımları "
    "elle (LCEL tarzı) birleştirildi; bu hem daha şeffaf hem de LangChain'in eski chain "
    "sınıflarındaki deprecation risklerinden bağımsız bir yaklaşımdır.)"
)
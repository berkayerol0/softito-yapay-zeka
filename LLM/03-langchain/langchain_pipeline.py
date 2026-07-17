import os
import csv
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

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
print("  TEKNİK DESTEK TRİYAJ ASİSTANI (LangChain Agent + Gemini)")
print("=" * 55)

# 2. Bilet deposu (demo için bellek içi, gerçek sistemde bir DB olurdu)
BILET_DEPOSU = []
BILET_SAYAC = {"deger": 1000}


def mesaj_metnini_al(mesaj) -> str:
    """AIMessage.content bazen düz string, bazen {'type': 'text', 'text': ...}
    bloklarından oluşan bir liste olarak gelir (ör. imza/metadata eklenince).
    Her iki durumda da temiz metni döner."""
    icerik = mesaj.content
    if isinstance(icerik, str):
        return icerik
    if isinstance(icerik, list):
        parcalar = []
        for blok in icerik:
            if isinstance(blok, dict) and blok.get("type") == "text":
                parcalar.append(blok.get("text", ""))
            elif isinstance(blok, str):
                parcalar.append(blok)
        return "".join(parcalar)
    return str(icerik)


# 3. Tool 1 — Mesajdaki aciliyet sinyallerine göre öncelik belirleme
@tool
def oncelik_belirle(mesaj: str) -> str:
    """Bir destek mesajının aciliyetini analiz eder ve öncelik seviyesi döner
    (Düşük, Orta, Yüksek, Kritik). Girdi: kullanıcının şikayet/talep metni."""
    mesaj_kucuk = mesaj.lower()
    kritik_kelimeler = ["çöktü", "hiçbir", "tüm müşteri", "veri kaybı", "güvenlik açığı"]
    yuksek_kelimeler = ["çalışmıyor", "acil", "hata veriyor", "erişemiyorum"]
    orta_kelimeler = ["yavaş", "bazen", "arada sırada"]

    if any(k in mesaj_kucuk for k in kritik_kelimeler):
        return "Kritik"
    if any(k in mesaj_kucuk for k in yuksek_kelimeler):
        return "Yüksek"
    if any(k in mesaj_kucuk for k in orta_kelimeler):
        return "Orta"
    return "Düşük"


# 4. Tool 2 — Bilet açma
@tool
def bilet_ac(kategori: str, oncelik: str, aciklama: str) -> str:
    """SADECE kullanıcı açıkça bir destek bileti/talebi açılmasını istediğinde çağrılır.
    Kullanıcı sadece bir sorunu bildiriyorsa (henüz bilet istemediyse) bu aracı ÇAĞIRMA,
    onun yerine sorunu özetleyip önceliğini belirt. Kategori (örn. 'Ödeme Sistemi',
    'Giriş Sorunu'), öncelik (Düşük/Orta/Yüksek/Kritik) ve kısa açıklama gerektirir."""
    bilet_no = f"TCK-{BILET_SAYAC['deger']}"
    BILET_SAYAC["deger"] += 1
    BILET_DEPOSU.append({
        "bilet_no": bilet_no, "kategori": kategori,
        "oncelik": oncelik, "aciklama": aciklama,
    })
    return f"Bilet oluşturuldu: {bilet_no} | Kategori: {kategori} | Öncelik: {oncelik}"


# 5. LLM ve Agent (güncel create_agent API'si + hafıza)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY, temperature=0.2)
checkpointer = InMemorySaver()

SISTEM_PROMPTU = (
    "Sen bir teknik destek triyaj asistanısın. Kullanıcı bir sorun bildirdiğinde önce "
    "oncelik_belirle aracıyla aciliyeti değerlendir ve durumu özetle. "
    "bilet_ac aracını SADECE kullanıcı açıkça 'bilet aç', 'talep oluştur' gibi bir istekte "
    "bulunduğunda çağır — sorun bildirildiği anda otomatik bilet açma."
)

agent = create_agent(
    model=llm,
    tools=[oncelik_belirle, bilet_ac],
    system_prompt=SISTEM_PROMPTU,
    checkpointer=checkpointer,
)
config = {"configurable": {"thread_id": "triaj-oturum-1"}}

# 6. Senaryo — memory (2. tur) ve tool-calling (1. ve 3. tur) test edilir
konusma_senaryosu = [
    "Merhaba, ödeme sistemimiz tamamen çöktü, hiçbir müşteri işlem yapamıyor.",
    "Az önce ne sorun bildirdiğimi ve bunun ne kadar öncelikli olduğunu hatırlıyor musun?",
    "Bu sorun için bir destek bileti aç.",
]

sonuclar = []
for i, soru in enumerate(konusma_senaryosu, start=1):
    print(f"\n[TUR {i}] Kullanıcı: {soru}")
    baslangic = time.time()
    sonuc = agent.invoke({"messages": [("user", soru)]}, config)
    sure = time.time() - baslangic
    cevap = mesaj_metnini_al(sonuc["messages"][-1])
    print(f"[TUR {i}] Asistan: {cevap}")
    print(f"[TUR {i}] Süre: {sure:.2f} sn")
    sonuclar.append({"tur": i, "soru": soru, "cevap": cevap, "sure_sn": round(sure, 2)})

# 7. Konuşma logunu CSV'ye kaydet
csv_yolu = os.path.join(OUTPUT_DIR, "konusma_log.csv")
with open(csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tur", "soru", "cevap", "sure_sn"])
    writer.writeheader()
    writer.writerows(sonuclar)
print(f"\n[SİSTEM] Konuşma logu kaydedildi: {csv_yolu}")

# 8. Açılan biletleri de ayrıca CSV'ye kaydet
bilet_csv_yolu = os.path.join(OUTPUT_DIR, "acilan_biletler.csv")
with open(bilet_csv_yolu, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["bilet_no", "kategori", "oncelik", "aciklama"])
    writer.writeheader()
    writer.writerows(BILET_DEPOSU)
print(f"[SİSTEM] Bilet kaydı kaydedildi: {bilet_csv_yolu}")

# 9. Yanıt sürelerini gösteren grafik
turlar = [r["tur"] for r in sonuclar]
sureler = [r["sure_sn"] for r in sonuclar]

plt.figure(figsize=(8, 5))
plt.bar(turlar, sureler, color="teal")
plt.title("Her Konuşma Turunun Yanıt Süresi")
plt.xlabel("Tur")
plt.ylabel("Süre (saniye)")
plt.xticks(turlar)
plt.grid(axis="y")
grafik_yolu = os.path.join(OUTPUT_DIR, "yanit_sureleri.png")
plt.savefig(grafik_yolu)
plt.close()
print(f"[SİSTEM] Yanıt süresi grafiği kaydedildi: {grafik_yolu}")

print(
    "\n(Not: Agent, LangChain'in güncel create_agent yapısıyla kuruldu — model her turda "
    "aracı çağırıp çağırmayacağına kendi karar verir. InMemorySaver ile oturum boyunca "
    "geçmiş turlar hatırlanır; bu da her turda kümülatif token maliyeti anlamına gelir.)"
)

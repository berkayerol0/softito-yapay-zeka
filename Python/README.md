# Python Programlama

Softito eğitim sürecinde geliştirdiğim Python çalışmalarını içeren repo. Temel sözdiziminden nesne yönelimli programlamaya kadar katmanlı bir ilerlemeyle ilerleyen alıştırma ve ders notlarından oluşuyor.

## İçerik

| Dosya | Konu | Seviye |
|-------|------|--------|
| `python_baslangic.py` | print, değişkenler, veri tipleri, koşullar, döngüler, listeler, fonksiyonlar | 🟢 Başlangıç |
| `python_1_ders.py` | String işlemleri, liste metodları, math kütüphanesi, sözlükler, class'a giriş, kalıtım, kapsülleme, dunder metodlar | 🟡 Temel |
| `temel_python.py` | Operatörler, f-string, list comprehension, dictionary, hata yönetimi, kapsamlı uygulama | 🟡 Temel |
| `temel_python_2.py` | Tuple, set, enumerate, zip, lambda, recursive fonksiyon, dosya işlemleri | 🟠 Orta |
| `python_class_sorular.py` | Class değişkenleri, @property, @classmethod, @staticmethod, iterator class, Mixin pattern, dunder metodlar | 🔴 İleri |

## Kapsanan Konular

**Temel Python**
- Değişkenler, veri tipleri, tip dönüşümleri
- Aritmetik, karşılaştırma ve mantıksal operatörler
- String metodları ve f-string formatlama
- Koşullu ifadeler (if / elif / else)
- Döngüler (for / while), break / continue, enumerate, zip
- Liste, tuple, set, dictionary işlemleri
- List comprehension ve dict comprehension
- Fonksiyonlar, \*args, \*\*kwargs, lambda, recursive fonksiyonlar
- Hata yönetimi (try / except / finally)
- Dosya işlemleri

**Nesne Yönelimli Programlama (OOP)**
- Class ve instance kavramı
- `__init__`, `__str__`, `__repr__`, `__add__`, `__eq__`, `__lt__`, `__len__`
- Kalıtım (inheritance) ve `super()`
- Kapsülleme ve `@property` / setter
- `@classmethod` ve `@staticmethod`
- Class değişkenleri vs instance değişkenleri
- Iterator protocol (`__iter__` / `__next__`)
- Mixin pattern ile çoklu kalıtım

## Örnek: Iterator Class

```python
class AralikIterator:
    def __init__(self, baslangic, bitis, adim):
        self.baslangic = baslangic
        self.bitis = bitis
        self.adim = adim

    def __iter__(self):
        self.mevcut = self.baslangic
        return self

    def __next__(self):
        if self.mevcut >= self.bitis:
            raise StopIteration
        deger = self.mevcut
        self.mevcut += self.adim
        return deger

list(AralikIterator(0, 10, 2))  # [0, 2, 4, 6, 8]
```

## Örnek: Mixin Pattern

```python
class KarsilastirmaMixin:
    def __eq__(self, diger): return self.maas == diger.maas
    def __lt__(self, diger): return self.maas < diger.maas

class YazdirMixin:
    def yazdir(self):
        for k, v in self.__dict__.items():
            print(f"{k}: {v}")

class Calisan(KarsilastirmaMixin, YazdirMixin):
    def __init__(self, isim, departman, maas):
        self.isim = isim
        self.departman = departman
        self.maas = maas
```

## Kurulum

```bash
git clone https://github.com/berkayerol0/softito-yapay-zeka.git
cd softito-yapay-zeka/Python
python python_baslangic.py
```

Python 3.8+ yeterli, harici kütüphane gerekmez.

## Lisans

MIT

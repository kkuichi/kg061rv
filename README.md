# Segmentácia ionosferických štruktúr na snímkach z celooblohových kamier

**Bakalárska práca**

**Autor:** Kristína Gvozdiaková  
**Rok:** 2025/2026  
**Školiace pracovisko:** Ústav umelej inteligencie, Fakulta elektrotechniky a informatiky, Technická univerzita v Košiciach  
**Študijný odbor:** Informatika  
**Študijný program:** Hospodárska informatika

Tento repozitár obsahuje zdrojový kód vytvorený v rámci bakalárskej práce. 

## Štruktúra projektu

```
├── model_EPB.ipynb        # Hlavný notebook - tréning, vyhodnotenie a nasadenie
├── requirements.txt       # Python závislosti
└── src/
    ├── model_scss_net.py  # Architektúra modelu (SCSS-Net)
    ├── data_loader.py     # Načítanie snímok a masiek
    ├── augmentation.py    # Augmentácia dát
    ├── predictor.py       # Dávková predikcia a export masiek
    ├── metrics.py         # Metriky Dice a IoU
    ├── plot_utils.py      # Vizualizácia snímok, masiek a priebehu trénovania
    └── pdf_export.py      # Generovanie PDF reportu
```

## Stiahnutie dát

Z dôvodu veľkosti súborov sú natrénované modely a datasety dostupné na stiahnutie na: <https://mega.nz/folder/uoQBBIrK#9Px3IazK7O3x1HkqYTl7eg>

Odkaz obsahuje dva priečinky:

- `models/`
    - `model_epb_v1.h5` - model natrénovaný na manuálne anotovaných dátach, veľkosť vstupu 640 × 640
    - `model_epb_final.h5` - finálny model natrénovaný na `EL_6300_EPB_2023-10--12_revised`, použitý na nasadenie
- `data/` – datasety ako ZIP archívy:
    - `manual_annotations.zip` - manuálne anotované snímky EPB použité ako počiatočná trénovacia množina
    - `EL_6300_EPB_2023-10--12_revised.zip` - vytriedené predikcie z modelu `model_epb_v1` použité na tréning finálneho modelu

Po stiahnutí a rozbalení by mala celková štruktúra projektu vyzerať nasledovne:

```
├── model_EPB.ipynb                          # Hlavný notebook
├── model_epb_v1.h5                          # Model natrénovaný na manuálne anotovaných dátach (vstup 640 × 640)
├── model_epb_final.h5                       # Finálny model použitý na nasadenie (vstup 512 × 512)
├── requirements.txt
├── src/
│   ├── model_scss_net.py
│   ├── data_loader.py
│   ├── augmentation.py
│   ├── predictor.py
│   ├── metrics.py
│   ├── plot_utils.py
│   └── pdf_export.py
└── data/
    ├── manual_annotations/                  # Manuálne anotované snímky EPB
    │   ├── imgs/
    │   └── masks/
    ├── EL_6300_EPB_2023-10--12_revised/     # Vytriedené predikcie z model_epb_v1; použité na tréning model_epb_final
    │   ├── imgs/
    │   └── masks/
```


## Popis súborov

**`model_EPB.ipynb`** - Hlavný notebook. Pokrýva celý postup: načítanie dát, tréning, vyhodnotenie na validačnej množine, predikciu na testovacej množine/nasadenie na dátach bez ground-truth masiek a export výsledkov do PDF.

---

**`requirements.txt`** - Zoznam závislostí potrebných na spustenie projektu.

---

**`src/model_scss_net.py`** - Definuje model `scss_net`. Prevzaté z SCSS-Net (pozri citáciu nižšie).

---

**`src/data_loader.py`** - Obsahuje funkcie `load_images()` a `load_masks()` na načítanie obrázkov PNG a masiek z adresára, zmenu ich veľkosti a normalizáciu hodnôt pixelov na `[0, 1]`. Funkcia `load_masks()` vráti `None` ak adresár s maskami neexistuje.

---

**`src/augmentation.py`** - Poskytuje triedu `DataGenerator` a funkciu `create_augmentations()`. Geometrické augmentácie (prevrátenia, rotácia) sa aplikujú na obrázky aj masky; farebné augmentácie (jas, kontrast, gama) sa aplikujú iba na obrázky.

---

**`src/predictor.py`** - Spúšťa predikciu v dávkach a ukladá predikované masky ako PNG súbory do zadaného adresára.

---

**`src/metrics.py`** - Implementuje metriky Dice a IoU. Funkcia `compute_metrics()` vypíše priemerné výsledky s prahovaním aj bez neho.

---

**`src/plot_utils.py`** - Funkcia `plot_imgs()` zobrazí snímky, ground-truth masky, predikcie a overlay vedľa seba; pri dostupnosti masiek aj predikcií zobrazí hodnoty Dice/IoU pre každú snímku. Funkcia `plot_metrics()` vykreslí vývoj metrík Dice, IoU a stratovej funkcie počas trénovania.

---

**`src/pdf_export.py`** - Generuje viacstránkové PDF s výsledkami predikcií; pri dostupnosti ground-truth masiek vytvorí 4 stĺpce (Image | Mask | Prediction | Overlay), bez masiek 3 stĺpce (Image | Prediction | Overlay).





## Citácia

Architektúra modelu je prevzatá z:

>Šimon Mackovjak, Martin Harman, Viera Maslej-Krešňáková, Peter Butka, *SCSS-Net: solar corona structures segmentation by deep learning*, Monthly Notices of the Royal Astronomical Society, Volume 508, Issue 3, December 2021, Pages 3111–3124, [https://doi.org/10.1093/mnras/stab2536](https://doi.org/10.1093/mnras/stab2536)

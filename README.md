# Lane Detection — AUROBI / Teknofest 2025 Robotaksi

> **YOLOPv2 + CLRNet** tabanli gercek zamanli serit takip sistemi  
> Teknofest 2025 Robotaksi Yarismasi — Hazir Arac Kategorisi  
> Hedef arac: **Bee1** (Beemobs) | Kamera: **ZED2 Stereo** | IMU: **XSENS MTI-680**

---

## Mimari

```
Kamera Goruntüsü
      |
      +---> YOLOPv2 -------> Drivable Area Maskesi (yesil overlay)
      |
      +---> CLRNet  -------> Serit Cizgileri + Polinom Fit
                    |
                    v
            Lane Metrics Pipeline
                    |
                    v
         17 Metrik (piksel + metre)  +  CSV  +  Gorsel
```

---

## Metrikler

| # | Metrik | Birim |
|---|--------|-------|
| 1 | Serit sapma mesafesi | m / px |
| 2 | Aracin serit merkezine yatay uzakligi | m / px |
| 3 | Yaw acisi (arac - serit yonu) | derece |
| 4 | Sol serit egriligi | m |
| 5 | Sag serit egriligi | m |
| 6 | Egrilik degisimi | m |
| 7 | Serit genisligi | m / px |
| 8 | Sol on teker - sol serit uzakligi | m |
| 9 | Sag on teker - sag serit uzakligi | m |
| 10 | Sol serit tipi | surekli / kesikli |
| 11 | Sag serit tipi | surekli / kesikli |
| 12 | Timestamp | s |
| 13 | Aracin bulundugu serit | sol / merkez / sag |
| 14 | Komsu sol serit varligi | var / yok |
| 15 | Komsu sag serit varligi | var / yok |
| 16 | Algilama mesafesi | m |
| 17 | Guvenilirlik skoru | yuksek / dusuk |

---

## Dosya Yapisi

```
lane-detection/
|
+-- lane_metrics.py          # Video metrik pipeline (YOLOPv2)
+-- lane_metrics_clrnet.py   # Video metrik pipeline (CLRNet + YOLOPv2)
+-- webcam_lane.py           # Gercek zamanli webcam / ZED2 akisi
+-- demo.py                  # Temel YOLOPv2 inference demo
|
+-- utils/
|   +-- utils.py             # YOLOPv2 yardimci fonksiyonlar
|   +-- clrnet_infer.py      # CLRNet ONNX inference wrapper
|
+-- configs/                 # CLRNet yapilandirma dosyalari
+-- data/
|   +-- weights/             # Model agirliklari (git'e dahil degil)
|   |   +-- yolopv2.pt       # YOLOPv2 pretrained
|   |   +-- culane_r18.pth   # CLRNet pretrained
|   +-- demo/                # Test videolari (git'e dahil degil)
|
+-- requirements.txt
+-- README.md
```

---

## Kurulum

### 1. Repo'yu klonla

```bash
git clone https://github.com/AUROBI/lane-detection.git
cd lane-detection
```

### 2. Sanal ortam ve bagimliliklar

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python numpy onnxruntime-gpu scipy tqdm pyyaml
```

### 3. Model agirliklarini indir

**YOLOPv2:**
```
https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
-> data/weights/yolopv2.pt
```

**CLRNet:**
```
PINTO model zoo uzerinden culane_r18 ONNX modeli
-> data/weights/culane_r18.pth
```

---

## Kullanim

### Video uzerinde test (YOLOPv2 + CLRNet)

```bash
python lane_metrics_clrnet.py
```

Cikti: `runs/metrics_clrnet/output_clrnet.mp4` + `metrics_clrnet.csv`

### Gercek zamanli webcam / ZED2

```bash
python webcam_lane.py
```

| Tus | Islem |
|-----|-------|
| Q | Cikis |
| S | Ekran goruntüsü al |
| R | Smoother sifirla |

### Temel YOLOPv2 demo

```bash
python demo.py --source data/demo/test.mp4 --device 0
```

---

## Arac Teknik Ozellikleri (Bee1)

| Parametre | Deger |
|-----------|-------|
| On iz genisligi | 886 mm |
| Arka iz genisligi | 850 mm |
| Dingil mesafesi | 1860 mm |
| Max hiz | 30 km/s (limitli) |
| Kamera | ZED2 Stereo (685mm yukseklik) |
| IMU | XSENS MTI-680-DK |
| LIDAR | Velodyne VLP-16 |
| GPU | NVIDIA RTX 3060 |
| OS | Ubuntu 20.04 + ROS |

---

## Gereksinimler

```
torch >= 2.0
torchvision
opencv-python
numpy
onnxruntime-gpu
scipy
tqdm
pyyaml
```

---

## Takim

**AUROBI** — Ankara Universitesi  
Teknofest 2025 Robotaksi Yarismasi — Hazir Arac Kategorisi

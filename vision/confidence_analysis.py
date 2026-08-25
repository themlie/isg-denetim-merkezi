import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo_ppe_best.pt"
    model = YOLO(MODEL_PATH)

    test_klasor = Path(__file__).parent.parent / "datasets" / "ppe_dataset" / "test" / "images"
    gorseller = list(test_klasor.glob("*.jpg"))[:150]

    skorlar = {}

    print(f"{len(gorseller)} görsel analiz ediliyor, lütfen bekleyin...")

    for gorsel_yolu in gorseller:
        frame = cv2.imread(str(gorsel_yolu))
        if frame is None:
            continue
            
        results = model(frame, conf=0.05, verbose=False)
        sonuc = results[0]

        for c, cf in zip(sonuc.boxes.cls, sonuc.boxes.conf):
            isim = model.names[int(c)]
            skorlar.setdefault(isim, []).append(float(cf))

    print("\n" + "="*92)
    print(f"{'Sınıf':<12} | {'Adet':<6} | {'Ortalama':<8} | {'Medyan':<8} | {'Max':<8} | {'>0.25 Adet':<10} | {'>0.40 Adet':<10} | {'>0.50 Adet':<10}")
    print("-" * 92)

    siniflar = ["helmet", "vest", "no-helmet", "no-vest"]
    
    for sinif in siniflar:
        if sinif in skorlar:
            dizi = np.array(skorlar[sinif])
            adet = len(dizi)
            ortalama = dizi.mean()
            medyan = np.median(dizi)
            max_deger = dizi.max()
            ust_adet_25 = (dizi > 0.25).sum()
            ust_adet_40 = (dizi > 0.40).sum()
            ust_adet_50 = (dizi > 0.50).sum()
        else:
            adet = 0
            ortalama = 0.0
            medyan = 0.0
            max_deger = 0.0
            ust_adet_25 = 0
            ust_adet_40 = 0
            ust_adet_50 = 0
            
        print(f"{sinif:<12} | {adet:6d} | {ortalama:8.2f} | {medyan:8.2f} | {max_deger:8.2f} | {ust_adet_25:10d} | {ust_adet_40:10d} | {ust_adet_50:10d}")
    print("="*92)

if __name__ == "__main__":
    main()

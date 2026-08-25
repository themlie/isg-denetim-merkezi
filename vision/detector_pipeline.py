import cv2
from pathlib import Path
from ultralytics import YOLO

import sys
from pathlib import Path

# ana dizindeki mainden cagiriyoz
sys.path.append(str(Path(__file__).parent.parent))
from main import run_audit

# modeli yukluyoruz
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo_ppe_best.pt"
model = YOLO(MODEL_PATH)

def detect_violations(frame, conf=0.25):
    # algilama yapiyo
    results = model(frame, conf=conf, verbose=False)
    
    # ilk resmi al
    sonuc = results[0]
    
    no_helmet_count = 0
    no_vest_count = 0
    
    # dongu ile sayimlari yapiyoruz
    for c in sonuc.boxes.cls:
        isim = model.names[int(c)]
        if isim == "no-helmet":
            no_helmet_count += 1
        elif isim == "no-vest":
            no_vest_count += 1
            
    # kutulari ciziyor
    plotted_frame = sonuc.plot()
    
    return plotted_frame, no_helmet_count, no_vest_count

def process_and_log(frame, facility_name, conf=0.25):
    # once tespit isini hallediyoz
    plotted_frame, baretsiz, yeleksiz = detect_violations(frame, conf)
    
    # ikisini topluyoz
    toplam = baretsiz + yeleksiz
    
    log_id = None # bastan none atiyoz yoksa hata verir
    
    # eger hata varsa
    if toplam > 0:
        # Bu bilgi kameradan gelmiyor, tesisin geçmiş kaza kaydına ait sabit bir bağlam değeri.
        gecmis_kaza = 2
        risk_bolgesi = 1
        
        # log isini db ye atiyoz ve donen id'yi aliyoz
        log_id, risk_level = run_audit(facility_name, baretsiz, yeleksiz, gecmis_kaza, risk_bolgesi)
        
    return plotted_frame, baretsiz, yeleksiz, log_id

if __name__ == "__main__":
    # test resmini seciyoruz
    test_klasor = Path(__file__).parent.parent / "datasets" / "ppe_dataset" / "test" / "images"
    gorseller = list(test_klasor.glob("*.jpg"))
    
    # resmi okuma kismi
    frame = cv2.imread(str(gorseller[0]))
    
    # yeni fonksiyonu test ediyoz
    tesis = "Test Tesisi"
    plotted_frame, no_helmet, no_vest, log_id = process_and_log(frame, tesis)
    
    # ekrana yazdirma
    print(f"Baretsiz sayısı: {no_helmet}")
    print(f"Yeleksiz sayısı: {no_vest}")
    
    # resmi goster
    cv2.imshow("Test Ekrani", plotted_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

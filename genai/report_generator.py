import os
import json
from datetime import datetime
from pathlib import Path
import sys

# Proje ana dizinini path'e ekle (eğer ana dizinden çalıştırılmazsa modül bulma hatasını önlemek için)
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DBManager
from genai.llm_client import ask_llm

REPORTS_DIR = Path(__file__).parent.parent / "reports"

def generate_daily_report():
    # 1. Veritabanından logları çek
    db = DBManager()
    logs = db.fetch_logs(limit=50) # Son 50 denetim
    
    if not logs:
        print("Veritabanında analiz edilecek log bulunamadı.")
        return
        
    # 2. Verileri aggrege et (Özetle)
    toplam_baret_ihlali = 0
    toplam_yelek_ihlali = 0
    toplam_risk_skoru = 0.0
    
    for row in logs:
        # DB Tablosu: id, timestamp, facility_name, no_helmet_count, no_vest_count, hazard_score, risk_level
        toplam_baret_ihlali += row[3]
        toplam_yelek_ihlali += row[4]
        toplam_risk_skoru += row[5]
        
    ortalama_risk = toplam_risk_skoru / len(logs) if logs else 0
    risk_durumu = "YÜKSEK" if ortalama_risk >= 10 else ("ORTA" if ortalama_risk >= 5 else "DÜŞÜK")
    
    ozet_veri = {
        "Analiz Edilen Denetim Sayısı": len(logs),
        "Toplam Baret İhlali (No Helmet)": toplam_baret_ihlali,
        "Toplam Yelek İhlali (No Vest)": toplam_yelek_ihlali,
        "Ortalama Risk Skoru": round(ortalama_risk, 2),
        "Genel Tesis Risk Durumu": risk_durumu
    }
    
    print("Veritabanından çekilen özet veri:")
    print(json.dumps(ozet_veri, indent=2, ensure_ascii=False))
    print("-" * 50)
    
    # 3. LLM Promptunu Hazırla
    prompt = f"""Aşağıdaki tesis ihlal verilerini inceleyerek fabrika yönetimi için resmi bir İSG Denetim Özeti ve 2 maddelik DÖF (Düzeltici Önleyici Faaliyet) Aksiyon Planı hazırla.
    
    VERİLER:
    {json.dumps(ozet_veri, indent=2, ensure_ascii=False)}

    GÖREV:
    - 1. Paragraf: Genel tesis risk durumunu ve ihlal istatistiklerini (baret/yelek sayıları) kurumsal bir dille özetle.
    - 2. Paragraf: Risk skorunun İSG standartları açısından değerlendirmesini yap.
    - Aksiyon Planı: Yalnızca 2 kısa, net ve uygulanabilir aksiyon maddesi yaz.
    - Cümleleri tam olarak bitir.
    """
    
    print("LLM raporu üretiyor, lütfen bekleyin...")
    rapor_metni = ask_llm(prompt, max_tokens=600)


    
    # 4. Raporu diske kaydet
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tarih_str = datetime.now().strftime("%Y_%m_%d")
    dosya_adi = f"audit_report_{tarih_str}.md"
    dosya_yolu = REPORTS_DIR / dosya_adi
    
    with open(dosya_yolu, "w", encoding="utf-8") as f:
        f.write("# İSG Günlük Denetim Raporu\n\n")
        f.write(rapor_metni)
        
    print(f"\nBaşarılı! Rapor oluşturuldu ve kaydedildi: {dosya_yolu}")
    
if __name__ == "__main__":
    generate_daily_report()

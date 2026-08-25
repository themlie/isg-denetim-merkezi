from core.risk_engine import predict_risk
from database.db_manager import DBManager

HELMET_WEIGHT = 2.5
VEST_WEIGHT = 1.5
INCIDENT_WEIGHT = 3.0
ZONE_WEIGHT = 2.0

db = DBManager()

def run_audit(facility_name, no_helmet_count, no_vest_count, past_incidents, high_risk_zone):
    """
    Kameradan gelen veya elle girilen ihlal sayılarını alır,
    risk seviyesini tahmin eder ve veritabanına kaydeder.
    """
    
   
    risk_level = predict_risk(no_helmet_count, no_vest_count, past_incidents, high_risk_zone)
    

    hazard_score = (no_helmet_count * HELMET_WEIGHT) + \
                   (no_vest_count * VEST_WEIGHT) + \
                   (past_incidents * INCIDENT_WEIGHT) + \
                   (high_risk_zone * ZONE_WEIGHT)
                   
    
    log_id = db.add_log(facility_name, no_helmet_count, no_vest_count, hazard_score, risk_level)
    
    
    print(f"Log ID: {log_id} - Risk Level: {risk_level}")
    
   
    return log_id, risk_level


if __name__ == "__main__":
    print("Entegrasyon Testi")
    
   
    print("Test 1 (Yüksek Riskli Durum):")
    run_audit(
        facility_name="Tesis A", 
        no_helmet_count=3, 
        no_vest_count=2, 
        past_incidents=0, 
        high_risk_zone=1
    )
    
    
   
    print("Test 2 (Düşük Riskli Durum):")
    run_audit(
        facility_name="Tesis B", 
        no_helmet_count=0, 
        no_vest_count=0, 
        past_incidents=0, 
        high_risk_zone=0
    )
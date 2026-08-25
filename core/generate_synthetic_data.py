import numpy as np
import pandas as pd
import collections

from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "data" / "synthetic_audit_logs_v2.csv"

N =500

np.random.seed(42)

no_helmet_count = np.random.randint (0, 6, N)

no_vest_count = np.random.randint (0, 6, N)

past_incidents = np.random.randint (0, 4, N)

high_risk_zone = np.random.randint (0, 2, N)

risk_score = ((no_helmet_count * 2.5)
+ (no_vest_count * 1.5)
+ (past_incidents * 3.0) 
+ (high_risk_zone * 2.0)
+ np.random.normal(0, 0.5, N) )

alt, ust = np.percentile(risk_score, [33.33, 66.67])
print(f"Alt ve ust yuzdelik esikleri: {alt:.4f}, {ust:.4f}")



print("No Helmet Count: ", no_helmet_count[:10])

print("No Vest Count: ", no_vest_count[:10])

print("Past Incident: ", past_incidents[:10])

print("High Risk Zone ", high_risk_zone[:10]) 

print("Risk Score: ", risk_score[:10])

print("Min: ", risk_score.min(), " Max: ", risk_score.max())

risk_level=[]

for score in risk_score:
    if score >= ust:
        risk_level.append("High")
    elif score >= alt:
        risk_level.append("Medium")
    else:
        risk_level.append("Low")


print("Risk Level: ", risk_level[:10])
print(collections.Counter(risk_level))

df = pd.DataFrame({
    "no_helmet_count": no_helmet_count,
    "no_vest_count": no_vest_count,
    "past_incidents": past_incidents,
    "high_risk_zone": high_risk_zone,
    "risk_level": risk_level,
})

print(df.head())
print(df.shape)

df.to_csv(CSV_PATH, index=False)
print(f"{len(df)} satir csvde: {CSV_PATH}")

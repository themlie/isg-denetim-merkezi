import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split





CSV_PATH = Path(__file__).parent.parent / "data" / "synthetic_audit_logs_v2.csv"
def preprocess_data():
    df = pd.read_csv(CSV_PATH)

    X = df[["no_helmet_count", "no_vest_count", "past_incidents", "high_risk_zone"]]
    y = df["risk_level"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test,scaler,encoder

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)



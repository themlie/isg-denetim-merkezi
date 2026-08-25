import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from data_preprocessor import preprocess_data

MODEL_DIR = Path(__file__).parent.parent / "models"

X_train, X_test, y_train, y_test, scaler, encoder = preprocess_data()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Doğruluk Oranı (Accuracy):")
print(accuracy_score(y_test, y_pred))

print("\nKarmaşıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

joblib.dump(model, MODEL_DIR / "risk_classifier_v2.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler_v2.pkl")
joblib.dump(encoder, MODEL_DIR / "label_encoder_v2.pkl")

print("\nTüm nesneler kaydedildi!")

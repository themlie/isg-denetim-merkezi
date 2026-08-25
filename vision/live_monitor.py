import cv2
import time
from vision.stream_handler import resize_frame
from vision.detector_pipeline import detect_violations
from main import run_audit

DETECT_EVERY = 30
LOG_COOLDOWN = 10

def run_live(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Kamera acilamadi.")
        return

    frame_counter = 0
    last_unhelmeted = 0
    last_unvested = 0
    last_risk_level = "Bilinmiyor"
    last_log_time = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        frame = resize_frame(frame)

        if frame_counter % DETECT_EVERY == 0:
            plotted_frame, no_helmet, no_vest = detect_violations(frame)
            last_unhelmeted = no_helmet
            last_unvested = no_vest

        if (last_unhelmeted > 0 or last_unvested > 0):
            current_time = time.time()
            if current_time - last_log_time > LOG_COOLDOWN:
                log_id, risk_level = run_audit("Gerçek Zamanlı Tesis", last_unhelmeted, last_unvested, 2, 1)
                last_risk_level = risk_level
                last_log_time = current_time

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        if last_risk_level == "High":
            color = (0, 0, 255)
        elif last_risk_level == "Medium":
            color = (0, 255, 255)
        elif last_risk_level == "Low":
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Baretsiz: {last_unhelmeted}  Yeleksiz: {last_unvested}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Tesis Risk Durumu: {last_risk_level}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Gercek Zamanli Ihlal Tespit Sistemi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()

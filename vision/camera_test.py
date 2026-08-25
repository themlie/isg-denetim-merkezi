import cv2


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

ret, frame = cap.read()

if ret:
    print(f"Kare boyutu: {frame.shape}")
else:
    print("Kameradan kare okunamadi.")

cap.release()
cv2.destroyAllWindows()

import cv2
import time


FRAME_SIZE = 640


def resize_frame(frame, size=FRAME_SIZE):
    return cv2.resize(frame, (size, size))


def run_stream(source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Kamera acilamadi.")
        cap.release()
        cv2.destroyAllWindows()
        return

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            break

        frame = resize_frame(frame)
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Kamera Akisi", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_stream()

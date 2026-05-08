import cv2
import numpy as np

# -----------------------------
# KNOWN VALUES (YOU SET THESE)
# -----------------------------
REAL_WIDTH_M = 0.05   # 5 cm square = 0.05 m
DISTANCE_M = 0.30     # <-- CHANGE THIS to your actual camera-to-square distance

# -----------------------------
# Camera setup
# -----------------------------
IMG_W = 640
IMG_H = 480

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

print("Press 'c' to capture for calibration")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold to detect dark square
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_box = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # look for roughly square shape
        if 0.7 < aspect_ratio < 1.3:
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)

    if best_box is not None:
        x, y, w, h = best_box

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"width_px = {w}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "No square detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Calibration", frame)
    cv2.imshow("Threshold", thresh)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and best_box is not None:
        _, _, w, _ = best_box

        # Compute focal length
        fx = (w * DISTANCE_M) / REAL_WIDTH_M

        print("\n=== CALIBRATION RESULT ===")
        print(f"Measured width (pixels): {w}")
        print(f"Distance (m): {DISTANCE_M}")
        print(f"Real width (m): {REAL_WIDTH_M}")
        print(f"FOCAL LENGTH FX = {fx:.2f} pixels\n")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import os
from ultralytics import YOLO
import time
# -------------------------
# Load model once (GLOBAL)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "violence_detect_yolo8s.pt")

model = YOLO(MODEL_PATH)

# -------------------------
# CONFIG (tune these)
# -------------------------
CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 500
CONSEC_FRAMES_THRESHOLD = 8
DURATION_THRESHOLD_SEC = 1.5
HIT_RATIO_THRESHOLD = 0.15

# CONF_THRESHOLD = 0.65
# MIN_BOX_AREA = 800
# CONSEC_FRAMES_THRESHOLD = 6
# DURATION_THRESHOLD_SEC = 0.8
# HIT_RATIO_THRESHOLD = 0.12

# -------------------------
# IMAGE DETECTION
# -------------------------
def detect_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return False

    results = model(image, conf=CONF_THRESHOLD)

    if results[0].boxes is None:
        return False

    for det in results[0].boxes:
        cls = int(det.cls)
        conf = float(det.conf)

        if cls == 1 and conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area >= MIN_BOX_AREA:
                return True

    return False


# -------------------------
# VIDEO DETECTION
# -------------------------
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    consecutive = 0
    max_consecutive = 0
    total_hits = 0
    total_frames = 0
    confidence_sum = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        detected_in_frame = False

        results = model(frame, conf=CONF_THRESHOLD)

        if results[0].boxes is not None:
            for det in results[0].boxes:
                cls = int(det.cls)
                conf = float(det.conf)

                if cls == 1 and conf >= CONF_THRESHOLD:
                    x1, y1, x2, y2 = map(int, det.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    if area < MIN_BOX_AREA:
                        continue

                    detected_in_frame = True
                    total_hits += 1
                    confidence_sum += conf

        if detected_in_frame:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    cap.release()

    # -------------------------
    # DECISION LOGIC
    # -------------------------
    if total_hits == 0:
        return False

    duration_sec = max_consecutive / fps
    hit_ratio = total_hits / total_frames
    avg_conf = confidence_sum / total_hits

    # Production-safe decision
    if duration_sec >= DURATION_THRESHOLD_SEC:
        print(f"Violence detected based on duration: {duration_sec:.2f} sec")
        return True

    if max_consecutive >= CONSEC_FRAMES_THRESHOLD:
        print(f"Violence detected based on consecutive frames: {max_consecutive} frames")
        return True

    if hit_ratio >= HIT_RATIO_THRESHOLD and avg_conf >= 0.6:
        print(f"Violence detected based on hit ratio: {hit_ratio:.2%} with avg confidence {avg_conf:.2f}")
        return True

    return False


# -------------------------
# UNIVERSAL ENTRY POINT
# -------------------------
def is_violence_detected(file_path):
    if not os.path.exists(file_path):
        return False

    ext = file_path.lower().split(".")[-1]

    image_exts = {"jpg", "jpeg", "png", "bmp", "webp"}
    video_exts = {"mp4", "avi", "mov", "mkv", "webm"}

    if ext in image_exts:
        return detect_image(file_path)

    elif ext in video_exts:
        return detect_video(file_path)

    else:
        # Unknown format
        print(f"Unsupported file format: {ext}")
        return False
    


# if __name__ == "__main__":

#     start_time = time.time()
#     # Example usage
#     # test_file = "videos/v4_test.mp4"  # Change to your test file
#     test_file = "images/test9.jpg"  # Change to your test file
#     result = is_violence_detected(test_file)
#     print(f"Violence detected: {result}")    
#     end_time = time.time()
#     print(f"Processing time: {end_time - start_time:.2f} seconds")


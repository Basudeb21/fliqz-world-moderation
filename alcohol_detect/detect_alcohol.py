import cv2
import os
from ultralytics import YOLO

# -------------------------
# LOAD MODEL (GLOBAL - DO NOT RELOAD)
# -------------------------
model = YOLO(os.path.join(os.path.dirname(__file__), "alcohol_detect_yolov8s.pt"))

# -------------------------
# CONFIG (TUNE BASED ON DATA)
# -------------------------
IMG_CONF_THRESHOLD = 0.8
VID_CONF_THRESHOLD = 0.75
MIN_BOX_AREA = 800
ALCOHOL_CLASS_ID = 0

# Video optimization
FRAME_SKIP = 2        # process every 2nd frame (speed ↑)
MIN_HITS = 5         # minimum detections to confirm alcohol


# -------------------------
# IMAGE DETECTION
# -------------------------
def detect_image(image_path: str) -> bool:
    image = cv2.imread(image_path)

    if image is None:
        return False

    results = model(image, conf=IMG_CONF_THRESHOLD, verbose=False)

    if not results or results[0].boxes is None:
        return False

    for det in results[0].boxes:
        cls = int(det.cls)
        conf = float(det.conf)

        if cls != ALCOHOL_CLASS_ID or conf < IMG_CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area >= MIN_BOX_AREA:
            return True

    return False


# -------------------------
# VIDEO DETECTION (FAST + STABLE)
# -------------------------
def detect_video(video_path: str) -> bool:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    hit_count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_id += 1

        # Skip frames for speed
        if frame_id % FRAME_SKIP != 0:
            continue

        # Resize for faster inference (huge speed gain)
        frame = cv2.resize(frame, (640, 640))

        results = model(frame, conf=VID_CONF_THRESHOLD, verbose=False)

        if results and results[0].boxes is not None:
            for det in results[0].boxes:
                cls = int(det.cls)
                conf = float(det.conf)

                if cls != ALCOHOL_CLASS_ID or conf < VID_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, det.xyxy[0])
                area = (x2 - x1) * (y2 - y1)

                if area >= MIN_BOX_AREA:
                    hit_count += 1
                    break  # only count once per frame

        # Early exit → faster response
        if hit_count >= MIN_HITS:
            cap.release()
            return True

    cap.release()
    return False


# -------------------------
# UNIVERSAL ENTRY POINT
# -------------------------
def is_alcohol_detected(file_path: str) -> bool:
    if not os.path.exists(file_path):
        return False

    ext = file_path.split(".")[-1].lower()

    image_exts = {"jpg", "jpeg", "png", "bmp", "webp"}
    video_exts = {"mp4", "avi", "mov", "mkv", "webm"}

    if ext in image_exts:
        return detect_image(file_path)

    elif ext in video_exts:
        return detect_video(file_path)

    return False
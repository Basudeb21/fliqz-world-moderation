import cv2
import os
import torch
from ultralytics import YOLO

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8s_weapon_detect.pt")

# fallback: try repo-relative path if file not found
if not os.path.exists(MODEL_PATH):
    alt = os.path.join(os.getcwd(), "weapon_detect", "models", "yolov8s_weapon_detect.pt")
    if os.path.exists(alt):
        MODEL_PATH = alt

CONF_THRESHOLD = 0.7

IMG_SIZE = 640

MIN_BOX_AREA = 500

MIN_HITS = 5

# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Failed loading YOLO model at {MODEL_PATH}: {e}")
    raise

# choose inference device for predict() calls
INFER_DEVICE = 0 if torch.cuda.is_available() else 'cpu'

# =========================================================
# VALID EXTENSIONS
# =========================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)

# =========================================================
# IMAGE DETECTION
# =========================================================

def detect_weapon_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    results = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        device=INFER_DEVICE,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        return False

    for box in result.boxes:

        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        area = (
            (x2 - x1)
            * (y2 - y1)
        )

        if area < MIN_BOX_AREA:
            continue

        print(
            f"Weapon detected in image "
            f"with confidence {conf:.2f}"
        )

        return True

    return False

# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_weapon_video(
    video_path,
    frame_skip=2
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return {
            "detected": False
        }

    frame_count = 0

    hit_count = 0

    detection_start_time = None

    detection_end_time = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # =================================================
        # CURRENT TIMESTAMP
        # =================================================

        timestamp_sec = (
            cap.get(cv2.CAP_PROP_POS_MSEC)
            / 1000
        )

        # =================================================
        # FRAME SKIP
        # =================================================

        if frame_count % frame_skip != 0:
            continue

        # =================================================
        # RESIZE
        # =================================================

        frame = cv2.resize(
            frame,
            (640, 640)
        )

        # =================================================
        # INFERENCE
        # =================================================

        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            device=INFER_DEVICE,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            continue

        for box in result.boxes:

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            area = (
                (x2 - x1)
                * (y2 - y1)
            )

            if area < MIN_BOX_AREA:
                continue

            hit_count += 1

            # =============================================
            # DURATION TRACKING
            # =============================================

            if detection_start_time is None:
                detection_start_time = timestamp_sec

            detection_end_time = timestamp_sec

            break

        # =================================================
        # EARLY EXIT
        # =================================================

        if hit_count >= MIN_HITS:

            duration = (
                detection_end_time
                - detection_start_time
            )

            cap.release()

            return {
                "detected": True,
                "start_time": round(
                    detection_start_time,
                    2
                ),
                "end_time": round(
                    detection_end_time,
                    2
                ),
                "duration": round(
                    duration,
                    2
                ),
                "hits": hit_count
            }

    cap.release()

    return {
        "detected": False
    }

# =========================================================
# UNIVERSAL API
# =========================================================

def is_weapon_detected(
    file_path,
    frame_skip=2
):

    if not os.path.exists(file_path):

        return False

    ext = os.path.splitext(
        file_path
    )[1].lower()

    # =====================================================
    # IMAGE
    # =====================================================

    if ext in IMAGE_EXTENSIONS:

        return detect_weapon_image(
            file_path
        )

    # =====================================================
    # VIDEO
    # =====================================================

    elif ext in VIDEO_EXTENSIONS:

        return detect_weapon_video(
            video_path=file_path,
            frame_skip=frame_skip
        )

    # =====================================================
    # INVALID
    # =====================================================

    return False

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    result = is_weapon_detected(
        "test/videos/test.mp4",
        frame_skip=2
    )

    print(result)
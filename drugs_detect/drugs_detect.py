import os
import cv2
from collections import deque
from ultralytics import YOLO
import torch

# =========================================================
# MODEL PATHS
# =========================================================

# Resolve model files relative to this module so imports from repo root work
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

SYRINGE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8s_syringe_detect.pt")

PILL_IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8s_image_pill_detection.pt")
PILL_VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8s_video_pill_detection.pt")

# Device selection: use GPU if available, otherwise CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD MODELS
# =========================================================


# Instantiate models
syringe_model = YOLO(SYRINGE_MODEL_PATH)

pill_image_model = YOLO(PILL_IMAGE_MODEL_PATH)
pill_video_model = YOLO(PILL_VIDEO_MODEL_PATH)

# Ensure models use the selected device (overrides any saved args)
for _m in (syringe_model, pill_image_model, pill_video_model):
    try:
        _m.args.device = DEVICE
    except Exception:
        pass

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
# COMMON CONFIG
# =========================================================

IMG_SIZE = 960

MIN_BOX_AREA = 400

# =========================================================
# SYRINGE CONFIG
# =========================================================

SYRINGE_IMAGE_CONF = 0.60

SYRINGE_VIDEO_CONF = 0.30

SYRINGE_HIGH_CONF = 0.70

SYRINGE_NORMAL_MIN_FRAMES = 5

SYRINGE_HIGH_MIN_FRAMES = 3

# =========================================================
# PILL CONFIG
# =========================================================

PILL_IMAGE_CONF = 0.74

PILL_VIDEO_CONF = 0.70

PILL_HIGH_CONF = 0.85

PILL_CONSEC_FRAMES = 5

PILL_HIGH_CONF_FRAMES = 2

PILL_WINDOW_SIZE = 30

PILL_HIT_RATIO = 0.20

# =========================================================
# SYRINGE IMAGE DETECTION
# =========================================================

def detect_syringe_image(image_path):

    results = syringe_model.predict(
        source=image_path,
        conf=SYRINGE_IMAGE_CONF,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    for box in result.boxes:

        cls_id = int(box.cls[0])

        class_name = syringe_model.names[cls_id]

        if class_name == "syringe":
            return True

    return False

# =========================================================
# SYRINGE VIDEO DETECTION
# =========================================================

def detect_syringe_video(
    video_path,
    frame_skip=15
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    frame_count = 0

    normal_hits = 0

    high_hits = 0

    detection_start_time = None

    detection_end_time = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # =================================================
        # FRAME SKIP
        # =================================================

        if frame_skip > 1:

            if frame_count % frame_skip != 0:
                continue

        # =================================================
        # INFERENCE
        # =================================================

        results = syringe_model.predict(
            source=frame,
            conf=SYRINGE_VIDEO_CONF,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        frame_hit = False

        high_hit = False

        timestamp_sec = (
            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        )

        for box in result.boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            class_name = syringe_model.names[cls_id]

            if class_name != "syringe":
                continue

            frame_hit = True

            # =============================================
            # DURATION TRACKING
            # =============================================

            if detection_start_time is None:
                detection_start_time = timestamp_sec

            detection_end_time = timestamp_sec

            if conf >= SYRINGE_HIGH_CONF:
                high_hit = True

        # =================================================
        # TEMPORAL LOGIC
        # =================================================

        if frame_hit:
            normal_hits += 1
        else:
            normal_hits = 0

            detection_start_time = None
            detection_end_time = None

        if high_hit:
            high_hits += 1
        else:
            high_hits = 0

        # =================================================
        # FINAL DECISION
        # =================================================

        if high_hits >= SYRINGE_HIGH_MIN_FRAMES:

            duration = (
                detection_end_time
                - detection_start_time
            )

            print(
                f"Syringe detected from "
                f"{detection_start_time:.2f}s "
                f"to {detection_end_time:.2f}s "
                f"(duration: {duration:.2f}s) "
                f"with high confidence"
            )

            cap.release()
            return {
                "detected": True,
                "type": "syringe",
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
                )
            }

        if normal_hits >= SYRINGE_NORMAL_MIN_FRAMES:

            duration = (
                detection_end_time
                - detection_start_time
            )

            print(
                f"Syringe detected from "
                f"{detection_start_time:.2f}s "
                f"to {detection_end_time:.2f}s "
                f"(duration: {duration:.2f}s) "
                f"with normal confidence"
            )

            cap.release()
            return {
                "detected": True,
                "type": "syringe",
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
                )
            }

    cap.release()

    return {
    "detected": False
}

# =========================================================
# PILL IMAGE DETECTION
# =========================================================

def detect_pill_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    results = pill_image_model.predict(
        source=image,
        conf=PILL_IMAGE_CONF,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    result = results[0]

    for box in result.boxes:

        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        area = (x2 - x1) * (y2 - y1)

        if area < MIN_BOX_AREA:
            continue

        if conf >= PILL_IMAGE_CONF:
            return True

    return False

# =========================================================
# PILL VIDEO DETECTION
# =========================================================

def detect_pill_video(
    video_path,
    frame_skip=15
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    frame_index = 0

    consecutive_hits = 0

    high_conf_hits = 0

    window_hits = deque(maxlen=PILL_WINDOW_SIZE)

    detection_start_time = None

    detection_end_time = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        # =================================================
        # FRAME SKIP
        # =================================================

        if frame_skip > 1:

            if frame_index % frame_skip != 0:
                continue

        # =================================================
        # INFERENCE
        # =================================================

        results = pill_video_model.predict(
            source=frame,
            conf=PILL_VIDEO_CONF,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        frame_has_pill = False

        timestamp_sec = (
            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        )

        for box in result.boxes:

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            if area < MIN_BOX_AREA:
                continue

            frame_has_pill = True

            # =============================================
            # DURATION TRACKING
            # =============================================

            if detection_start_time is None:
                detection_start_time = timestamp_sec

            detection_end_time = timestamp_sec

            if conf >= PILL_HIGH_CONF:
                high_conf_hits += 1

        # =================================================
        # TEMPORAL LOGIC
        # =================================================

        if frame_has_pill:
            consecutive_hits += 1
            window_hits.append(1)
        else:
            consecutive_hits = 0
            window_hits.append(0)

            detection_start_time = None
            detection_end_time = None

        # =================================================
        # HIT RATIO
        # =================================================

        hit_ratio = sum(window_hits) / len(window_hits)

        # =================================================
        # FINAL DECISION
        # =================================================

        if high_conf_hits >= PILL_HIGH_CONF_FRAMES:

            duration = (
                detection_end_time
                - detection_start_time
            )

            print(
                f"Pill detected from "
                f"{detection_start_time:.2f}s "
                f"to {detection_end_time:.2f}s "
                f"(duration: {duration:.2f}s) "
                f"with high confidence"
            )

            cap.release()
            return {
                "detected": True,
                "type": "pill",
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
                )
            }

        if consecutive_hits >= PILL_CONSEC_FRAMES:

            duration = (
                detection_end_time
                - detection_start_time
            )

            print(
                f"Pill detected from "
                f"{detection_start_time:.2f}s "
                f"to {detection_end_time:.2f}s "
                f"(duration: {duration:.2f}s) "
                f"with normal confidence"
            )

            cap.release()
            return {
                "detected": True,
                "type": "pill",
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
                )
            }

        if (
            len(window_hits) >= PILL_WINDOW_SIZE
            and hit_ratio >= PILL_HIT_RATIO
        ):

            duration = (
                detection_end_time
                - detection_start_time
            )

            print(
                f"Pill detected from "
                f"{detection_start_time:.2f}s "
                f"to {detection_end_time:.2f}s "
                f"(duration: {duration:.2f}s) "
                f"with low confidence"
            )

            cap.release()
            return {
                "detected": True,
                "type": "pill",
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
                )
            }

    cap.release()

    return {
        "detected": False
    }

# =========================================================
# MAIN DRUG DETECTION API
# =========================================================

# def is_drug_detected(
#     file_path,
#     frame_skip=1
# ):

#     ext = os.path.splitext(file_path)[1].lower()

#     # =====================================================
#     # IMAGE
#     # =====================================================

#     if ext in IMAGE_EXTENSIONS:

#         if detect_syringe_image(file_path):
#             print("Syringe detected in image")
#             return True

#         if detect_pill_image(file_path):
#             print("Pill detected in image")
#             return True

#         return False

#     # =====================================================
#     # VIDEO
#     # =====================================================

#     elif ext in VIDEO_EXTENSIONS:

#         if detect_syringe_video(
#             video_path=file_path,
#             frame_skip=frame_skip
#         ):
#             print("Syringe detected in video")
#             return True

#         if detect_pill_video(
#             video_path=file_path,
#             frame_skip=frame_skip
#         ):
#             print("Pill detected in video")
#             return True

#         return False

#     # =====================================================
#     # INVALID FILE
#     # =====================================================

#     else:
#         raise ValueError(
#             f"Unsupported file type: {ext}"
#         )


# =========================================================
# MAIN DRUG DETECTION API
# =========================================================

def is_drug_detected(
    file_path,
    frame_skip=1
):

    ext = os.path.splitext(file_path)[1].lower()

    # =====================================================
    # IMAGE
    # =====================================================

    if ext in IMAGE_EXTENSIONS:

        if detect_syringe_image(file_path):

            print("Syringe detected in image")

            return True

        if detect_pill_image(file_path):

            print("Pill detected in image")

            return True

        return False

    # =====================================================
    # VIDEO
    # =====================================================

    elif ext in VIDEO_EXTENSIONS:

        # =============================================
        # SYRINGE VIDEO
        # =============================================

        syringe_result = detect_syringe_video(
            video_path=file_path,
            frame_skip=frame_skip
        )

        if syringe_result["detected"]:

            print("Syringe detected in video")

            return syringe_result

        # =============================================
        # PILL VIDEO
        # =============================================

        pill_result = detect_pill_video(
            video_path=file_path,
            frame_skip=frame_skip
        )

        if pill_result["detected"]:

            print("Pill detected in video")

            return pill_result

        return {
            "detected": False
        }

    # =====================================================
    # INVALID FILE
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    test_folder = "test"

    subfolders = ["images", "videos"]

    for subfolder in subfolders:

        folder_path = os.path.join(
            test_folder,
            subfolder
        )

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):

            ext = os.path.splitext(file)[1].lower()

            if (
                ext in IMAGE_EXTENSIONS
                or ext in VIDEO_EXTENSIONS
            ):

                file_path = os.path.join(
                    folder_path,
                    file
                )

                result = is_drug_detected(
                    file_path=file_path,
                    frame_skip=1
                )

                print("\n================================")
                print(f"FILE: {file}")
                print(f"Drug Detected: {result}")
                print("================================")
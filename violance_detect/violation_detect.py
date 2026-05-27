import cv2
import os
from ultralytics import YOLO
import time

# =========================================================
# LOAD MODEL ONCE (GLOBAL)
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "violence_detect_yolo8s.pt"
)

model = YOLO(MODEL_PATH)

# =========================================================
# CONFIG
# =========================================================

CONF_THRESHOLD = 0.7

MIN_BOX_AREA = 500

CONSEC_FRAMES_THRESHOLD = 10

DURATION_THRESHOLD_SEC = 1.5

HIT_RATIO_THRESHOLD = 0.15

# =========================================================
# IMAGE DETECTION
# =========================================================

def detect_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False

    results = model(
        image,
        conf=CONF_THRESHOLD
    )

    if results[0].boxes is None:
        return False

    for det in results[0].boxes:

        cls = int(det.cls)

        conf = float(det.conf)

        if (
            cls == 1
            and conf >= CONF_THRESHOLD
        ):

            x1, y1, x2, y2 = map(
                int,
                det.xyxy[0]
            )

            area = (
                (x2 - x1)
                * (y2 - y1)
            )

            if area >= MIN_BOX_AREA:

                print(
                    f"Violence detected in image "
                    f"with confidence "
                    f"{conf:.2f}"
                )

                return True

    return False


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return {
            "detected": False
        }

    fps = (
        cap.get(cv2.CAP_PROP_FPS)
        or 25
    )

    consecutive = 0

    max_consecutive = 0

    total_hits = 0

    total_frames = 0

    confidence_sum = 0.0

    # tracks the current consecutive window
    window_start_time = None
    window_end_time = None

    # tracks the BEST (longest) consecutive window — used in final result
    best_start_time = None
    best_end_time = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        detected_in_frame = False

        # =================================================
        # CURRENT VIDEO TIMESTAMP
        # =================================================

        timestamp_sec = (
            cap.get(cv2.CAP_PROP_POS_MSEC)
            / 1000
        )

        # =================================================
        # MODEL INFERENCE
        # =================================================

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        if results[0].boxes is not None:

            for det in results[0].boxes:

                cls = int(det.cls)

                conf = float(det.conf)

                if (
                    cls == 1
                    and conf >= CONF_THRESHOLD
                ):

                    x1, y1, x2, y2 = map(
                        int,
                        det.xyxy[0]
                    )

                    area = (
                        (x2 - x1)
                        * (y2 - y1)
                    )

                    if area < MIN_BOX_AREA:
                        continue

                    detected_in_frame = True

                    total_hits += 1

                    confidence_sum += conf

        # =================================================
        # TEMPORAL TRACKING
        # =================================================

        if detected_in_frame:

            if window_start_time is None:
                window_start_time = timestamp_sec

            window_end_time = timestamp_sec

            consecutive += 1

            if consecutive > max_consecutive:
                max_consecutive = consecutive
                # save this window as the best
                best_start_time = window_start_time
                best_end_time = window_end_time

        else:

            consecutive = 0
            window_start_time = None
            window_end_time = None

    cap.release()

    # =====================================================
    # NO DETECTIONS
    # =====================================================

    if total_hits == 0:

        return {
            "detected": False
        }

    # =====================================================
    # METRICS
    # =====================================================

    duration_sec = (
        max_consecutive / fps
    )

    hit_ratio = (
        total_hits / total_frames
    )

    avg_conf = (
        confidence_sum / total_hits
    )

    # =====================================================
    # DURATION TEXT
    # =====================================================

    if best_start_time is not None and best_end_time is not None:

        detected_duration = (
            best_end_time
            - best_start_time
        )

        duration_text = (
            f"from "
            f"{best_start_time:.2f}s "
            f"to "
            f"{best_end_time:.2f}s "
            f"(duration: "
            f"{detected_duration:.2f}s)"
        )

    else:

        detected_duration = 0

        duration_text = (
            "duration unavailable"
        )

    # =====================================================
    # PRODUCTION DECISION
    # =====================================================

    if duration_sec >= DURATION_THRESHOLD_SEC:

        print(
            f"Violence detected based on "
            f"duration {duration_text}"
        )

        return {
            "detected": True,

            "start_time": round(
                best_start_time, 2
            ) if best_start_time is not None else None,

            "end_time": round(
                best_end_time, 2
            ) if best_end_time is not None else None,

            "duration": round(
                detected_duration, 2
            ),

            "reason": "duration",

            "max_consecutive_frames": (
                max_consecutive
            ),

            "hit_ratio": round(
                hit_ratio, 2
            ),

            "avg_confidence": round(
                avg_conf, 2
            )
        }

    # =====================================================
    # CONSECUTIVE FRAMES LOGIC
    # =====================================================

    if (
        max_consecutive
        >= CONSEC_FRAMES_THRESHOLD
    ):

        print(
            f"Violence detected based on "
            f"consecutive frames "
            f"({max_consecutive}) "
            f"{duration_text}"
        )

        return {
            "detected": True,

            "start_time": round(
                best_start_time, 2
            ) if best_start_time is not None else None,

            "end_time": round(
                best_end_time, 2
            ) if best_end_time is not None else None,

            "duration": round(
                detected_duration, 2
            ),

            "reason": "consecutive_frames",

            "max_consecutive_frames": (
                max_consecutive
            ),

            "hit_ratio": round(
                hit_ratio, 2
            ),

            "avg_confidence": round(
                avg_conf, 2
            )
        }

    # =====================================================
    # HIT RATIO LOGIC
    # =====================================================

    if (
        hit_ratio >= HIT_RATIO_THRESHOLD
        and avg_conf >= 0.6
    ):

        print(
            f"Violence detected based on "
            f"hit ratio "
            f"{hit_ratio:.2%} "
            f"with avg confidence "
            f"{avg_conf:.2f} "
            f"{duration_text}"
        )

        return {
            "detected": True,

            "start_time": round(
                best_start_time, 2
            ) if best_start_time is not None else None,

            "end_time": round(
                best_end_time, 2
            ) if best_end_time is not None else None,

            "duration": round(
                detected_duration, 2
            ),

            "reason": "hit_ratio",

            "max_consecutive_frames": (
                max_consecutive
            ),

            "hit_ratio": round(
                hit_ratio, 2
            ),

            "avg_confidence": round(
                avg_conf, 2
            )
        }

    return {
        "detected": False
    }


# =========================================================
# UNIVERSAL ENTRY POINT
# =========================================================

def is_violence_detected(file_path):
    """
    IMAGE:
        Returns only True / False

    VIDEO:
        Returns structured metadata
    """

    if not os.path.exists(file_path):

        return False

    ext = file_path.lower().split(".")[-1]

    image_exts = {
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    }

    video_exts = {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm"
    }

    # =====================================================
    # IMAGE
    # =====================================================

    if ext in image_exts:

        return detect_image(file_path)

    # =====================================================
    # VIDEO
    # =====================================================

    elif ext in video_exts:

        return detect_video(file_path)

    # =====================================================
    # INVALID
    # =====================================================

    else:

        print(
            f"Unsupported file format: "
            f"{ext}"
        )

        return False


# =========================================================
# MAIN
# =========================================================

# if __name__ == "__main__":

#     start_time = time.time()

#     test_file = "videos/v4_test.mp4"

#     result = is_violence_detected(
#         test_file
#     )

#     print(result)

#     end_time = time.time()

#     print(
#         f"Processing time: "
#         f"{end_time - start_time:.2f} "
#         f"seconds"
#     )
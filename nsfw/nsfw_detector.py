import os
import cv2
import uuid
import tempfile
from nudenet import NudeDetector

# ----------------------------
# Init model once (IMPORTANT)
# ----------------------------
detector = NudeDetector()

# ----------------------------
# NSFW policy
# ----------------------------
HARD_NSFW = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED"
}

THRESHOLD = 0.5

VIDEO_NSFW_FRAME_LIMIT = 3


# ----------------------------
# Image NSFW detection
# ----------------------------
def image_nsfw(image_path: str) -> bool:
    """
    Returns True if image is NSFW
    """

    try:

        detections = detector.detect(image_path)

        print(
            "[NSFW][IMAGE] Detections:",
            detections
        )

    except Exception as e:

        print(
            "[NSFW][IMAGE] Detection failed:",
            e
        )

        return False

    for d in detections:

        if (
            d.get("class") in HARD_NSFW
            and d.get("score", 0) >= THRESHOLD
        ):

            print(
                "[NSFW][IMAGE] HARD NSFW detected"
            )

            return True

    return False


# ----------------------------
# Video NSFW detection
# ----------------------------
def video_nsfw(
    video_path: str,
    skip_frames: int = 10
):
    """
    Returns structured response for videos.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            "[NSFW][VIDEO] Failed to open video:",
            video_path
        )

        return {
            "detected": False
        }

    frame_count = 0

    nsfw_frames = 0

    detection_start_time = None

    detection_end_time = None

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # =============================================
            # FRAME SKIP LOGIC
            # =============================================

            if (
                skip_frames > 0
                and frame_count % (skip_frames + 1) != 1
            ):
                continue

            # =============================================
            # CURRENT VIDEO TIMESTAMP
            # =============================================

            timestamp_sec = (
                cap.get(cv2.CAP_PROP_POS_MSEC)
                / 1000
            )

            # =============================================
            # CREATE TEMP FILE
            # =============================================

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as tmp:

                temp_path = tmp.name

            cv2.imwrite(temp_path, frame)

            try:

                detections = detector.detect(
                    temp_path
                )

            except Exception as e:

                print(
                    "[NSFW][VIDEO] Detection error:",
                    e
                )

                detections = []

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            frame_has_nsfw = False

            # =============================================
            # DETECTION LOOP
            # =============================================

            for d in detections:

                if (
                    d.get("class") in HARD_NSFW
                    and d.get("score", 0) >= THRESHOLD
                ):

                    frame_has_nsfw = True

                    nsfw_frames += 1

                    # =====================================
                    # DURATION TRACKING
                    # =====================================

                    if detection_start_time is None:

                        detection_start_time = (
                            timestamp_sec
                        )

                    detection_end_time = (
                        timestamp_sec
                    )

                    print(
                        f"[NSFW][VIDEO] "
                        f"NSFW frame detected "
                        f"({nsfw_frames}/"
                        f"{VIDEO_NSFW_FRAME_LIMIT}) "
                        f"at "
                        f"{timestamp_sec:.2f}s"
                    )

                    break

            # =============================================
            # RESET TEMPORAL STATE
            # =============================================

            if not frame_has_nsfw:

                detection_start_time = None

                detection_end_time = None

            # =============================================
            # FINAL DECISION
            # =============================================

            if nsfw_frames >= VIDEO_NSFW_FRAME_LIMIT:

                if (
                    detection_start_time is not None
                    and detection_end_time is not None
                ):

                    duration = (
                        detection_end_time
                        - detection_start_time
                    )

                    print(
                        f"[NSFW][VIDEO] "
                        f"HARD NSFW detected "
                        f"from "
                        f"{detection_start_time:.2f}s "
                        f"to "
                        f"{detection_end_time:.2f}s "
                        f"(duration: "
                        f"{duration:.2f}s)"
                    )

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

                        "nsfw_frames": nsfw_frames
                    }

                else:

                    print(
                        "[NSFW][VIDEO] "
                        "HARD NSFW detected"
                    )

                    return {
                        "detected": True
                    }

    finally:

        cap.release()

    return {
        "detected": False
    }


# ----------------------------
# Unified NSFW entry function
# ----------------------------
def is_nsfw(path: str):
    """
    Detect NSFW for image or video

    IMAGE:
        Returns only True / False

    VIDEO:
        Returns structured metadata
    """

    if not os.path.exists(path):

        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    video_exts = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    print(f"[NSFW] Checking file: {path}")

    # =============================================
    # IMAGE
    # =============================================

    if ext in image_exts:

        return image_nsfw(path)

    # =============================================
    # VIDEO
    # =============================================

    if ext in video_exts:

        return video_nsfw(
            video_path=path
        )

    raise ValueError(
        f"[NSFW] Unsupported file type: {ext}"
    )
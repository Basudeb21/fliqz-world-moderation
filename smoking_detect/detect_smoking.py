from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

from PIL import Image

import torch
import requests
from io import BytesIO
import time
import cv2
import os

# =========================================================
# LOAD MODEL + PROCESSOR
# =========================================================

model_name = (
    "dima806/smoker_image_classification"
)

processor = (
    AutoImageProcessor.from_pretrained(
        model_name
    )
)

model = (
    AutoModelForImageClassification
    .from_pretrained(model_name)
)

# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

model.to(device)

model.eval()

# =========================================================
# IMAGE PREDICTION
# =========================================================

def predict_smoking(
    image_path_or_url
):

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    if (
        isinstance(image_path_or_url, str)
        and image_path_or_url.startswith(
            ('http://', 'https://')
        )
    ):

        response = requests.get(
            image_path_or_url
        )

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

    else:

        image = Image.open(
            image_path_or_url
        ).convert("RGB")

    # =====================================================
    # PREPROCESS
    # =====================================================

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    # =====================================================
    # INFERENCE
    # =====================================================

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = (
            torch.nn.functional.softmax(
                logits,
                dim=-1
            )
        )

        predicted_class_idx = (
            logits.argmax(-1).item()
        )

        predicted_label = (
            model.config.id2label[
                predicted_class_idx
            ]
        )

        confidence = (
            probabilities[0][
                predicted_class_idx
            ].item()
        )

    result = {
        "label": predicted_label,
        "confidence": round(
            confidence * 100,
            2
        ),
        "is_smoking": (
            predicted_label.lower()
            == "smoking"
        )
    }

    return result


# =========================================================
# VIDEO DETECTION
# =========================================================

def detect_smoking_in_video(
    video_path,
    frame_skip=15,
    threshold_low=0.70,
    threshold_high=0.90,
    min_frames=5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return {
            "detected": False
        }

    smoking_frame_count = 0

    frame_index = 0

    detection_start_time = None

    detection_end_time = None

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =================================================
        # FRAME SKIP
        # =================================================

        if frame_index % frame_skip != 0:

            frame_index += 1

            continue

        # =================================================
        # CONVERT FRAME
        # =================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame_rgb)

        # =================================================
        # PREPROCESS
        # =================================================

        inputs = processor(
            images=image,
            return_tensors="pt"
        ).to(device)

        # =================================================
        # INFERENCE
        # =================================================

        with torch.no_grad():

            outputs = model(**inputs)

            logits = outputs.logits

            probs = (
                torch.nn.functional.softmax(
                    logits,
                    dim=-1
                )
            )

            pred_idx = (
                logits.argmax(-1).item()
            )

            label = (
                model.config.id2label[
                    pred_idx
                ].lower()
            )

            confidence = (
                probs[0][pred_idx].item()
            )

        # =================================================
        # CURRENT VIDEO TIMESTAMP
        # =================================================

        timestamp_sec = (
            cap.get(cv2.CAP_PROP_POS_MSEC)
            / 1000
        )

        # =================================================
        # DEBUG PRINT
        # =================================================

        print(
            f"Frame {frame_index}: "
            f"{label} "
            f"({confidence*100:.2f}%) "
            f"at {timestamp_sec:.2f}s"
        )

        # =================================================
        # SMOKING DETECTED
        # =================================================

        if label == "smoking":

            # =============================================
            # START/END TIME TRACKING
            # =============================================

            if detection_start_time is None:

                detection_start_time = (
                    timestamp_sec
                )

            detection_end_time = (
                timestamp_sec
            )

            # =============================================
            # HIGH CONFIDENCE EARLY STOP
            # =============================================

            if confidence >= threshold_high:

                duration = (
                    detection_end_time
                    - detection_start_time
                )

                cap.release()

                print(
                    f"🔥 Smoking detected from "
                    f"{detection_start_time:.2f}s "
                    f"to "
                    f"{detection_end_time:.2f}s "
                    f"(duration: "
                    f"{duration:.2f}s) "
                    f"with HIGH confidence"
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

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "type": "smoking"
                }

            # =============================================
            # MODERATE CONFIDENCE FRAME COUNT
            # =============================================

            if confidence >= threshold_low:

                smoking_frame_count += 1

            # =============================================
            # MULTI FRAME DETECTION
            # =============================================

            if smoking_frame_count >= min_frames:

                duration = (
                    detection_end_time
                    - detection_start_time
                )

                cap.release()

                print(
                    f"🔥 Smoking detected from "
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

                    "confidence": round(
                        confidence * 100,
                        2
                    ),

                    "type": "smoking"
                }

        else:

            # =============================================
            # RESET TEMPORAL STATE
            # =============================================

            smoking_frame_count = 0

            detection_start_time = None

            detection_end_time = None

        frame_index += 1

    cap.release()

    end_time = time.time()

    print(
        f"No smoking detected | "
        f"Time: "
        f"{end_time - start_time:.2f}s"
    )

    return {
        "detected": False
    }


# =========================================================
# UNIVERSAL API
# =========================================================

def is_smoking_detected(
    media_path,
    frame_skip=15
):
    """
    IMAGE:
        Returns only True / False

    VIDEO:
        Returns structured metadata
    """

    image_exts = (
        '.jpg',
        '.jpeg',
        '.png',
        '.bmp',
        '.webp'
    )

    video_exts = (
        '.mp4',
        '.avi',
        '.mov',
        '.mkv',
        '.webm'
    )

    # =====================================================
    # URL IMAGE
    # =====================================================

    if (
        isinstance(media_path, str)
        and media_path.startswith(
            ('http://', 'https://')
        )
    ):

        result = predict_smoking(
            media_path
        )

        return result["is_smoking"]

    # =====================================================
    # FILE EXTENSION
    # =====================================================

    _, ext = os.path.splitext(
        media_path.lower()
    )

    # =====================================================
    # IMAGE
    # =====================================================

    if ext in image_exts:

        result = predict_smoking(
            media_path
        )

        return result["is_smoking"]

    # =====================================================
    # VIDEO
    # =====================================================

    elif ext in video_exts:

        return detect_smoking_in_video(
            video_path=media_path,
            frame_skip=frame_skip
        )

    # =====================================================
    # INVALID
    # =====================================================

    else:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )
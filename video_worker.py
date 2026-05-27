#video_worker.py
import os
import cv2
import redis
import time
import json
from pathlib import Path
from PIL import Image

from animal_detect.animal_detect import AnimalDetector
from weapon_detect.weapon_detect import is_weapon_detected
from drugs_detect.drugs_detect import is_drug_detected
from model import DEVICE

from face_detect.minor_detect import is_minor
from meetup_detect.personal_details_detect import detect_personal_info
from violance_detect.violation_detect import is_violence_detected
from alcohol_detect.detect_alcohol import is_alcohol_detected
from smoking_detect.detect_smoking import is_smoking_detected
from nsfw.nsfw_detector import is_nsfw

from dynamic_update import dynamic_update, insert_detection_timestamps
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, VIDEO_QUEUE, REDIS_BRPOP_TIMEOUT


# =====================================================
# REDIS
# =====================================================
print("[INIT] Connecting to Redis...")
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)
print("[INIT] Redis connected")

# instantiate tested detectors
animal_detector = AnimalDetector()


# =====================================================
# PATH HANDLING
# =====================================================
POSSIBLE_BASE_PATHS = [
    os.getcwd(), 
    "/var/www/html/admin.fliqzworld.com/public/storage",
    "/var/www/html/admin.fliqzworld.com/storage",
    "/var/www/html/admin.fliqzworld.com/public_html/storage",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage",
    "C:/CodeX/fliqz-world-moderation"
]

def get_valid_base_path():
    for base in POSSIBLE_BASE_PATHS:
        if os.path.exists(base):
            print(f"[PATH] Using base path: {base}")
            return base
    print("[PATH] Using fallback base path")
    return POSSIBLE_BASE_PATHS[0]

SERVER_STORAGE_PATH = get_valid_base_path()

def normalize_file_path(original_file: str) -> str:
    print(f"[PATH] Normalizing file path: {original_file}")
    clean_path = (
        original_file.replace("\\", "/")
        .replace("//", "/")
        .strip()
        .lstrip("/")
    )

    for base in POSSIBLE_BASE_PATHS:
        full_path = os.path.join(base, clean_path).replace("\\", "/")
        if os.path.exists(full_path):
            print(f"[PATH] Resolved path: {full_path}")
            return full_path

    fallback = os.path.join(SERVER_STORAGE_PATH, clean_path).replace("\\", "/")
    print(f"[PATH] Using fallback path: {fallback}")
    return fallback


# =====================================================
# MEDIA TYPES
# =====================================================
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# =====================================================
# STATUS HELPER
# =====================================================
def set_process_status(payload: dict, status: int):
    """Update only ai_process_status field in DB."""
    try:
        success, result = dynamic_update(
            payload=payload,
            ai_process_status=status
        )
        print(f"📊 ai_process_status → {status} ({'✅' if success else '❌ ' + result})")
    except Exception as e:
        print(f"❌ Status update error: {e}")


def try_dynamic_update(orig_payload: dict, **kwargs):
    """Call dynamic_update and if row_not_found try fallback by post_id."""
    import copy
    payload = copy.deepcopy(orig_payload)
    success, status = dynamic_update(payload=payload, **kwargs)

    if not success and status == "row_not_found":
        # attempt fallback using post_id if available
        post_id = payload.get("data", {}).get("post_id") or payload.get("post_id")
        if post_id is not None:
            payload_fb = copy.deepcopy(orig_payload)
            payload_fb["primary_key"] = "post_id"
            payload_fb["key_value"] = post_id
            try:
                success2, status2 = dynamic_update(payload=payload_fb, **kwargs)
                return success2, status2
            except Exception as e:
                return False, str(e)

    return success, status


# =====================================================
# KEYFRAME EXTRACTION
# =====================================================
def extract_candidate_frames(
    video_path: str,
    max_frames: int = 12,
    scene_threshold: float = 25.0
):
    print(f"[VIDEO] Opening video for keyframe extraction: {video_path}")
    cap = cv2.VideoCapture(video_path)

    prev_gray = None
    candidates = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = diff.mean()

            if score > scene_threshold:
                print(f"[KEYFRAME] Scene change at frame {frame_idx} (score={score:.2f})")
                candidates.append(frame)

        prev_gray = gray

        if len(candidates) >= max_frames:
            print("[KEYFRAME] Reached max candidate frames")
            break

    cap.release()

    # fallback
    if not candidates:
        print("[KEYFRAME] No scene changes detected, using fallback frame")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            candidates.append(frame)
        cap.release()

    print(f"[KEYFRAME] Selected {len(candidates)} candidate frames")
    return candidates[:max_frames]




   

# =====================================================
# PROCESS REDIS MESSAGE
# =====================================================
def process_redis(payload: dict):
    print("\n[WORKER] New moderation task")

    payload["table_name"] = payload.get("table")
    payload["primary_key"] = "id"
    payload["key_value"] = payload.get("id")

    file_rel = payload.get("data", {}).get("file")
    if not file_rel:
        print("[SKIP] No file path")
        return

    file_path = normalize_file_path(file_rel)
    if not os.path.exists(file_path):
        print("[SKIP] File not found:", file_path)
        return

    ext = Path(file_path).suffix.lower()
    print("[FILE]", file_path)

    # ✅ STATUS → 2 (Processing started)
    set_process_status(payload, 2)

    # -----------------------------
    # FLAGS
    # -----------------------------
    animal_detected = False
    das_detected = False
    drugs_detected = False
    alcohol_detected = False
    smoking_detected = False
    weapon_detected = False
    minor_detected = False
    personal_info_detected = False
    violence_detected = False
    nsfw_detected = None

    # collect timestamp records for DB (start early so minor/personal info can append)
    timestamp_records = []

    # =====================================================
    # 1️⃣ MINOR
    # =====================================================
    try:
        minor_result = is_minor(file_path)
        if isinstance(minor_result, dict):
            minor_detected = bool(minor_result.get("detected", False))
            if minor_detected and minor_result.get("start_time") is not None:
                timestamp_records.append({
                    "label": "minor",
                    "start_time": minor_result.get("start_time"),
                    "end_time": minor_result.get("end_time"),
                    "duration": minor_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": minor_result.get("type") if minor_result.get("type") is not None else None
                })
        else:
            minor_detected = bool(minor_result)

        print("[CHECK] Minor:", minor_detected)
    except Exception as e:
        print("[ERROR] Minor:", e)

    if minor_detected:
        if nsfw_detected is None:
            try:
                nsfw_result = is_nsfw(file_path)
                if isinstance(nsfw_result, dict):
                    nsfw_detected = bool(nsfw_result.get("detected", False))
                    if nsfw_detected and nsfw_result.get("start_time") is not None:
                        timestamp_records.append({
                            "label": "nsfw",
                            "start_time": nsfw_result.get("start_time"),
                            "end_time": nsfw_result.get("end_time"),
                            "duration": nsfw_result.get("duration"),
                            "post_id": payload.get("data", {}).get("post_id"),
                            "detector_type": nsfw_result.get("type") if nsfw_result.get("type") is not None else None
                        })
                else:
                    nsfw_detected = bool(nsfw_result)
                print("[CHECK] NSFW (minor):", nsfw_detected)
            except Exception as e:
                print("[ERROR] NSFW (minor):", e)

        if nsfw_detected:
            print("[STOP] Child + NSFW")
            # ✅ STATUS → 3 (Done - early exit)
            success, status = try_dynamic_update(
                payload,
                minor_detected=1 if minor_detected else 0,
                nsfw_detected=1 if nsfw_detected else 0,
                ai_process_status=3
            )
            print("[DB RESULT]", status if success else f"FAILED ({status})")
            if timestamp_records:
                try:
                    ts_success, ts_res = insert_detection_timestamps(payload, timestamp_records)
                    print("[TS INSERT]", ts_res if ts_success else f"FAILED ({ts_res})")
                except Exception as e:
                    print("[TS INSERT] Exception:", e)
            return

    # =====================================================
    # 2️⃣ PII
    # =====================================================
    try:
        personal_info_result = detect_personal_info(file_path)
        if isinstance(personal_info_result, dict):
            personal_info_detected = bool(personal_info_result.get("detected", False))
            if personal_info_detected and personal_info_result.get("start_time") is not None:
                timestamp_records.append({
                    "label": "personal_info",
                    "start_time": personal_info_result.get("start_time"),
                    "end_time": personal_info_result.get("end_time"),
                    "duration": personal_info_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": personal_info_result.get("type") if personal_info_result.get("type") is not None else None
                })
        else:
            personal_info_detected = bool(personal_info_result)

        print("[CHECK] PII:", personal_info_detected)
    except Exception as e:
        print("[ERROR] PII:", e)

    if personal_info_detected:
        print("[STOP] Personal Info")
        # ✅ STATUS → 3 (Done - early exit)
        success, status = try_dynamic_update(
            payload,
            personal_info_detected=1 if personal_info_detected else 0,
            ai_process_status=3
        )
        print("[DB RESULT]", status if success else f"FAILED ({status})")
        if timestamp_records:
            try:
                ts_success, ts_res = insert_detection_timestamps(payload, timestamp_records)
                print("[TS INSERT]", ts_res if ts_success else f"FAILED ({ts_res})")
            except Exception as e:
                print("[TS INSERT] Exception:", e)
        return

    # =====================================================
    # 3️⃣ ANIMAL & WEAPON (use tested detectors)
    # =====================================================
    if ext not in VIDEO_EXT:
        print("[SKIP] Unsupported type")
        return

    # timestamp_records list already initialized earlier

    # animal detection (may return dict for video)
    try:
        print("[CHECK] Animal...")
        animal_result = animal_detector.is_animal(file_path, quick_mode=True)
        if isinstance(animal_result, dict):
            animal_detected = bool(animal_result.get("detected", False))
            if animal_detected and animal_result.get("start_time") is not None:
                timestamp_records.append({
                    "label": "animal",
                    "start_time": animal_result.get("start_time"),
                    "end_time": animal_result.get("end_time"),
                    "duration": animal_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": animal_result.get("type") if animal_result.get("type") is not None else None
                })
        else:
            animal_detected = bool(animal_result)
    except Exception as e:
        print("[ERROR] Animal:", e)

    # weapon detection (YOLO video returns dict)
    try:
        print("[CHECK] Weapon...")
        weapon_result = is_weapon_detected(file_path)
        if isinstance(weapon_result, dict):
            weapon_detected = bool(weapon_result.get("detected", False))
            if weapon_detected and weapon_result.get("start_time") is not None:
                timestamp_records.append({
                    "label": "weapon",
                    "start_time": weapon_result.get("start_time"),
                    "end_time": weapon_result.get("end_time"),
                    "duration": weapon_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": weapon_result.get("type") if weapon_result.get("type") is not None else None
                })
        else:
            weapon_detected = bool(weapon_result)
    except Exception as e:
        print("[ERROR] Weapon:", e)

    # =====================================================
    # 3️⃣a ANIMAL + NSFW
    # =====================================================
    if animal_detected:
        if nsfw_detected is None:
            try:
                nsfw_result = is_nsfw(file_path)
                if isinstance(nsfw_result, dict):
                    nsfw_detected = bool(nsfw_result.get("detected", False))
                    if nsfw_detected and nsfw_result.get("start_time") is not None:
                        timestamp_records.append({
                            "label": "nsfw",
                            "start_time": nsfw_result.get("start_time"),
                            "end_time": nsfw_result.get("end_time"),
                            "duration": nsfw_result.get("duration"),
                            "post_id": payload.get("data", {}).get("post_id"),
                            "detector_type": nsfw_result.get("type") if nsfw_result.get("type") is not None else None
                        })
                else:
                    nsfw_detected = bool(nsfw_result)

                print("[CHECK] NSFW (animal):", nsfw_detected)
            except Exception as e:
                print("[ERROR] NSFW (animal):", e)

        if nsfw_detected:
            print("[STOP] Animal + NSFW")
            # ✅ STATUS → 3 (Done - early exit)
            success, status = try_dynamic_update(
                payload,
                animal_detected=animal_detected,
                weapon_detected=weapon_detected,
                nsfw_detected=1 if nsfw_detected else 0,
                ai_process_status=3
            )
            print("[DB RESULT]", status if success else f"FAILED ({status})")
            # INSERT TIMESTAMPS — was missing here, causing animal/weapon/nsfw
            # timestamp records collected before this point to be silently dropped
            if timestamp_records:
                try:
                    ts_success, ts_res = insert_detection_timestamps(payload, timestamp_records)
                    print("[TS INSERT]", ts_res if ts_success else f"FAILED ({ts_res})")
                except Exception as e:
                    print("[TS INSERT] Exception:", e)
            return

    # =====================================================
    # 4️⃣ VIOLENCE
    # =====================================================
    try:
        violence_result = is_violence_detected(file_path)
        if isinstance(violence_result, dict):
            violence_detected = bool(violence_result.get("detected", False))
            # Add timestamp record even if start/end not provided (best-effort)
            if violence_detected:
                if violence_result.get("start_time") is not None:
                    timestamp_records.append({
                        "label": "violence",
                        "start_time": violence_result.get("start_time"),
                        "end_time": violence_result.get("end_time"),
                        "duration": violence_result.get("duration"),
                        "post_id": payload.get("data", {}).get("post_id"),
                        "detector_type": violence_result.get("type") if violence_result.get("type") is not None else None
                    })
                else:
                    # insert a minimal record so moderators can see an event occurred
                    timestamp_records.append({
                        "label": "violence",
                        "post_id": payload.get("data", {}).get("post_id"),
                        "detector_type": violence_result.get("type") if violence_result.get("type") is not None else None,
                        "metadata": "detected_without_times"
                    })
        else:
            violence_detected = bool(violence_result)
        print("[CHECK] Violence:", violence_detected)
    except Exception as e:
        print("[ERROR] Violence:", e)

    # =====================================================
    # 5️⃣ ALCOHOL DETECTION (YOLO)
    # =====================================================
    try:
        print("[CHECK] Alcohol...")
        alcohol_result = is_alcohol_detected(file_path)
        if isinstance(alcohol_result, dict):
            alcohol_detected = bool(alcohol_result.get("detected", False))
            if alcohol_detected:
                if alcohol_result.get("start_time") is not None:
                    timestamp_records.append({
                        "label": "alcohol",
                        "start_time": alcohol_result.get("start_time"),
                        "end_time": alcohol_result.get("end_time"),
                        "duration": alcohol_result.get("duration"),
                        "post_id": payload.get("data", {}).get("post_id"),
                        "detector_type": alcohol_result.get("type") if alcohol_result.get("type") is not None else None
                    })
                else:
                    # No timestamps returned but detection confirmed — still record it
                    timestamp_records.append({
                        "label": "alcohol",
                        "post_id": payload.get("data", {}).get("post_id"),
                        "detector_type": alcohol_result.get("type") if alcohol_result.get("type") is not None else None
                    })
        else:
            alcohol_detected = bool(alcohol_result)
    except Exception as e:
        print("[ERROR] Alcohol:", e)

    # =====================================================
    # 6️⃣ SMOKING DETECTION (YOLO)
    # =====================================================
    try:
        print("[CHECK] Smoking...")
        smoking_result = is_smoking_detected(file_path)
        if isinstance(smoking_result, dict):
            smoking_detected = bool(smoking_result.get("detected", False))
            if smoking_detected:
                if smoking_result.get("start_time") is not None:
                    timestamp_records.append({
                        "label": "smoking",
                        "start_time": smoking_result.get("start_time"),
                        "end_time": smoking_result.get("end_time"),
                        "duration": smoking_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": smoking_result.get("type") if smoking_result.get("type") is not None else None
                })
        else:
            smoking_detected = bool(smoking_result)
    except Exception as e:
        print("[ERROR] Smoking:", e)

    # =====================================================
    # 7️⃣ DRUGS DETECTION
    # =====================================================
    try:
        print("[CHECK] Drugs...")
        drugs_result = is_drug_detected(file_path)
        if isinstance(drugs_result, dict):
            drugs_detected = bool(drugs_result.get("detected", False))
            if drugs_detected and drugs_result.get("start_time") is not None:
                timestamp_records.append({
                    "label": "drugs",
                    "start_time": drugs_result.get("start_time"),
                    "end_time": drugs_result.get("end_time"),
                    "duration": drugs_result.get("duration"),
                    "post_id": payload.get("data", {}).get("post_id"),
                    "detector_type": drugs_result.get("type") if drugs_result.get("type") is not None else None
                })
        else:
            drugs_detected = bool(drugs_result)
        print("[CHECK] Drugs:", drugs_detected)
    except Exception as e:
        print("[ERROR] Drugs:", e)

    # MAP TO DAS (alcohol, smoking, drugs)
    das_detected = alcohol_detected or smoking_detected or drugs_detected

    # =====================================================
    # FINAL NSFW CHECK
    # =====================================================
    if nsfw_detected is None:
        try:
            print("[CHECK] Final NSFW check...")
            nsfw_result = is_nsfw(file_path)
            if isinstance(nsfw_result, dict):
                nsfw_detected = bool(nsfw_result.get("detected", False))
                if nsfw_detected and nsfw_result.get("start_time") is not None:
                    timestamp_records.append({
                        "label": "nsfw",
                        "start_time": nsfw_result.get("start_time"),
                        "end_time": nsfw_result.get("end_time"),
                        "duration": nsfw_result.get("duration"),
                        "post_id": payload.get("data", {}).get("post_id"),
                        "detector_type": nsfw_result.get("type") if nsfw_result.get("type") is not None else None
                    })
            else:
                nsfw_detected = bool(nsfw_result)
        except Exception as e:
            print("[ERROR] NSFW:", e)

    # =====================================================
    # FINAL UPDATE — STATUS → 3 (Fully done)
    # =====================================================
    print("[FINAL] Saving results")
    print("✅ Detection complete.")
    print(f"   Animal Detected:        {animal_detected}")
    print(f"   Alcohol Detected:       {alcohol_detected}")
    print(f"   Smoking Detected:       {smoking_detected}")
    print(f"   DAS (mapped):           {das_detected}")
    print(f"   Minor Detected:         {minor_detected}")
    print(f"   Personal Info Detected: {personal_info_detected}")
    print(f"   NSFW Detected:          {nsfw_detected}")
    print(f"   Violence Detected:      {violence_detected}")
    print(f"   Weapon Detected:        {weapon_detected}")

    success, status = try_dynamic_update(
        payload,
        animal_detected=animal_detected,
        das_detected=das_detected,
        weapon_detected=weapon_detected,
        minor_detected=minor_detected,
        personal_info_detected=personal_info_detected,
        nsfw_detected=bool(nsfw_detected),   # None → False, True → True, False → False
        violence_detected=violence_detected,
        ai_process_status=3
    )

    print("[DB RESULT]", status if success else f"FAILED ({status})")

    # Insert timestamp records (best-effort)
    if timestamp_records:
        try:
            ts_success, ts_res = insert_detection_timestamps(payload, timestamp_records)
            print("[TS INSERT]", ts_res if ts_success else f"FAILED ({ts_res})")
        except Exception as e:
            print("[TS INSERT] Exception:", e)
    print("[DONE] Moderation completed")


# =====================================================
# WORKER LOOP
# =====================================================
def worker():
    print("🚀 Video Moderation Worker started")
    print("📥 Listening on:", VIDEO_QUEUE)

    while True:
        try:
            item = r.brpop(VIDEO_QUEUE, timeout=REDIS_BRPOP_TIMEOUT)
            if not item:
                time.sleep(0.1)
                continue

            _, message = item

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON")
                continue

            process_redis(payload)

        except Exception as e:
            print("❌ Worker error:", e)
            time.sleep(1)


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    worker()
import os
import cv2
import redis
import time
import json
from pathlib import Path
from PIL import Image
from model import DEVICE
from animal_detect.animal_detect import AnimalDetector
from weapon_detect.weapon_detect import is_weapon_detected
from drugs_detect.drugs_detect import is_drug_detected

from face_detect.minor_detect import is_minor
from meetup_detect.personal_details_detect import detect_personal_info  
from violance_detect.violation_detect import is_violence_detected
from alcohol_detect.detect_alcohol import is_alcohol_detected
from smoking_detect.detect_smoking import is_smoking_detected
from merged_owlvit_detector import run_merged_detection
from nsfw.nsfw_detector import is_nsfw

from dynamic_update import dynamic_update
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, STREAM_QUEUE, REDIS_BRPOP_TIMEOUT

# -----------------------------
# Redis
# -----------------------------
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)


# -----------------------------
# ORIGINAL PATH HANDLING (RESTORED)
# -----------------------------
POSSIBLE_BASE_PATHS = [
    "/var/www/html/admin.fliqzworld.com/public/storage/stream",
    "/var/www/html/admin.fliqzworld.com/storage/stream",
    "/var/www/html/admin.fliqzworld.com/public_html/storage/stream",
    "D:/codex/bots/NSFW-DETECT-BOT/var/www/html/admin.fliqzworld.com/public/storage/stream",
    "C:/CodeX/fliqz-world-moderation/stream"
]

def get_valid_base_path():
    """Auto-detect which base path exists."""
    print("🔍 Checking possible base paths...")
    for base in POSSIBLE_BASE_PATHS:
        if os.path.exists(base):
            print(f"✅ Using detected base path: {base}")
            return base
        else:
            print(f"❌ Not found: {base}")
    print("⚠️ No valid storage path found! Using default first one.")
    return POSSIBLE_BASE_PATHS[0]

SERVER_STORAGE_PATH = get_valid_base_path()

def normalize_file_path(original_file: str) -> str:
    """Convert relative upload paths to absolute filesystem paths."""
    if not original_file:
        return ""

    clean_path = (
        original_file.replace("\\\\", "/")
        .replace("\\", "/")
        .replace("\\/", "/")
        .replace("//", "/")
        .strip()
        .lstrip("/")
    )

    print(f"🧭 Normalizing file path: {original_file} → {clean_path}")

    for base in POSSIBLE_BASE_PATHS:
        full_path = os.path.join(base, clean_path).replace("\\", "/")
        if os.path.exists(full_path):
            print(f"✅ Matched existing file path: {full_path}")
            return full_path

    fallback_path = os.path.join(SERVER_STORAGE_PATH, clean_path).replace("\\", "/")
    print(f"⚠️ Fallback path used: {fallback_path}")
    return fallback_path


# =====================================================
# LOAD MEDIA ONCE (NEW)
# =====================================================
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

def load_media(file_path):
    ext = Path(file_path).suffix.lower()

    # -------- IMAGE --------
    if ext in IMAGE_EXT:
        print("🖼️ Decoding image once in worker")
        return Image.open(file_path).convert("RGB")

    # -------- VIDEO --------
    if ext in VIDEO_EXT:
        print("🎞️ Extracting video frames once in worker")
        cap = cv2.VideoCapture(file_path)
        frames = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # sample every 20 frames
            if frame_id % 20 == 0:
                frames.append(
                    Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                )
            frame_id += 1

        cap.release()
        return frames

    return None



# =====================================================
# PROCESS ONE REDIS MESSAGE
# =====================================================
def process_redis(payload: dict):

    # 1. Map table name
    payload["table_name"] = payload.get("table")

    # 2. Primary key mapping
    payload["primary_key"] = "id"
    payload["key_value"] = payload.get("id")

    # 3. Extract file path from nested data
    data = payload.get("data", {})
    payload["file_path"] = data.get("file")

    # Safety checks
    if not payload["table_name"] or not payload["key_value"]:
        print("❌ Missing DB identifiers, skipping")
        return

    if not payload["file_path"]:
        print("❌ No file in payload, skipping")
        return

    # -------------------------------------------------
    # FILE PATH NORMALIZATION
    # -------------------------------------------------
    original_file = payload["file_path"]
    file_path = normalize_file_path(original_file)

    print(f"\n🖼️ Processing file: {file_path}")

    if not os.path.exists(file_path):
        print("❌ File not found after normalization")
        return



    # -----------------------------
    # FLAGS (DEFAULT FALSE)
    # -----------------------------
    animal_detected = False
    das_detected = False
    alcohol_detected = False
    smoking_detected = False
    minor_detected = False
    personal_info_detected = False
    nsfw_detected = None
    violence_detected = False
    weapon_detected = False
    drugs_detected = False

    # instantiate detectors
    animal_detector = AnimalDetector()



    # =====================================================
    # 1️⃣ MINOR DETECTION
    # =====================================================
    try:
        print("🔍 Checking for minors...")
        minor_detected = is_minor(file_path)
    except Exception as e:
        print("Minor error:", e)

    if minor_detected:
        if nsfw_detected is None:
            try:
                print("🔍 Minor detected → checking NSFW...")
                nsfw_detected = is_nsfw(file_path)
            except Exception as e:
                print("NSFW error:", e)

        if nsfw_detected:
            print("⛔ Minor + NSFW → STOP")
            success, status = dynamic_update(
                payload=payload,
                minor_detected=minor_detected,
                nsfw_detected=nsfw_detected
            )
            print("✅ Detection complete.")
            print(f"   Minor Detected: {minor_detected}")
            print(f"   NSFW Detected: {nsfw_detected}")
            print("💾 DB Update:", status if success else f"FAILED ({status})")
            return

    # =====================================================
    # 2️⃣ PERSONAL INFO DETECTION
    # =====================================================
    try:
        print("🔍 Checking for personal info...")
        personal_info_detected = detect_personal_info(file_path)
    except Exception as e:
        print("PII error:", e)

    if personal_info_detected:
        print("⛔ Personal info detected → STOP")
        success, status = dynamic_update(
            payload=payload,
            personal_info_detected=personal_info_detected
        )
        print("✅ Detection complete.")
        print(f"   Personal Info Detected: {personal_info_detected}")
        print(f"   NSFW Detected: {nsfw_detected}")
        print("💾 DB Update:", status if success else f"FAILED ({status})")
        return

    # =====================================================
    # 3️⃣ ANIMAL / WEAPON / DRUGS DETECTION (replace OWL)
    # =====================================================
    try:
        print("🔍 Checking for animals...")
        animal_result = animal_detector.is_animal(file_path)
        if isinstance(animal_result, dict):
            animal_detected = bool(animal_result.get("detected", False))
        else:
            animal_detected = bool(animal_result)
    except Exception as e:
        print("Animal error:", e)

    try:
        print("🔍 Checking for weapons...")
        weapon_result = is_weapon_detected(file_path)
        if isinstance(weapon_result, dict):
            weapon_detected = bool(weapon_result.get("detected", False))
        else:
            weapon_detected = bool(weapon_result)
    except Exception as e:
        print("Weapon error:", e)

    try:
        print("🔍 Checking for drugs...")
        drugs_result = is_drug_detected(file_path)
        if isinstance(drugs_result, dict):
            drugs_detected = bool(drugs_result.get("detected", False))
        else:
            drugs_detected = bool(drugs_result)
    except Exception as e:
        print("Drugs error:", e)

    if animal_detected:
        if nsfw_detected is None:
            try:
                nsfw_result = is_nsfw(file_path)
                if isinstance(nsfw_result, dict):
                    nsfw_detected = bool(nsfw_result.get("detected", False))
                else:
                    nsfw_detected = bool(nsfw_result)
            except Exception as e:
                print("NSFW error:", e)

        if nsfw_detected:
            print("⛔ Animal + NSFW → STOP")
            success, status = dynamic_update(
                payload=payload,
                animal_detected=animal_detected,
                weapon_detected=weapon_detected,
                nsfw_detected=nsfw_detected
            )
            print("✅ Detection complete.")
            print(f"   Animal Detected: {animal_detected}")
            print(f"   Weapon Detected: {weapon_detected}")
            print(f"   Drugs Detected: {drugs_detected}")
            print(f"   NSFW Detected: {nsfw_detected}")
            print("💾 DB Update:", status if success else f"FAILED ({status})")
            return

    # =====================================================
    # 4️⃣ VIOLENCE DETECTION
    # =====================================================
    try:
        print("🔍 Checking for violence...")
        violence_result = is_violence_detected(file_path)
        if isinstance(violence_result, dict):
            violence_detected = bool(violence_result.get("detected", False))
        else:
            violence_detected = bool(violence_result)
    except Exception as e:
        print("Violence error:", e)


    # =====================================================
    # 5️⃣ ALCOHOL DETECTION (YOLO)
    # =====================================================
    try:
        print("🍺 Checking for alcohol...")
        alcohol_result = is_alcohol_detected(file_path)
        if isinstance(alcohol_result, dict):
            alcohol_detected = bool(alcohol_result.get("detected", False))
        else:
            alcohol_detected = bool(alcohol_result)
    except Exception as e:
        print("Alcohol error:", e)

    # =====================================================
    # 6️⃣ SMOKING DETECTION (YOLO)
    # =====================================================
    try:
        print("🚬 Checking for smoking...")
        smoking_result = is_smoking_detected(file_path)
        if isinstance(smoking_result, dict):
            smoking_detected = bool(smoking_result.get("detected", False))
        else:
            smoking_detected = bool(smoking_result)
    except Exception as e:
        print("Smoking error:", e)

    # =====================================================
    # 7️⃣ DRUGS DETECTION
    # =====================================================
    try:
        print("💊 Checking for drugs...")
        drugs_result = is_drug_detected(file_path)
        if isinstance(drugs_result, dict):
            drugs_detected = bool(drugs_result.get("detected", False))
        else:
            drugs_detected = bool(drugs_result)
    except Exception as e:
        print("Drugs error:", e)

    # =====================================================
    # MAP TO DAS (DB COMPATIBILITY)
    # =====================================================
    das_detected = alcohol_detected or smoking_detected or drugs_detected

    # =====================================================
    # ENSURE NSFW WAS AT LEAST CHECKED ONCE
    # =====================================================
    if nsfw_detected is None:
        try:
            print("🔍 Final NSFW check...")
            nsfw_result = is_nsfw(file_path)
            if isinstance(nsfw_result, dict):
                nsfw_detected = bool(nsfw_result.get("detected", False))
            else:
                nsfw_detected = bool(nsfw_result)
        except Exception as e:
            print("NSFW error:", e)
    
    print("✅ Detection complete.")   
    print(f"   Animal Detected: {animal_detected}")
    print(f"   Alcohol Detected: {alcohol_detected}")
    print(f"   Smoking Detected: {smoking_detected}")
    print(f"   DAS (mapped): {das_detected}")
    print(f"   Minor Detected: {minor_detected}")
    print(f"   Personal Info Detected: {personal_info_detected}")
    print(f"   NSFW Detected: {nsfw_detected}")
    print(f"   Violence Detected: {violence_detected}")
    print(f"   Weapon Detected: {weapon_detected}") 


# =====================================================
    # FINAL DB UPDATE (FULL STATE)
    # =====================================================
    print("⌛ Updating DB")
    # -----------------------------
    # DB UPDATE (UPDATE-ONLY)
    # -----------------------------
    success, status = dynamic_update(
        payload=payload,
        animal_detected=animal_detected,
        das_detected=das_detected,
        minor_detected=minor_detected,
        personal_info_detected=personal_info_detected,
        nsfw_detected=nsfw_detected,
        violence_detected=violence_detected,
        weapon_detected=weapon_detected
    )

    print("💾 DB Update:", status if success else f"FAILED ({status})")

# =====================================================
# WORKER LOOP
# =====================================================
def worker():
    print("🚀 Media Moderation Worker started")
    print("📥 Listening on:", STREAM_QUEUE)

    while True:
        try:
            item = r.brpop(STREAM_QUEUE, timeout=REDIS_BRPOP_TIMEOUT)
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


#"{\"type\":\"attachment\",\"table\":\"attachments\",\"id\":\"a83edaf06cb94905ad6d9f20b9e7dfc9\",\"data\":
# {\"file\":\"uploads\\/posts\\/images\\/17665066961999.jpg\",\"type\":\"images\",\"post_id\":512,\"user_id\":6}}"
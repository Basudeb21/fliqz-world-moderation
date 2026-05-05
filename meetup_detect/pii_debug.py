# import re
# import spacy
# import platform
# import pytesseract
# from PIL import Image
# import requests
# from io import BytesIO
# import os
# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode as qr_decode

# # =========================================================
# # Load NLP model (ONCE)
# # =========================================================
# nlp = spacy.load("en_core_web_sm")

# # =========================================================
# # Regex patterns (STRICT)
# # =========================================================
# email_pattern = re.compile(
#     r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
# )

# phone_pattern = re.compile(
#     r'\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b'
# )

# url_pattern = re.compile(
#     r'(https?:\/\/[^\s]+|www\.[^\s]+)'
# )

# # Require NUMBER + STREET WORD → real address
# address_pattern = re.compile(
#     r'\b\d{1,5}\s+(street|st|road|rd|lane|ln|avenue|ave|block|sector|floor)\b',
#     re.IGNORECASE
# )

# PLATFORM_DOMAIN = "myvault-web.codextechnolife.com"

# # =========================================================
# # Tesseract path
# # =========================================================
# if platform.system() == "Windows":
#     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# else:
#     pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# # =========================================================
# # Helper checks (SAFE)
# # =========================================================
# def isEmail(text: str) -> bool:
#     return bool(email_pattern.search(text))


# def hasPhoneNumber(text: str) -> bool:
#     return bool(phone_pattern.search(text))


# def hasForbiddenURL(text: str) -> bool:
#     for match in url_pattern.finditer(text):
#         url = match.group(0).rstrip('.,!?')
#         if PLATFORM_DOMAIN not in url:
#             return True
#     return False


# def hasAddress(text: str) -> bool:
#     return bool(address_pattern.search(text))


# # =========================================================
# # DEBUG REASON ENGINE
# # =========================================================
# def detect_pii_reasons(text: str) -> dict:
#     reasons = {
#         "email": False,
#         "phone": False,
#         "forbidden_url": False,
#         "address": False
#     }

#     if isEmail(text):
#         reasons["email"] = True

#     if hasPhoneNumber(text):
#         reasons["phone"] = True

#     if hasForbiddenURL(text):
#         reasons["forbidden_url"] = True

#     if hasAddress(text):
#         reasons["address"] = True

#     return reasons


# def isPersonalDetails_debug(text: str):
#     reasons = detect_pii_reasons(text)
#     detected = any(reasons.values())
#     return detected, reasons


# # =========================================================
# # QR helpers
# # =========================================================
# def extract_qr_from_frame(frame) -> list[str]:
#     try:
#         decoded = qr_decode(frame)
#         return [
#             obj.data.decode("utf-8", errors="ignore")
#             for obj in decoded
#         ]
#     except Exception:
#         return []


# # =========================================================
# # OCR + QR (Image / URL)
# # =========================================================
# def extract_text_and_qr_from_file(file_path_or_url: str):
#     text = ""
#     qr_payloads = []

#     try:
#         if file_path_or_url.startswith("http"):
#             response = requests.get(file_path_or_url, timeout=10)
#             img = Image.open(BytesIO(response.content))
#         else:
#             img = Image.open(file_path_or_url)

#         text = pytesseract.image_to_string(img)

#         frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         qr_payloads = extract_qr_from_frame(frame)

#     except Exception as e:
#         print(f"[OCR/QR ERROR] {e}")

#     return text, qr_payloads


# # =========================================================
# # OCR + QR (Video)
# # =========================================================
# def detect_personal_info_video(video_path, frame_skip=30) -> bool:
#     if not os.path.exists(video_path):
#         return False

#     cap = cv2.VideoCapture(video_path)
#     frame_id = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_id % frame_skip == 0:
#             text = pytesseract.image_to_string(Image.fromarray(frame))
#             qr_payloads = extract_qr_from_frame(frame)

#             if text.strip():
#                 detected, reasons = isPersonalDetails_debug(text)
#                 if detected:
#                     print("\n🚨 PII DETECTED (VIDEO OCR)")
#                     print("FRAME:", frame_id)
#                     print("TEXT:", repr(text))
#                     print("REASONS:", reasons)
#                     cap.release()
#                     return True

#             for qr_text in qr_payloads:
#                 detected, reasons = isPersonalDetails_debug(qr_text)
#                 if detected:
#                     print("\n🚨 PII DETECTED (VIDEO QR)")
#                     print("FRAME:", frame_id)
#                     print("QR TEXT:", repr(qr_text))
#                     print("REASONS:", reasons)
#                     cap.release()
#                     return True

#         frame_id += 1

#     cap.release()
#     return False


# # =========================================================
# # PUBLIC API
# # =========================================================
# def detect_personal_info(data) -> bool:
#     """
#     Detect personal information with DEBUG output
#     """

#     # -----------------------------
#     # STRING INPUT
#     # -----------------------------
#     if isinstance(data, str):

#         # Video
#         if data.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
#             return detect_personal_info_video(data)

#         # Image / URL
#         text, qr_payloads = extract_text_and_qr_from_file(data)

#         detected, reasons = isPersonalDetails_debug(text)
#         if detected:
#             print("\n🚨 PII DETECTED (IMAGE/TEXT)")
#             print("TEXT:", repr(text))
#             print("REASONS:", reasons)
#             return True

#         for qr_text in qr_payloads:
#             detected, reasons = isPersonalDetails_debug(qr_text)
#             if detected:
#                 print("\n🚨 PII DETECTED (QR)")
#                 print("QR TEXT:", repr(qr_text))
#                 print("REASONS:", reasons)
#                 return True

#         return False

#     # -----------------------------
#     # DICT INPUT
#     # -----------------------------
#     elif isinstance(data, dict):
#         combined_text = data.get("text", "")

#         if "file" in data:
#             file_path = data["file"]

#             # Video
#             if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")):
#                 return detect_personal_info_video(file_path)

#             # Image
#             text, qr_payloads = extract_text_and_qr_from_file(file_path)
#             combined_text += " " + text

#             for qr_text in qr_payloads:
#                 detected, reasons = isPersonalDetails_debug(qr_text)
#                 if detected:
#                     print("\n🚨 PII DETECTED (DICT QR)")
#                     print("QR TEXT:", repr(qr_text))
#                     print("REASONS:", reasons)
#                     return True

#         detected, reasons = isPersonalDetails_debug(combined_text)
#         if detected:
#             print("\n🚨 PII DETECTED (DICT TEXT)")
#             print("TEXT:", repr(combined_text))
#             print("REASONS:", reasons)
#             return True

#         return False

#     return False



import re
import spacy
import platform
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode as qr_decode

# =========================================================
# Load NLP model (ONCE)
# =========================================================
nlp = spacy.load("en_core_web_sm")

# =========================================================
# Regex patterns
# =========================================================
email_pattern = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)

url_pattern = re.compile(
    r'(https?:\/\/[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?)'
)

phone_pattern = re.compile(
    r'\+?\d[\d\s\-()]{7,}\d'
)

PLATFORM_DOMAIN = "myvault-web.codextechnolife.com"

number_words = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion"
}

# =========================================================
# Tesseract path
# =========================================================
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# =========================================================
# Helper checks (UNCHANGED)
# =========================================================
def isEmail(text: str) -> bool:
    return bool(email_pattern.search(text))


def hasPhoneNumber(text: str) -> bool:
    return bool(phone_pattern.search(text))


def hasNumber(n) -> bool:
    if isinstance(n, int):
        return True
    if isinstance(n, str):
        if PLATFORM_DOMAIN in n:
            return False
        return any(ch.isdigit() for ch in n)
    return False


def hasNumberWords(text: str) -> bool:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return any(word in number_words for word in words)


def hasForbiddenURL(text: str) -> bool:
    for match in url_pattern.finditer(text):
        url = match.group(0).rstrip('.,!?')
        if not re.search(r'\.[a-zA-Z]{2,}', url):
            continue
        if PLATFORM_DOMAIN not in url:
            return True
    return False


def hasAddress(text: str) -> bool:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC", "FAC"}:
            return True
    return False


# =========================================================
# DEBUG: WHY PII DETECTED
# =========================================================
def detect_pii_reasons(text: str) -> dict:
    return {
        "forbidden_url": hasForbiddenURL(text),
        "email": isEmail(text),
        "phone": hasPhoneNumber(text),
        "number_words": hasNumberWords(text),
        "digit_found": hasNumber(text),
        "address_entity": hasAddress(text)
    }


def isPersonalDetails_debug(text: str):
    reasons = detect_pii_reasons(text)
    detected = any(reasons.values())
    return detected, reasons


# =========================================================
# Core decision logic (UNCHANGED)
# =========================================================
def isPersonalDetails(text: str) -> bool:
    return any([
        hasForbiddenURL(text),
        isEmail(text),
        hasPhoneNumber(text),
        hasNumberWords(text),
        hasNumber(text),
        hasAddress(text)
    ])


# =========================================================
# QR helpers
# =========================================================
def extract_qr_from_frame(frame) -> list[str]:
    try:
        decoded = qr_decode(frame)
        return [
            obj.data.decode("utf-8", errors="ignore")
            for obj in decoded
        ]
    except Exception:
        return []


# =========================================================
# OCR + QR (Image / URL)
# =========================================================
def extract_text_and_qr_from_file(file_path_or_url: str):
    text = ""
    qr_payloads = []

    try:
        if file_path_or_url.startswith("http"):
            response = requests.get(file_path_or_url, timeout=10)
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(file_path_or_url)

        text = pytesseract.image_to_string(img)

        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        qr_payloads = extract_qr_from_frame(frame)

    except Exception as e:
        print(f"OCR/QR error: {e}")

    return text, qr_payloads


# =========================================================
# OCR + QR (Video)
# =========================================================
def detect_personal_info_video(video_path, frame_skip=30) -> bool:
    if not os.path.exists(video_path):
        return False

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_skip == 0:
            text = pytesseract.image_to_string(Image.fromarray(frame))
            qr_payloads = extract_qr_from_frame(frame)

            if text:
                detected, reasons = isPersonalDetails_debug(text)
                if detected:
                    print("\n🚨 PII DETECTED (VIDEO OCR)")
                    print("FRAME:", frame_id)
                    print("TEXT:", repr(text))
                    print("REASONS:", reasons)
                    cap.release()
                    return True

            for qr_text in qr_payloads:
                detected, reasons = isPersonalDetails_debug(qr_text)
                if detected:
                    print("\n🚨 PII DETECTED (VIDEO QR)")
                    print("FRAME:", frame_id)
                    print("QR TEXT:", repr(qr_text))
                    print("REASONS:", reasons)
                    cap.release()
                    return True

        frame_id += 1

    cap.release()
    return False


# =========================================================
# PUBLIC API
# =========================================================
def detect_personal_info(data) -> bool:

    # -----------------------------
    # STRING INPUT
    # -----------------------------
    if isinstance(data, str):

        if data.lower().endswith((
            ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"
        )):
            return detect_personal_info_video(data)

        text, qr_payloads = extract_text_and_qr_from_file(data)

        detected, reasons = isPersonalDetails_debug(text)
        if detected:
            print("\n🚨 PII DETECTED (IMAGE / TEXT)")
            print("TEXT:", repr(text))
            print("REASONS:", reasons)
            return True

        for qr_text in qr_payloads:
            detected, reasons = isPersonalDetails_debug(qr_text)
            if detected:
                print("\n🚨 PII DETECTED (QR)")
                print("QR TEXT:", repr(qr_text))
                print("REASONS:", reasons)
                return True

        return False

    # -----------------------------
    # DICT INPUT
    # -----------------------------
    elif isinstance(data, dict):
        text_to_check = data.get("text", "")

        if "file" in data:
            file_path = data["file"]

            if file_path.lower().endswith((
                ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"
            )):
                return detect_personal_info_video(file_path)

            text, qr_payloads = extract_text_and_qr_from_file(file_path)
            text_to_check += " " + text

            for qr_text in qr_payloads:
                detected, reasons = isPersonalDetails_debug(qr_text)
                if detected:
                    print("\n🚨 PII DETECTED (DICT QR)")
                    print("QR TEXT:", repr(qr_text))
                    print("REASONS:", reasons)
                    return True

        detected, reasons = isPersonalDetails_debug(text_to_check)
        if detected:
            print("\n🚨 PII DETECTED (DICT TEXT)")
            print("TEXT:", repr(text_to_check))
            print("REASONS:", reasons)
            return True

        return False

    return False

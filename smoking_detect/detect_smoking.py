from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO
import time
import cv2
import os
# Load model and processor
model_name = "dima806/smoker_image_classification"

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Set device (GPU if available)
device = torch.device("cuda" if (1<0) else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

def predict_smoking(image_path_or_url):
    # Load image from local path or URL
    if isinstance(image_path_or_url, str) and image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
        confidence = probabilities[0][predicted_class_idx].item()
    
    result = {
        "label": predicted_label,           # e.g., "smoking" or "notsmoking"
        "confidence": round(confidence * 100, 2),
        "is_smoking": predicted_label.lower() == "smoking"
    }
    
    return result

# video detection function

def detect_smoking_in_video(
    video_path,
    frame_skip=15,          # process every Nth frame
    threshold_low=0.70,    # 70%
    threshold_high=0.90,   # 90%
    min_frames=5           # number of smoking frames needed
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error opening video file")

    smoking_frame_count = 0
    frame_index = 0

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_index % frame_skip != 0:
            frame_index += 1
            continue

        # Convert OpenCV BGR → RGB → PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            pred_idx = logits.argmax(-1).item()
            label = model.config.id2label[pred_idx].lower()
            confidence = probs[0][pred_idx].item()

        # Debug (optional)
        print(f"Frame {frame_index}: {label} ({confidence*100:.2f}%)")

        # 🔴 High confidence early stop
        if label == "smoking" and confidence >= threshold_high:
            cap.release()
            print("🔥 Early stop: High confidence smoking detected")
            return True

        # 🟡 Count moderate confidence frames
        if label == "smoking" and confidence >= threshold_low:
            smoking_frame_count += 1

        # 🔴 If enough frames detected
        if smoking_frame_count >= min_frames:
            cap.release()
            print("🔥 Smoking detected after multiple frames")
            return True

        frame_index += 1

    cap.release()
    end_time = time.time()

    print(f"No smoking detected | Time: {end_time - start_time:.2f}s")
    return False



def is_smoking_detected(media_path):
    """
    Detects smoking in an image or video.
    Returns only True or False.
    """

    # Supported formats
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

    # Handle URLs (basic check)
    if isinstance(media_path, str) and media_path.startswith(('http://', 'https://')):
        # Assume image URL (since video streaming is not handled here)
        result = predict_smoking(media_path)
        return result["is_smoking"]

    # Get file extension
    _, ext = os.path.splitext(media_path.lower())

    # Image case
    if ext in image_exts:
        result = predict_smoking(media_path)
        return result["is_smoking"]

    # Video case
    elif ext in video_exts:
        return detect_smoking_in_video(media_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
    

"""
ai_engine.py
------------
Drop this file into the root of pasumai-kural-ai (same folder as app.py).
Also copy the `model/` folder (with sensor_model.pkl + image_model.pkl) alongside it.

Exposes two functions your Flask routes call:

    predict_from_sensors(soil_moisture, temperature, humidity, light) -> dict
    predict_from_image(image_bytes) -> dict

Both return:
    {
        "status": "Healthy" | "Thirsty" | "Pest Risk" | ... ,
        "confidence": 0.0-1.0,
        "tamil_message": "..."  # ready to feed straight into gTTS(lang='ta')
    }

No TensorFlow/PyTorch here on purpose — keeps the Vercel serverless bundle small.
"""
import os
import io
import joblib
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENSOR_MODEL_PATH = os.path.join(BASE_DIR, "model", "sensor_model.pkl")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "model", "image_model.pkl")

_sensor_bundle = None
_image_bundle = None


def _load_sensor_model():
    global _sensor_bundle
    if _sensor_bundle is None:
        _sensor_bundle = joblib.load(SENSOR_MODEL_PATH)
    return _sensor_bundle


def _load_image_model():
    global _image_bundle
    if _image_bundle is None:
        _image_bundle = joblib.load(IMAGE_MODEL_PATH)
    return _image_bundle


# ---------- Tamil voice-alert messages per status ----------
TAMIL_MESSAGES = {
    "Healthy": "உங்கள் செடி ஆரோக்கியமாக உள்ளது. நல்ல பராமரிப்பு தொடரவும்.",
    "Thirsty": "செடிக்கு தண்ணீர் தேவை. உடனே நீர் பாய்ச்சவும்.",
    "Pest Risk": "பூச்சி தாக்குதல் ஏற்படும் வாய்ப்பு உள்ளது. கவனமாக பரிசோதிக்கவும்.",
    "Nutrient Deficient": "செடிக்கு சத்துக்கள் குறைவாக உள்ளது. உரம் இடவும்.",
    "Disease Spots": "இலைகளில் நோய் தாக்கும் அறிகுறி உள்ளது. மருந்து தெளிக்கவும்.",
    "Pest Damage": "பூச்சி தாக்குதலால் இலை பாதிக்கப்பட்டுள்ளது. உடனடி நடவடிக்கை எடுக்கவும்.",
}


def predict_from_sensors(soil_moisture: float, temperature: float,
                          humidity: float, light: float) -> dict:
    """
    soil_moisture: 0-100 (%)
    temperature: degrees C
    humidity: 0-100 (%)
    light: 0.0-1.0 (normalized lux, e.g. lux_reading / max_lux)
    """
    bundle = _load_sensor_model()
    model = bundle["model"]

    X = np.array([[soil_moisture, temperature, humidity, light]])
    pred = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return {
        "status": pred,
        "confidence": round(confidence, 3),
        "tamil_message": TAMIL_MESSAGES.get(pred, ""),
    }


def _extract_image_features(image_bytes: bytes) -> np.ndarray:
    """Turns a leaf photo into 4 simple color-based features."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))  # keep it fast + memory-light on serverless
    arr = np.asarray(img).astype(np.float32) / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # green_ratio: how "green/leafy" the photo is
    green_mask = (g > r) & (g > b)
    green_ratio = float(np.mean(green_mask))

    # yellow_brown_ratio: yellowing / browning leaves (nutrient issues, aging)
    yellow_brown_mask = (r > 0.5) & (g > 0.35) & (b < 0.4) & (~green_mask)
    yellow_brown_ratio = float(np.mean(yellow_brown_mask))

    # dark_spot_ratio: dark lesions/spots (disease/pest damage)
    brightness_per_px = arr.mean(axis=-1)
    dark_spot_mask = brightness_per_px < 0.25
    dark_spot_ratio = float(np.mean(dark_spot_mask))

    brightness = float(np.mean(brightness_per_px))

    return np.array([[green_ratio, yellow_brown_ratio, dark_spot_ratio, brightness]])


def predict_from_image(image_bytes: bytes) -> dict:
    bundle = _load_image_model()
    model = bundle["model"]

    X = _extract_image_features(image_bytes)
    pred = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))

    return {
        "status": pred,
        "confidence": round(confidence, 3),
        "tamil_message": TAMIL_MESSAGES.get(pred, ""),
    }

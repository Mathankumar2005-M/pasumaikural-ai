"""
ai_routes.py
------------
Flask Blueprint that plugs the AI engine into your existing app.py with
minimal changes.

In your app.py, add these two lines near the top:

    from ai_routes import ai_bp
    app.register_blueprint(ai_bp)

That's it — you now have two new endpoints:

    POST /api/predict-sensor
        JSON body: {"soil_moisture": 32, "temperature": 29, "humidity": 60, "light": 0.5}
        -> {"status": "...", "confidence": 0.83, "tamil_message": "...",
            "audio_b64": "<base64 mp3, or null if TTS failed>"}

    POST /api/predict-image
        multipart/form-data, field name "image" = the leaf photo file
        -> same shape as above

Wherever your current code does `random.choice([...])` to fake the plant
status, replace that call with a request to one of these endpoints instead
(or call ai_engine.predict_from_sensors / predict_from_image directly if
you prefer to keep it server-side only, no extra HTTP hop).
"""
import io
import base64
from flask import Blueprint, request, jsonify, current_app
from gtts import gTTS

from ai_engine import predict_from_sensors, predict_from_image

ai_bp = Blueprint("ai_bp", __name__)


def _make_tamil_audio_b64(text: str):
    """
    Generates a Tamil mp3 and returns it as a base64 string (no disk writes).
    Vercel's serverless filesystem is read-only except /tmp, and /tmp isn't
    guaranteed to persist or be reachable across invocations/instances, so we
    return audio inline in the JSON response instead of saving a file.

    Frontend usage:
        const audio = new Audio(`data:audio/mp3;base64,${result.audio_b64}`);
        audio.play();
    """
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="ta").write_to_fp(buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        current_app.logger.warning(f"gTTS generation failed: {e}")
        return None


@ai_bp.route("/api/predict-sensor", methods=["POST"])
def predict_sensor_route():
    data = request.get_json(force=True, silent=True) or {}
    try:
        soil_moisture = float(data.get("soil_moisture"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        light = float(data.get("light"))
    except (TypeError, ValueError):
        return jsonify({"error": "soil_moisture, temperature, humidity, light (0-1) are required numbers"}), 400

    result = predict_from_sensors(soil_moisture, temperature, humidity, light)
    result["audio_b64"] = _make_tamil_audio_b64(result["tamil_message"])
    return jsonify(result)


@ai_bp.route("/api/predict-image", methods=["POST"])
def predict_image_route():
    if "image" not in request.files:
        return jsonify({"error": "multipart form field 'image' is required"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"error": "empty image file"}), 400

    result = predict_from_image(image_bytes)
    result["audio_b64"] = _make_tamil_audio_b64(result["tamil_message"])
    return jsonify(result)

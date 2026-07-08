"""
ai_chat.py
----------
Flask Blueprint that adds a real generative-AI chat assistant on top of the
existing sensor/image ML models. Farmers can type (or the frontend can send)
a plain-language question — in Tamil or English — and get back a short,
practical answer in Tamil, optionally read aloud via the same gTTS pipeline
used elsewhere in this app.

This uses the Anthropic API (Claude) via the official `anthropic` Python
SDK. It is a genuinely different kind of "AI" from ai_engine.py: that file
does classical ML (RandomForest / LogisticRegression) on numeric/color
features; this file calls a large language model to have an actual
conversation with the farmer.

SETUP (one-time):
    1. Get an API key from https://console.anthropic.com/
    2. Set it as an environment variable before running the app:

        export ANTHROPIC_API_KEY="sk-ant-..."          # mac/linux
        set ANTHROPIC_API_KEY=sk-ant-...                # windows cmd

       On Vercel: Project Settings -> Environment Variables -> add
       ANTHROPIC_API_KEY there (never commit the key to git).

    3. `pip install anthropic` (already added to requirements.txt)

New endpoint:

    POST /api/chat
        JSON body: {
            "question": "என் தக்காளி செடி இலைகள் மஞ்சளாக மாறுது, என்ன பண்ணுவேன்?",
            "context": {                       # optional, all fields optional
                "status": "Nutrient Deficient", # last prediction, if any
                "crop": "tomato"                 # what crop the farmer is growing
            }
        }
        -> {
            "answer": "...(Tamil text)...",
            "audio_b64": "<base64 mp3, or null if TTS failed>"
        }

If ANTHROPIC_API_KEY isn't set, the endpoint returns a clear 500 error
instead of silently failing, so it's obvious what to fix during setup.
"""
import io
import base64
import os

from flask import Blueprint, request, jsonify, current_app
from gtts import gTTS
from anthropic import Anthropic

chat_bp = Blueprint("chat_bp", __name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Get a key from https://console.anthropic.com/ and set it "
                "before starting the app."
            )
        _client = Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are the in-app farming assistant for "Pasumai Kural" \
(பசுமை குரல்), a plant-health app used by small farmers in Tamil Nadu, India.

Rules:
- Always reply in simple, conversational Tamil (Tamil script), never in English, \
unless the farmer's question is itself in English — then reply in simple English.
- Keep answers short: 3-5 sentences, practical, no jargon. This is read aloud by \
text-to-speech, so avoid long lists, headers, or markdown formatting.
- If the app already ran a sensor or photo analysis and passed you a "status" \
in context, ground your advice in that status (e.g. Thirsty, Pest Risk, \
Nutrient Deficient, Disease Spots, Pest Damage, Healthy).
- Give concrete next steps a smallholder farmer can act on today (e.g. how much \
water, what kind of organic/cheap remedy, when to consult a local agri officer), \
not vague generalities.
- If asked something outside farming/plant-care, politely say you can only help \
with plant and farm questions, in Tamil.
- Never invent specific pesticide brand names or dosages you're not confident \
about; when unsure, recommend consulting the local agriculture extension office \
(வேளாண் அலுவலகம்).
"""


def _make_tamil_audio_b64(text: str):
    """Same approach as ai_routes.py — inline base64 mp3, no disk writes."""
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="ta").write_to_fp(buf)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        current_app.logger.warning(f"gTTS generation failed: {e}")
        return None


def _build_user_message(question: str, context: dict) -> str:
    if not context:
        return question

    parts = []
    status = context.get("status")
    crop = context.get("crop")
    if status:
        parts.append(f"(App's last analysis result: {status})")
    if crop:
        parts.append(f"(Crop: {crop})")
    parts.append(question)
    return "\n".join(parts)


@chat_bp.route("/api/chat", methods=["POST"])
def chat_route():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    context = data.get("context") or {}

    if not question:
        return jsonify({"error": "'question' is required"}), 400

    try:
        client = _get_client()
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    user_message = _build_user_message(question, context)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception as e:
        current_app.logger.error(f"Anthropic API call failed: {e}")
        return jsonify({"error": f"AI request failed: {e}"}), 502

    return jsonify({
        "answer": answer,
        "audio_b64": _make_tamil_audio_b64(answer),
    })

# Pasumai Kural — AI Addon Integration Guide

Da, ithu real AI (scikit-learn ML models) — sensor data + leaf photo rendume
handle pannum. TensorFlow/PyTorch use pannalaye, Vercel serverless size limit
mீri poidum nu light-weight-ah vச்சிருக்கேன்.

## 1. Files to copy into your repo

```
pasumai-kural-ai/
├── app.py                     (your existing file — edit, see step 3)
├── ai_engine.py                <-- NEW, copy as-is
├── ai_routes.py                <-- NEW, copy as-is
└── model/
    ├── train_sensor_model.py   <-- NEW
    ├── train_image_model.py    <-- NEW
    ├── sensor_model.pkl        <-- NEW (generated, see step 2)
    └── image_model.pkl         <-- NEW (generated, see step 2)
```

## 2. Generate the trained models (one-time, do this locally before deploying)

```bash
pip install -r requirements-ai-addon.txt
python model/train_sensor_model.py   # -> model/sensor_model.pkl
python model/train_image_model.py    # -> model/image_model.pkl
```

Commit the two `.pkl` files to git (they're small, a few hundred KB) — Vercel's
build won't retrain them at deploy time, it just loads them.

## 3. Wire into app.py

Near your other imports, add:

```python
from ai_routes import ai_bp
app.register_blueprint(ai_bp)
```

Then wherever your current code fakes plant status with `random.choice([...])`,
replace it with a call to the AI directly (no HTTP round trip needed since
it's the same process):

```python
from ai_engine import predict_from_sensors, predict_from_image

# instead of: status = random.choice(["Healthy", "Thirsty", ...])
result = predict_from_sensors(soil_moisture, temperature, humidity, light)
status = result["status"]
tamil_text = result["tamil_message"]
```

Or just call the two new endpoints from your frontend JS:
- `POST /api/predict-sensor` — JSON body `{soil_moisture, temperature, humidity, light}`
- `POST /api/predict-image` — multipart form, field `image` = leaf photo file

Both return `{status, confidence, tamil_message, audio_b64}`. Play the Tamil
alert client-side with:

```js
const audio = new Audio(`data:audio/mp3;base64,${result.audio_b64}`);
audio.play();
```

## 4. Add a photo-upload input to your frontend

If your `templates/index.html` currently only shows simulated sensor values,
add a simple file input so the image AI is reachable:

```html
<input type="file" id="leafPhoto" accept="image/*" capture="environment">
<button onclick="analyzeLeaf()">Analyze Leaf</button>
<script>
async function analyzeLeaf() {
  const file = document.getElementById('leafPhoto').files[0];
  const fd = new FormData();
  fd.append('image', file);
  const res = await fetch('/api/predict-image', { method: 'POST', body: fd });
  const data = await res.json();
  console.log(data.status, data.confidence, data.tamil_message);
  new Audio(`data:audio/mp3;base64,${data.audio_b64}`).play();
}
</script>
```

## 5. Vercel-specific notes

- **Filesystem is read-only** except `/tmp` in serverless functions, and `/tmp`
  isn't guaranteed to persist across invocations. That's why `ai_routes.py`
  returns the Tamil TTS audio as base64 in the JSON response instead of saving
  an mp3 file to disk — no filesystem writes needed at request time.
- **Bundle size**: scikit-learn + numpy + Pillow + joblib is comfortably inside
  Vercel's function size limits (this is why we avoided TensorFlow/PyTorch —
  those alone can exceed the limit).
- **Cold starts**: `ai_engine.py` caches the loaded models in memory
  (`_sensor_bundle` / `_image_bundle` globals) so only the *first* request per
  serverless instance pays the model-loading cost.

## 6. How the "AI" actually works (so you can explain it in your project demo)

- **Sensor model**: a Random Forest classifier trained on data generated from
  agronomy rules of thumb (low soil moisture → thirsty, high humidity+temp →
  pest risk, etc.) plus noise, so the model has to learn to generalize rather
  than just look up a hardcoded rule.
- **Image model**: extracts 4 simple color features from the leaf photo
  (green ratio, yellow/brown ratio, dark-spot ratio, brightness) using
  Pillow + numpy, then a Logistic Regression classifier maps those features
  to Healthy / Nutrient Deficient / Disease Spots / Pest Damage. This
  color-based-features + classifier approach is a real, standard lightweight
  technique used in agri-tech before reaching for a full CNN.

## 7. AI chat assistant (Claude API) — new in this version

Beyond the two ML models, the app now has a real generative-AI chat assistant
so farmers can type any plant-care question and get a Tamil answer, using
Claude (Anthropic's API).

**Files:** `ai_chat.py` (Blueprint), wired into `app.py`, UI card already
added to `templates/index.html`.

**One-time setup:**
1. Get an API key at https://console.anthropic.com/
2. Set it as an environment variable before running the app:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
   On Vercel: Project Settings → Environment Variables → add
   `ANTHROPIC_API_KEY` there. **Never commit the key to git.**
3. `anthropic` is already in `requirements.txt`, so `pip install -r
   requirements.txt` covers it.

**New endpoint:**
```
POST /api/chat
JSON body: {"question": "...", "context": {"status": "Nutrient Deficient"}}
-> {"answer": "...(Tamil text)...", "audio_b64": "<base64 mp3 or null>"}
```

The frontend automatically sends the last sensor/photo prediction's `status`
as context, so the assistant's advice is grounded in what the app already
detected — e.g. if the last leaf-photo result was "Pest Damage", ask "என்ன
பண்ணுவேன்?" and it'll answer about pest control specifically, not generic
advice.

If `ANTHROPIC_API_KEY` isn't set, the endpoint returns a clear 500 error
telling you exactly what's missing, instead of failing silently.

## 8. Upgrading later with real data

Both training scripts are self-contained and swappable. Once you log real
sensor readings with confirmed outcomes, or collect labelled leaf photos,
just replace `build_dataset()` in the relevant training script with your real
data (CSV for sensors, or image folder + `_extract_image_features` for
photos) and retrain — `ai_engine.py` doesn't need to change at all.

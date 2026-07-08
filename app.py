"""
app.py
------
Pasumai Kural — main Flask app.

Serves the dashboard (templates/index.html) and wires in two AI blueprints:

    ai_routes.py  ->  POST /api/predict-sensor   (JSON sensor readings -> status + Tamil audio)
                      POST /api/predict-image    (leaf photo upload -> status + Tamil audio)
                      (classical ML: RandomForest / LogisticRegression, scikit-learn)

    ai_chat.py    ->  POST /api/chat             (farmer's question -> Tamil AI answer + audio)
                      (generative AI: Claude API, needs ANTHROPIC_API_KEY set — see ai_chat.py)

Run locally:
    pip install -r requirements.txt
    python model/train_sensor_model.py     # one-time, generates model/sensor_model.pkl
    python model/train_image_model.py      # one-time, generates model/image_model.pkl
    export ANTHROPIC_API_KEY="sk-ant-..."  # needed only for the /api/chat assistant
    python app.py
    -> open http://127.0.0.1:5000
"""
from flask import Flask, render_template

from ai_routes import ai_bp
from ai_chat import chat_bp

app = Flask(__name__)
app.register_blueprint(ai_bp)
app.register_blueprint(chat_bp)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # debug=True is fine for local dev; Vercel/production should not use this.
    app.run(debug=True)

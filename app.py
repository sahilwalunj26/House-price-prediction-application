
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

# Load the trained model once when the serverless instance starts.
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Haven | House Price Predictor</title>
<style>
:root {
    --primary: #6558e8;
    --primary-dark: #5043ce;
    --ink: #20243a;
    --muted: #777f93;
    --border: #e7e9f2;
    --surface: #ffffff;
    --background: #f5f6fc;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 sans-serif;
    background: var(--background);
    color: var(--ink);
}
header {
    background: rgba(255,255,255,.92);
    border-bottom: 1px solid var(--border);
    padding: 17px 6%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.5px;
}
.logo {
    display: grid;
    place-items: center;
    width: 42px;
    height: 42px;
    border-radius: 13px;
    color: white;
    background: linear-gradient(135deg, #8174ff, #5043ce);
    font-size: 23px;
}
.pill {
    font-size: 12px;
    font-weight: 700;
    color: #5144cb;
    background: #eeecff;
    padding: 9px 13px;
    border-radius: 30px;
}
main {
    width: min(1080px, 92%);
    margin: 45px auto;
}
.hero { text-align: center; margin-bottom: 30px; }
.eyebrow {
    color: var(--primary);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 800;
    font-size: 11px;
}
h1 {
    font-size: clamp(30px, 5vw, 46px);
    letter-spacing: -1.7px;
    margin: 12px 0;
}
.hero p {
    color: var(--muted);
    max-width: 570px;
    margin: 0 auto;
    line-height: 1.7;
}
.layout {
    display: grid;
    grid-template-columns: minmax(0, 1.5fr) minmax(260px, .8fr);
    gap: 22px;
    align-items: start;
}
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 27px;
    box-shadow: 0 12px 38px rgba(35, 39, 75, .045);
}
.card h2 { font-size: 19px; margin: 0 0 7px; }
.subheading {
    color: var(--muted);
    font-size: 13px;
    margin: 0 0 24px;
    line-height: 1.6;
}
.fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 19px 16px;
}
.field label {
    display: block;
    font-size: 12px;
    font-weight: 750;
    margin-bottom: 9px;
}
.field input, .field select {
    width: 100%;
    padding: 13px 12px;
    border: 1px solid #dfe2ed;
    border-radius: 11px;
    background: #fff;
    font-size: 14px;
    color: var(--ink);
    outline: none;
    transition: border .2s, box-shadow .2s;
}
.field input:focus, .field select:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px #6558e81c;
}
.hint { color: var(--muted); font-size: 11px; margin-top: 7px; }
.submit {
    width: 100%;
    margin-top: 25px;
    padding: 15px;
    border: 0;
    border-radius: 12px;
    background: var(--primary);
    color: white;
    font-size: 14px;
    font-weight: 800;
    cursor: pointer;
    transition: background .2s, transform .2s;
}
.submit:hover { background: var(--primary-dark); transform: translateY(-1px); }
.error {
    margin-top: 18px;
    padding: 13px;
    border-radius: 10px;
    color: #a32635;
    background: #fff0f1;
    font-size: 13px;
}
.result {
    margin-top: 20px;
    border: 1px solid #dcd8ff;
    border-radius: 15px;
    padding: 20px;
    background: linear-gradient(135deg, #f5f3ff, #fff);
}
.result-label {
    color: #6356d9;
    font-size: 12px;
    font-weight: 750;
}
.price {
    font-size: clamp(27px, 4vw, 36px);
    font-weight: 850;
    letter-spacing: -1px;
    margin: 10px 0;
    overflow-wrap: anywhere;
}
.result small { color: var(--muted); line-height: 1.6; }
.side-card {
    color: white;
    border-radius: 22px;
    padding: 25px;
    background: linear-gradient(145deg, #29264c, #5146a7);
    box-shadow: 0 15px 40px #29264c1a;
}
.side-icon {
    width: 48px; height: 48px; border-radius: 15px;
    background: #ffffff20; display: grid; place-items: center;
    font-size: 25px; margin-bottom: 22px;
}
.side-card h2 { font-size: 21px; line-height: 1.4; margin: 0 0 12px; }
.side-card p { color: #e0def8; font-size: 13px; line-height: 1.8; }
.side-divider { height: 1px; background: #ffffff30; margin: 22px 0; }
.feature { display: flex; gap: 10px; margin: 13px 0; font-size: 13px; color: #f1efff; }
.note {
    margin-top: 20px; font-size: 11px; color: var(--muted);
    line-height: 1.7; text-align: center;
}
footer { text-align: center; padding: 12px 20px 30px; color: var(--muted); font-size: 12px; }
@media (max-width: 760px) {
    main { margin: 30px auto; }
    .layout { grid-template-columns: 1fr; }
    .side-card { order: 2; }
    .card { padding: 21px; }
}
@media (max-width: 430px) {
    .fields { grid-template-columns: 1fr; }
    header { padding: 14px 4%; }
    .pill { font-size: 10px; }
}
</style>
</head>
<body>
<header>
    <div class="brand"><div class="logo">⌂</div> Haven<span style="color:#6558e8">.</span></div>
    <div class="pill">✦ AI PRICE ESTIMATOR</div>
</header>

<main>
    <section class="hero">
        <div class="eyebrow">Smart property insights</div>
        <h1>Find your home's estimated value.</h1>
        <p>Enter the property details below. Our trained machine learning
        model will estimate a price based on the information provided.</p>
    </section>

    <div class="layout">
        <section class="card">
            <h2>Property details</h2>
            <p class="subheading">Complete all seven fields for your estimate.</p>

            <form method="POST" action="/">
                <div class="fields">
                    <div class="field">
                        <label for="square_footage">Square footage</label>
                        <input id="square_footage" name="Square_Footage"
                               type="number" min="1" max="1000000" step="any"
                               placeholder="e.g. 1800" required
                               value="{{ form.get('Square_Footage', '') }}">
                        <div class="hint">Total property area</div>
                    </div>

                    <div class="field">
                        <label for="bedrooms">Number of bedrooms</label>
                        <input id="bedrooms" name="Num_Bedrooms" type="number"
                               min="0" max="100" step="1" placeholder="e.g. 3"
                               required value="{{ form.get('Num_Bedrooms', '') }}">
                    </div>

                    <div class="field">
                        <label for="bathrooms">Number of bathrooms</label>
                        <input id="bathrooms" name="Num_Bathrooms" type="number"
                               min="0" max="100" step="any" placeholder="e.g. 2"
                               required value="{{ form.get('Num_Bathrooms', '') }}">
                    </div>

                    <div class="field">
                        <label for="year">Year built</label>
                        <input id="year" name="Year_Built" type="number"
                               min="1800" max="2100" step="1"
                               placeholder="e.g. 2015" required
                               value="{{ form.get('Year_Built', '') }}">
                    </div>

                    <div class="field">
                        <label for="lot">Lot size</label>
                        <input id="lot" name="Lot_Size" type="number"
                               min="0" max="100000000" step="any"
                               placeholder="e.g. 4500" required
                               value="{{ form.get('Lot_Size', '') }}">
                        <div class="hint">Use the same unit as your training data</div>
                    </div>

                    <div class="field">
                        <label for="garage">Garage size</label>
                        <input id="garage" name="Garage_Size" type="number"
                               min="0" max="100" step="any"
                               placeholder="e.g. 2" required
                               value="{{ form.get('Garage_Size', '') }}">
                        <div class="hint">Use the same unit as your training data</div>
                    </div>

                    <div class="field">
                        <label for="quality">Neighborhood quality</label>
                        <select id="quality" name="Neighborhood_Quality" required>
                            {% for q in range(1, 11) %}
                            <option value="{{ q }}"
                              {% if form.get('Neighborhood_Quality', '5')|string == q|string %}selected{% endif %}>
                              {{ q }} / 10{% if q == 5 %} — Average{% elif q == 10 %} — Excellent{% elif q == 1 %} — Low{% endif %}
                            </option>
                            {% endfor %}
                        </select>
                        <div class="hint">Select a rating from 1 to 10</div>
                    </div>
                </div>

                <button class="submit" type="submit">Calculate estimated price &nbsp; →</button>
            </form>

            {% if error %}
            <div class="error" role="alert">{{ error }}</div>
            {% endif %}

            {% if prediction is not none %}
            <div class="result" aria-live="polite">
                <div class="result-label">YOUR MODEL'S ESTIMATED PRICE</div>
                <div class="price">{{ prediction }}</div>
                <small>This is a model estimate, not a guaranteed market price.
                Currency and units depend on the original training dataset.</small>
            </div>
            {% endif %}
        </section>

        <aside>
            <div class="side-card">
                <div class="side-icon">⌂</div>
                <h2>A clearer view of your next property decision.</h2>
                <p>Get an instant estimate from your trained Linear Regression
                model using the property features you provide.</p>
                <div class="side-divider"></div>
                <div class="feature"><span>✓</span> Seven property features</div>
                <div class="feature"><span>✓</span> Instant model inference</div>
                <div class="feature"><span>✓</span> Responsive mobile-friendly UI</div>
                <div class="feature"><span>✓</span> Powered by machine learning</div>
            </div>
            <p class="note">For educational and estimation purposes only.
            Actual property values can differ significantly.</p>
        </aside>
    </div>
</main>
<footer>Haven Property Insights · Built with Flask and Machine Learning</footer>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    form = request.form.to_dict() if request.method == "POST" else {}
    prediction = None
    error = None

    if request.method == "POST":
        try:
            values = []

            for feature in FEATURES:
                raw_value = request.form.get(feature, "").strip()
                if not raw_value:
                    raise ValueError(f"Please enter a value for {feature.replace('_', ' ')}.")

                value = float(raw_value)
                if not np.isfinite(value):
                    raise ValueError("Please enter valid numeric values.")

                values.append(value)

            # Basic validation for property fields.
            if values[0] <= 0:
                raise ValueError("Square footage must be greater than zero.")
            if values[1] < 0 or not values[1].is_integer():
                raise ValueError("Bedrooms must be a non-negative whole number.")
            if values[2] < 0:
                raise ValueError("Bathrooms cannot be negative.")
            if not values[3].is_integer() or not 1800 <= values[3] <= 2100:
                raise ValueError("Please enter a valid year built.")
            if values[4] < 0 or values[5] < 0:
                raise ValueError("Lot size and garage size cannot be negative.")
            if not values[6].is_integer() or not 1 <= values[6] <= 10:
                raise ValueError("Neighborhood quality must be a whole number from 1 to 10.")

            # Preserve the exact feature order used during model training.
            input_df = pd.DataFrame([values], columns=FEATURES)
            predicted_value = float(model.predict(input_df)[0])

            if not np.isfinite(predicted_value):
                raise ValueError("The model returned an invalid estimate.")

            prediction = f"{predicted_value:,.2f}"

        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check that the model file and its "
                "scikit-learn dependencies match the training environment."
            )

    return render_template_string(
        HTML,
        form=form,
        prediction=prediction,
        error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model": "loaded"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

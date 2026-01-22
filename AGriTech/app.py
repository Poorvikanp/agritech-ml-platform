from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask import render_template


print("Starting Flask app...")

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("AGriTech/crop_model.pkl")
scaler = joblib.load("AGriTech/scaler.pkl")

REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    data = request.get_json()

    # 1️⃣ Check JSON presence
    if data is None:
        return jsonify({"error": "No JSON data received"}), 400

    # 2️⃣ Required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # 3️⃣ Type validation
    try:
        values = [
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]
    except (ValueError, TypeError):
        return jsonify({"error": "All input values must be numeric"}), 400

    try:
        input_data = np.array([values])
        input_scaled = scaler.transform(input_data)

        # Predict class
        prediction = model.predict(input_scaled)[0]

        # Predict probabilities
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100

        return jsonify({
            "recommended_crop": str(prediction),
            "confidence": f"{confidence:.2f}%",
            "input_received": {
                "N": values[0],
                "P": values[1],
                "K": values[2],
                "temperature": values[3],
                "humidity": values[4],
                "ph": values[5],
                "rainfall": values[6]
            }
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500



if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import joblib
import numpy as np

print("Starting Flask app...")

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("AGriTech/crop_model.pkl")
scaler = joblib.load("AGriTech/scaler.pkl")

REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@app.route("/")
def home():
    return "AgriTech Crop Recommendation API is running!"

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
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "recommended_crop": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from PIL import Image
import tensorflow as tf


print("Starting Flask app...")

app = Flask(__name__)
# Load disease detection model
disease_model = tf.keras.models.load_model("AGriTech/disease_model.h5")

# Load saved model and scaler
model = joblib.load("AGriTech/crop_model.pkl")
scaler = joblib.load("AGriTech/scaler.pkl")

REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DISEASE_CLASSES = [
    "Healthy",
    "Leaf Blight",
    "Rust",
    "Powdery Mildew"
]


# -------------------------------
# Soil Health Analysis
# -------------------------------
def analyze_soil_health(N, P, K, ph):
    advice = {}

    # Nitrogen analysis
    if N < 50:
        advice["nitrogen"] = "Low — add urea"
        n_status = "Low"
    elif 50 <= N <= 100:
        advice["nitrogen"] = "Optimal"
        n_status = "Optimal"
    else:
        advice["nitrogen"] = "High — reduce fertilizer"
        n_status = "High"

    # Phosphorus analysis
    if P < 40:
        advice["phosphorus"] = "Low — add DAP"
        p_status = "Low"
    elif 40 <= P <= 80:
        advice["phosphorus"] = "Optimal"
        p_status = "Optimal"
    else:
        advice["phosphorus"] = "High — reduce fertilizer"
        p_status = "High"

    # Potassium analysis
    if K < 40:
        advice["potassium"] = "Low — add potash"
        k_status = "Low"
    elif 40 <= K <= 80:
        advice["potassium"] = "Optimal"
        k_status = "Optimal"
    else:
        advice["potassium"] = "High — reduce fertilizer"
        k_status = "High"

    # pH analysis
    if ph < 5.5:
        advice["ph"] = "Acidic — add lime"
        ph_status = "Low"
    elif 5.5 <= ph <= 7.5:
        advice["ph"] = "Neutral"
        ph_status = "Optimal"
    else:
        advice["ph"] = "Alkaline — add organic matter"
        ph_status = "High"

    # Overall soil health
    optimal_count = [n_status, p_status, k_status, ph_status].count("Optimal")

    if optimal_count >= 3:
        soil_health = "Good"
    elif optimal_count == 2:
        soil_health = "Moderate"
    else:
        soil_health = "Poor"

    return soil_health, advice


# -------------------------------
# Weather Risk Analysis
# -------------------------------
def analyze_weather_risks(temperature, humidity, rainfall):
    alerts = []

    # Rainfall risks
    if rainfall > 250:
        alerts.append("High rainfall — risk of flooding")
    elif rainfall < 50:
        alerts.append("Low rainfall — drought risk")

    # Temperature risks
    if temperature > 35:
        alerts.append("High temperature — heat stress risk")
    elif temperature < 10:
        alerts.append("Low temperature — cold stress risk")

    # Humidity risks
    if humidity > 85:
        alerts.append("High humidity — fungal disease risk")
    elif humidity < 30:
        alerts.append("Low humidity — dry air stress")

    if not alerts:
        alerts.append("Weather conditions are favorable")

    return alerts
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# -------------------------------
# Routes
# -------------------------------
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

        # Soil health analysis
        soil_health, nutrient_advice = analyze_soil_health(
            values[0], values[1], values[2], values[5]
        )

        # Weather risk analysis
        weather_alerts = analyze_weather_risks(
            values[3], values[4], values[6]
        )

        return jsonify({
            "recommended_crop": str(prediction),
            "confidence": f"{confidence:.2f}%",
            "soil_health": soil_health,
            "nutrient_advice": nutrient_advice,
            "weather_alerts": weather_alerts,
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
@app.route("/detect-disease", methods=["POST"])
def detect_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        processed_image = preprocess_image(image)

        predictions = disease_model.predict(processed_image)[0]
        class_index = np.argmax(predictions)
        confidence = predictions[class_index] * 100

        disease_name = DISEASE_CLASSES[class_index]

        treatment_advice = get_treatment_advice(disease_name)

        return jsonify({
            "disease": disease_name,
            "confidence": f"{confidence:.2f}%",
            "treatment_advice": treatment_advice
        })

    except Exception as e:
        return jsonify({
            "error": "Disease detection failed",
            "details": str(e)
        }), 500
def get_treatment_advice(disease):
    advice_map = {
        "Healthy": "No treatment needed. Keep monitoring the crop.",
        "Leaf Blight": "Use copper-based fungicides and remove infected leaves.",
        "Rust": "Apply sulfur fungicide and improve air circulation.",
        "Powdery Mildew": "Use neem oil or potassium bicarbonate spray."
    }

    return advice_map.get(disease, "Consult an agriculture expert.")



if __name__ == "__main__":
    app.run(debug=True)

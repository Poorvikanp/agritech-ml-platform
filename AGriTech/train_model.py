import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Decision Tree Model
# -------------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)     # Train the model
dt_pred = dt.predict(X_test_scaled) # Predict on test data
dt_acc = accuracy_score(y_test, dt_pred)

# -------------------------------
# Random Forest Model
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)



import joblib
# Save the trained model
joblib.dump(rf, "AGriTech/crop_model.pkl")
# Save the scaler
joblib.dump(scaler, "AGriTech/scaler.pkl")
print("Model and scaler saved successfully!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# Separate features and target
X = df.drop('label', axis=1)
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete!")
print(X_train_scaled.shape, X_test_scaled.shape)

# Why did you use StandardScaler?
# “Because my features had different ranges, 
# I applied StandardScaler to normalize them so that no feature dominates the
# learning process and the model performs more reliably.”

# When You SHOULD Use StandardScaler

# Use when:
# ✔ Features have different units
# ✔ Using distance-based or tree-based ML
# ✔ Want faster convergence
# ✔ Comparing multiple models

# When You DON’T Need It

# Not required when:

# Using categorical features only

# Using tree-based models sometimes (but still good practice)

# Data already normalized
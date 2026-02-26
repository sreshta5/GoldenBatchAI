import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv("data/golden_batches.csv")

# Convert columns to lowercase for consistency
df.columns = df.columns.str.lower()

# Features
X = df[["temperature", "pressure", "ph", "mixing_speed", "energy_used"]]

# Dummy label for quality (you can improve later)
y = df["cluster"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "models/golden_batch_model.pkl")

print("Golden batch model trained successfully.")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/history.csv")

# Simple rule-based labeling for training
def risk_label(score):
    if score < 10:
        return 0
    elif score < 20:
        return 1
    else:
        return 2

df["Risk_Level"] = df["severity_score"].apply(risk_label)

X = df[["temperature","pressure","ph","mixing_speed","energy_used"]]
y = df["Risk_Level"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/risk_model.pkl")

print("Risk model trained successfully")
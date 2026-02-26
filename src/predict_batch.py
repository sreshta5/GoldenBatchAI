import pandas as pd
import joblib
import numpy as np

# ===============================
# LOAD TRAINED MODEL
# ===============================

model = joblib.load("models/golden_batch_model.pkl")

# Load golden signature
signature = pd.read_csv("data/golden_signature.csv", index_col=0)

golden_mean = signature["Mean"]
golden_std = signature["Std"]

print("Golden Signature Loaded Successfully!\n")

# ===============================
# SIMULATE NEW BATCH INPUT
# (Later this will come from Streamlit)
# ===============================

new_batch = {
    "Temperature": float(input("Enter Temperature: ")),
    "Pressure": float(input("Enter Pressure: ")),
    "pH": float(input("Enter pH: ")),
    "Mixing_Speed": float(input("Enter Mixing Speed: ")),
    "Energy_Used": float(input("Enter Energy Used: "))
}

new_df = pd.DataFrame([new_batch])

# ===============================
# PREDICT
# ===============================

prediction = model.predict(new_df)[0]

if prediction == 1:
    print("\n✅ This batch matches GOLDEN standards.")
else:
    print("\n⚠ Deviation Detected!")


# ===============================
# PARAMETER CORRECTION ENGINE
# ===============================

print("\nSuggested Parameter Adjustments:")

for param in ["Temperature", "Pressure", "pH", "Mixing_Speed"]:
    diff = new_batch[param] - golden_mean[param]

    if abs(diff) > golden_std[param]:
        if diff > 0:
            print(f" - Decrease {param} by approx {round(abs(diff),2)} units")
        else:
            print(f" - Increase {param} by approx {round(abs(diff),2)} units")
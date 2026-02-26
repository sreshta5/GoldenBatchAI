import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="GoldenBatchAI",
    layout="wide"
)

st.title("🏭 GoldenBatchAI")
st.subheader("Industrial Batch Quality & Risk Intelligence Platform")

# -----------------------------------
# Problem Statement
# -----------------------------------
st.markdown("""
### 🚨 Problem Statement

Industrial batch manufacturing often suffers from:
- Parameter drift over time
- Energy inefficiency
- Quality inconsistency
- Late detection of operational risk

GoldenBatchAI provides real-time predictive monitoring,
statistical deviation analysis, and prescriptive AI recommendations
to ensure optimal batch performance.
""")

# -----------------------------------
# Define Paths
# -----------------------------------
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

GOLDEN_SIGNATURE_PATH = os.path.join(DATA_DIR, "golden_signature.csv")
BATCH_MODEL_PATH = os.path.join(MODEL_DIR, "golden_batch_model.pkl")
RISK_MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")

# -----------------------------------
# Load Golden Signature
# -----------------------------------
if not os.path.exists(GOLDEN_SIGNATURE_PATH):
    st.error("❌ golden_signature.csv not found.")
    st.stop()

golden_signature = pd.read_csv(GOLDEN_SIGNATURE_PATH)
golden_signature.columns = golden_signature.columns.str.lower()

# -----------------------------------
# Load Models
# -----------------------------------
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model not found: {path}")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

batch_model = load_model(BATCH_MODEL_PATH)
risk_model = load_model(RISK_MODEL_PATH)

st.success("✅ Models Loaded Successfully")

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("🔧 Enter Current Batch Parameters")

temperature = st.sidebar.number_input("Temperature", value=180.0)
pressure = st.sidebar.number_input("Pressure", value=30.0)
ph = st.sidebar.number_input("pH", value=7.0)
mixing_speed = st.sidebar.number_input("Mixing Speed", value=1150.0)
energy_used = st.sidebar.number_input("Energy Used", value=450.0)

# -----------------------------------
# Prediction Section
# -----------------------------------
st.header("📊 AI Prediction Results")

if st.button("Run AI Analysis"):

    input_data = pd.DataFrame({
        "temperature": [temperature],
        "pressure": [pressure],
        "ph": [ph],
        "mixing_speed": [mixing_speed],
        "energy_used": [energy_used]
    })

    try:
        # -----------------------------
        # Model Predictions
        # -----------------------------
        batch_prediction = batch_model.predict(input_data)
        risk_prediction = risk_model.predict(input_data)

        risk_map = {
            0: "Low Risk 🟢",
            1: "Moderate Risk 🟡",
            2: "High Risk 🔴"
        }

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Batch Quality Prediction", batch_prediction[0])

        with col2:
            st.metric("Risk Level Prediction", risk_map.get(risk_prediction[0], "Unknown"))

        # -----------------------------
        # Deviation Analysis
        # -----------------------------
        st.header("📉 Deviation Analysis (vs Golden Signature)")

        deviation_results = {}

        for col in ["temperature", "pressure", "ph", "mixing_speed", "energy_used"]:
            mean_val = golden_signature[col].values[0]
            std_val = golden_signature[f"{col}_std"].values[0]
            z_score = (input_data[col].values[0] - mean_val) / std_val
            deviation_results[col] = z_score

        deviation_df = pd.DataFrame.from_dict(
            deviation_results,
            orient="index",
            columns=["Z-Score"]
        )

        st.dataframe(deviation_df)

        # -----------------------------
        # Risk Highlighting
        # -----------------------------
        st.subheader("⚠ Parameters Out of Optimal Range")

        risky_params = [param for param, z in deviation_results.items() if abs(z) > 2]

        if risky_params:
            for p in risky_params:
                st.error(f"{p.upper()} is significantly deviating from optimal range.")
        else:
            st.success("All parameters within optimal range ✅")

        # -----------------------------
        # Visual Comparison Chart
        # -----------------------------
        st.header("📊 Batch vs Golden Comparison")

        comparison_df = pd.DataFrame({
            "Parameter": ["temperature", "pressure", "ph", "mixing_speed", "energy_used"],
            "Current": [
                temperature,
                pressure,
                ph,
                mixing_speed,
                energy_used
            ],
            "Golden Mean": [
                golden_signature["temperature"].values[0],
                golden_signature["pressure"].values[0],
                golden_signature["ph"].values[0],
                golden_signature["mixing_speed"].values[0],
                golden_signature["energy_used"].values[0],
            ]
        })

        st.bar_chart(comparison_df.set_index("Parameter"))

        # -----------------------------
        # AI Recommendations
        # -----------------------------
        st.header("🧠 AI Recommendations")

        for param, z in deviation_results.items():
            if abs(z) > 2:
                mean_val = golden_signature[param].values[0]
                st.warning(
                    f"Adjust {param} closer to optimal value ≈ {round(mean_val,2)}"
                )

        # -----------------------------
        # Health Score
        # -----------------------------
        st.header("🏥 Overall Batch Health Score")

        avg_deviation = np.mean([abs(v) for v in deviation_results.values()])
        health_score = max(0, 100 - (avg_deviation * 15))

        st.metric("Batch Health Score", f"{round(health_score,2)} / 100")

        if health_score > 80:
            st.success("Batch operating near optimal performance 🚀")
        elif health_score > 60:
            st.warning("Moderate deviation detected ⚠")
        else:
            st.error("High deviation detected ❌ Immediate correction recommended")

    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")

# -----------------------------------
# Golden Signature Display
# -----------------------------------
st.header("📈 Golden Signature Reference")
st.dataframe(golden_signature)

# -----------------------------------
# Business Impact Section
# -----------------------------------
st.header("💰 Business Impact")

st.markdown("""
- Reduces defective batches
- Prevents costly downtime
- Improves energy efficiency
- Enables predictive monitoring
- Supports smart manufacturing environments
""")

st.markdown("---")
st.caption("GoldenBatchAI | Industrial Optimization Platform")
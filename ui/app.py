
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk (Binary)")

# --- Paths ---
from pathlib import Path

# Resolve paths relative to THIS file (ui/app.py)
BASE_DIR = Path(__file__).resolve().parent          # .../Heart_Disease_Project/ui
PROJECT_ROOT = BASE_DIR.parent                      # .../Heart_Disease_Project
MODELS_DIR = PROJECT_ROOT / "models"

PIPELINE_PATH = MODELS_DIR / "model_pipeline.pkl"
META_PATH     = MODELS_DIR / "model_metadata.json"


# --- Load pipeline & metadata ---
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not PIPELINE_PATH.exists():
        st.error("Missing ../models/model_pipeline.pkl. Please run Step 2.7 first.")
        st.stop()
    pipe = joblib.load(PIPELINE_PATH)

    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        final_features = meta["features"]
    else:
        # Fallback: read features from pipeline preprocessor if metadata missing
        st.warning("model_metadata.json not found. Falling back to a default feature list.")
        final_features = [
            "age","sex_Male","chol","trestbps","thalch",
            "oldpeak","ca","exang",
            "cp_non-anginal","cp_atypical angina","cp_typical angina",
            "thal_normal","thal_reversable defect",
        ]
    return pipe, final_features

pipe, FINAL_FEATURES = load_artifacts()

# Handle thalach vs thalch automatically
HEARTRATE_CANDIDATES = ["thalach", "thalch"]
HR_COL = next((c for c in HEARTRATE_CANDIDATES if c in FINAL_FEATURES), "thalch")

st.caption("Note: this app uses your trained pipeline (preprocessing + model).")

# --- Input widgets ---
col1, col2 = st.columns(2)
with col1:
    age      = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=60, max_value=240, value=130)
    chol     = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=700, value=240)
    thalach  = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=230, value=150)
    oldpeak  = st.number_input("ST Depression (oldpeak)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
with col2:
    sex      = st.selectbox("Sex", ["Female", "Male"])
    exang    = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
    ca       = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, value=0)
    cp_label = st.selectbox("Chest Pain Type", ["asymptomatic", "non-anginal", "atypical angina", "typical angina"])
    thal_lbl = st.selectbox("Thal Result", ["fixed defect", "normal", "reversable defect"])

threshold = st.slider("Decision threshold (probability ≥ threshold → Disease)",
                      min_value=0.05, max_value=0.95, value=0.50, step=0.01)

# --- Build one-row DataFrame in the exact training schema ---
def build_row() -> pd.DataFrame:
    row = pd.Series(0.0, index=FINAL_FEATURES, dtype=float)

    # numeric
    row["age"]      = float(age)
    row["trestbps"] = float(trestbps)
    row["chol"]     = float(chol)
    row[HR_COL]     = float(thalach)    # 'thalch' or 'thalach' depending on training
    row["oldpeak"]  = float(oldpeak)
    row["ca"]       = float(ca)

    # binaries/one-hots
    row["sex_Male"] = 1.0 if sex == "Male" else 0.0
    row["exang"]    = 1.0 if exang == "Yes" else 0.0

    # chest pain (baseline is asymptomatic → all zeros)
    if "cp_non-anginal" in row.index:     row["cp_non-anginal"]     = 1.0 if cp_label == "non-anginal" else 0.0
    if "cp_atypical angina" in row.index: row["cp_atypical angina"] = 1.0 if cp_label == "atypical angina" else 0.0
    if "cp_typical angina" in row.index:  row["cp_typical angina"]  = 1.0 if cp_label == "typical angina" else 0.0

    # thal (baseline is fixed defect → all zeros)
    if "thal_normal" in row.index:            row["thal_normal"]            = 1.0 if thal_lbl == "normal" else 0.0
    if "thal_reversable defect" in row.index: row["thal_reversable defect"] = 1.0 if thal_lbl == "reversable defect" else 0.0

    return pd.DataFrame([row])

# --- Predict (single) ---
if st.button("Predict"):
    row_df = build_row()
    # ensure column order matches training
    row_df = row_df[FINAL_FEATURES]

    try:
        proba = float(pipe.predict_proba(row_df)[:, 1][0])
        pred  = int(proba >= threshold)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Result")
    if pred == 1:
        st.error(f"High likelihood of Heart Disease (p = {proba:.3f})")
    else:
        st.success(f"Low likelihood of Heart Disease (p = {proba:.3f})")

    with st.expander("Show model inputs"):
        st.write(row_df)

st.markdown("---")
st.subheader("Batch Predictions (CSV) — Optional")
st.caption("Upload a CSV with the same columns as the model expects (we’ll align by name).")
up = st.file_uploader("Upload CSV", type=["csv"])
if up is not None:
    try:
        df_in = pd.read_csv(up)
        # Keep only expected columns; fill missing ones with 0
        for c in FINAL_FEATURES:
            if c not in df_in.columns:
                df_in[c] = 0.0
        df_in = df_in[FINAL_FEATURES]
        probs = pipe.predict_proba(df_in)[:, 1]
        preds = (probs >= threshold).astype(int)
        out = df_in.copy()
        out["prob_disease"] = probs
        out["pred_label"]   = preds
        st.dataframe(out.head())
        st.download_button("Download predictions.csv", data=out.to_csv(index=False),
                           file_name="predictions.csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

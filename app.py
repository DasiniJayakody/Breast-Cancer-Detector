import streamlit as st
import numpy as np
import pickle

# ===================== Page Config =====================
st.set_page_config(page_title="🩺 Breast Cancer Detection", layout="wide")

# ===================== Custom CSS =====================
st.markdown(
    """
    <style>
        .stApp {
        background-color: #373738FF; 
        }
        h1, h2, h3, h4, h5, h6, h7, h8, h9, body, p {
            color: #FFFFFFFF !important;
        }
        .block-container {
            max-width: 1000px;
            margin: auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 0.6rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .reset-btn>button {
            width: 100%;
            border-radius: 10px;
            background-color: #f44336;
            color: white;
            font-size: 16px;
            padding: 0.6rem;
        }
        .reset-btn>button:hover {
            background-color: #d32f2f;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== Load Models =====================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===================== Keys & Defaults =====================
FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean"
]

for k in FEATURE_KEYS:
    if k not in st.session_state:
        st.session_state[k] = 0.0

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "prob_malignant" not in st.session_state:
    st.session_state["prob_malignant"] = None

def reset_inputs():
    for k in FEATURE_KEYS:
        st.session_state[k] = 0.0
    st.session_state["prediction"] = None
    st.session_state["prob_malignant"] = None

def is_malignant_label(lbl):
    """Robustly detect malignant label across possible encodings."""
    if isinstance(lbl, str):
        return lbl.strip().upper().startswith("M")  # 'M' or 'Malignant'
    return lbl == 1  # numeric encoding where 1 means Malignant

def malignant_probability(proba, classes):
    """Return P(Malignant) if available, else None."""
    if proba is None or classes is None:
        return None
    # classes may be numeric [0,1] or strings ['B','M']
    idx = None
    for i, c in enumerate(classes):
        if (isinstance(c, str) and c.strip().upper().startswith("M")) or (c == 1):
            idx = i
            break
    if idx is None:
        return None
    return float(proba[0][idx])

# ===================== App UI =====================
st.title("🩺 Breast Cancer Detection App")
st.markdown("This app predicts whether a tumor is **Benign (B)** or **Malignant (M)** based on input features.")
st.markdown("###  Enter Tumor Measurements")

# First row
col1, col2, col3, col4 = st.columns(4)
col1.number_input("Radius Mean",        min_value=0.0, step=0.01,   key="radius_mean")
col2.number_input("Texture Mean",       min_value=0.0, step=0.01,   key="texture_mean")
col3.number_input("Perimeter Mean",     min_value=0.0, step=0.01,   key="perimeter_mean")
col4.number_input("Area Mean",          min_value=0.0, step=0.01,   key="area_mean")

# Second row
col5, col6, col7, col8 = st.columns(4)
col5.number_input("Smoothness Mean",    min_value=0.0, step=0.01, format="%.5f", key="smoothness_mean")
col6.number_input("Compactness Mean",   min_value=0.0, step=0.01, format="%.5f", key="compactness_mean")
col7.number_input("Concavity Mean",     min_value=0.0, step=0.01, format="%.5f", key="concavity_mean")
col8.number_input("Concave Points Mean",min_value=0.0, step=0.01,  format="%.5f", key="concave_points_mean")

# ===================== Predict / Reset =====================
st.markdown("---")
col_predict, col_reset = st.columns([3, 1])

with col_predict:
    if st.button("🔍 Predict Tumor Type"):
        # Build input in the exact training order
        X = np.array([[
            st.session_state["radius_mean"],
            st.session_state["texture_mean"],
            st.session_state["perimeter_mean"],
            st.session_state["area_mean"],
            st.session_state["smoothness_mean"],
            st.session_state["compactness_mean"],
            st.session_state["concavity_mean"],
            st.session_state["concave_points_mean"]
        ]], dtype=float)

        Xs = scaler.transform(X)

        # Predict label robustly
        y_pred = model.predict(Xs)
        label = y_pred[0]
        st.session_state["prediction"] = "Malignant" if is_malignant_label(label) else "Benign"

        # Predict probability if available
        proba = model.predict_proba(Xs) if hasattr(model, "predict_proba") else None
        classes = getattr(model, "classes_", None)
        st.session_state["prob_malignant"] = malignant_probability(proba, classes)

with col_reset:
    st.button("🔄 Reset", on_click=reset_inputs)

# ===================== Result =====================
if st.session_state["prediction"] is not None:
    if st.session_state["prediction"] == "Malignant":
        st.error(f"⚠️ The tumor is predicted to be **Malignant (M)**.")
    else:
        st.success(f"✅ The tumor is predicted to be **Benign (B)**.")

    if st.session_state["prob_malignant"] is not None:
        st.caption(f"Probability Malignant: **{st.session_state['prob_malignant']:.2%}**")


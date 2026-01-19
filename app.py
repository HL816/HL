# app.py
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator

st.set_page_config(page_title="Heart Disease Risk – RF (Manual)", page_icon="❤️", layout="centered")
st.title("Heart Disease Risk – Random Forest (Manual)")
st.write("Use this page to **simulate** predictions with the trained model. Upload the exported model file (`rf_manual.pkl`) or place it next to this app.")

# ---------- Utilities ----------
FEATURES = [
    'age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
]

def load_model(uploaded_file=None, default_path="rf_manual.pkl"):
    try:
        if uploaded_file is not None:
            obj = joblib.load(uploaded_file)
        else:
            obj = joblib.load(default_path)
    except Exception as e:
        st.warning(f"Model not found or failed to load: {e}")
        return None, None

    # Support both dict payload (with 'model' and optional 'features') and bare estimator
    if isinstance(obj, dict):
        model = obj.get('model', None)
        feat = obj.get('features', FEATURES)
    else:
        model = obj
        feat = FEATURES

    # Basic validation
    if not hasattr(model, 'predict_proba'):
        st.error("Loaded object does not support predict_proba(). Ensure it is a scikit-learn classifier or Pipeline.")
        return None, None
    return model, feat

# ---------- Sidebar: Model loading ----------
st.sidebar.header("Model")
model_file = st.sidebar.file_uploader("Upload rf_manual.pkl (optional)", type=["pkl"])
model, model_features = load_model(model_file)

if model is not None:
    st.success("Model loaded successfully.")
    st.caption("Features expected by the model: " + ", ".join(model_features))
else:
    st.info("To proceed, upload `rf_manual.pkl` or place it in the same folder and refresh.")

# ---------- Input form ----------
st.header("Enter Patient Features")
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=55)
        sex_label = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        cp_label = st.selectbox("Chest pain type (cp)", [
            "Typical angina (0)", "Atypical angina (1)", "Non-anginal pain (2)", "Asymptomatic (3)"
        ])
        trestbps = st.number_input("Resting blood pressure (trestbps)", min_value=0, max_value=300, value=130)
        chol = st.number_input("Serum cholesterol (chol)", min_value=0, max_value=800, value=240)
        fbs_label = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", ["No (0)", "Yes (1)"])
        restecg_label = st.selectbox("Resting ECG (restecg)", [
            "Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"
        ])

    with col2:
        thalach = st.number_input("Max heart rate (thalach)", min_value=0, max_value=300, value=150)
        exang_label = st.selectbox("Exercise-induced angina (exang)", ["No (0)", "Yes (1)"])
        oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope_label = st.selectbox("Slope of ST segment (slope)", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        ca = st.number_input("# of major vessels (0–3) (ca)", min_value=0, max_value=4, value=0)
        thal_label = st.selectbox("Thalassemia (thal)", ["Normal (1)", "Fixed defect (2)", "Reversible defect (3)"])
        threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    submitted = st.form_submit_button("Predict")

# ---------- Mapping labels to numeric codes ----------
sex = 1 if sex_label.startswith("Male") else 0
cp_map = {0:0, 1:1, 2:2, 3:3}
cp = cp_map[["Typical angina (0)", "Atypical angina (1)", "Non-anginal pain (2)", "Asymptomatic (3)"].index(cp_label)]
fbs = 1 if fbs_label.endswith("(1)") else 0
restecg = ["Normal (0)", "ST-T abnormality (1)", "LV hypertrophy (2)"].index(restecg_label)
exang = 1 if exang_label.endswith("(1)") else 0
slope = ["Upsloping (0)", "Flat (1)", "Downsloping (2)"].index(slope_label)
thal = ["Normal (1)", "Fixed defect (2)", "Reversible defect (3)"].index(thal_label) + 1

# Build input in the exact feature order the model expects
row = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}

if submitted:
    if model is None:
        st.stop()

    # Align columns to model's expectation; fill missing with NaN then 0 if needed
    X = pd.DataFrame([row])
    # Reorder/ensure columns
    for c in model_features:
        if c not in X.columns:
            X[c] = np.nan
    X = X[model_features]

    try:
        proba = float(model.predict_proba(X)[:, 1][0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred = int(proba >= threshold)
    st.subheader("Result")
    st.metric(label="Predicted probability of heart disease", value=f"{proba*100:.1f}%")
    st.write(f"**Decision at threshold {threshold:.2f}:** {'Positive (1)' if pred==1 else 'Negative (0)'}")

    # Simple guidance text
    st.caption("Note: This page is a **simulation tool** for model behavior and is **not** a medical device.")

# ---------- Batch testing (optional) ----------
st.header("Batch Test (CSV)")
st.caption("Upload a CSV with columns: " + ", ".join(FEATURES))
csv = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="csv")
if csv is not None and model is not None:
    try:
        df = pd.read_csv(csv)
        # align columns
        for c in model_features:
            if c not in df.columns:
                df[c] = np.nan
        df = df[model_features]
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)
        out = df.copy()
        out['proba_positive'] = probs
        out['pred_at_threshold'] = preds
        st.dataframe(out.head(20))
        st.download_button(
            label="Download predictions as CSV",
            data=out.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv'
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

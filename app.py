"""
Streamlit front-end for the oil classifier.

Density is deliberately absent from the inputs: the label is derived from API
gravity, which is a closed-form function of density, so a model given density
would be reading the answer rather than predicting it. See the README.
"""

import pickle

import pandas as pd
import streamlit as st

import features  # noqa: F401  -- required so pickle can resolve log_viscosity

MODEL_PATH = "model.pkl"

with open(MODEL_PATH, "rb") as f:
    artefact = pickle.load(f)

pipeline = artefact["pipeline"]
FEATURES = artefact["features"]

st.set_page_config(page_title="Oil Classifier", layout="centered")
st.title("Crude Oil Classification")
st.caption(
    "Predicts the API-gravity class (light / medium / heavy) from laboratory "
    "measurements. Trained on the NOAA Oil Library."
)

st.info(
    "Density and API gravity are not accepted as inputs. The class is defined by "
    "thresholds on API gravity, and API is computed directly from density — a model "
    "given either one would simply be restating the definition. This classifier uses "
    "the remaining physical and compositional measurements instead.",
    icon="ℹ️",
)

st.markdown("Leave any field empty if the measurement is unavailable — "
            "missing values are imputed inside the model pipeline.")


def number_input(label, help_text, value=""):
    raw = st.text_input(label, value, help=help_text)
    try:
        return float(raw)
    except ValueError:
        return None


col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical properties")
    viscosity = number_input("Dynamic viscosity (mPa·s)", "Measured near 15 °C.", "25.0")
    pour_point = number_input("Pour point (°C)", "Lowest temperature at which the oil flows.", "-9")
    flash_point = number_input("Flash point (°C)", "Lowest temperature at which vapour ignites.", "14")

with col2:
    st.subheader("Composition")
    st.caption("SARA fractions, expressed as fractions of 1 (not percentages).")
    saturates = number_input("Saturates", "e.g. 0.65", "")
    aromatics = number_input("Aromatics", "e.g. 0.25", "")
    resins = number_input("Resins", "e.g. 0.07", "")
    asphaltenes = number_input("Asphaltenes", "e.g. 0.03", "")
    sulfur = number_input("Sulfur", "Fraction of 1, e.g. 0.015", "")

st.markdown("---")

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "viscosity": viscosity,
        "pour_point": pour_point,
        "flash_point": flash_point,
        "saturates": saturates,
        "aromatics": aromatics,
        "resins": resins,
        "asphaltenes": asphaltenes,
        "sulfur": sulfur,
    }])[FEATURES]

    if row["viscosity"].isna().all():
        st.warning("Viscosity is the single strongest predictor. "
                   "Without it the prediction rests entirely on imputed medians.")

    prediction = pipeline.predict(row)[0]
    proba = pipeline.predict_proba(row)[0]

    st.subheader(f"Predicted class: {prediction.upper()}")

    proba_df = (
        pd.DataFrame({"class": pipeline.classes_, "probability": proba})
        .sort_values("probability", ascending=False)
        .set_index("class")
    )
    st.bar_chart(proba_df)
    st.dataframe(proba_df.style.format({"probability": "{:.3f}"}))

    supplied = int(row.notna().sum(axis=1).iloc[0])
    st.caption(
        f"{supplied} of {len(FEATURES)} measurements supplied; the rest were imputed. "
        "Held-out accuracy of this model is 0.85 (macro-F1 0.83), against 0.53 for "
        "always predicting the majority class."
    )

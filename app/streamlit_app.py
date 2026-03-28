"""
streamlit_app.py
----------------
Interactive demo for the Parkinson's UPDRS Severity Estimator.

Run:
    streamlit run app/streamlit_app.py
"""

import sys
import numpy as np
import torch
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import FeatureAttentionMLP
from utils import severity_bin

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_INFO = {
    "Jitter(%)":     ("Jitter (%)",          0.0, 2.0,   0.006,  "Pitch frequency instability"),
    "Jitter(Abs)":   ("Jitter Abs (s)",       0.0, 0.0002,0.00004,"Absolute pitch variation"),
    "Jitter:RAP":    ("Jitter RAP",           0.0, 1.0,   0.003,  "Relative average perturbation"),
    "Jitter:PPQ5":   ("Jitter PPQ5",          0.0, 1.0,   0.003,  "5-point perturbation quotient"),
    "Jitter:DDP":    ("Jitter DDP",           0.0, 2.0,   0.01,   "Average absolute difference"),
    "Shimmer":       ("Shimmer",              0.0, 0.5,   0.03,   "Amplitude instability"),
    "Shimmer(dB)":   ("Shimmer (dB)",         0.0, 4.0,   0.28,   "Shimmer in decibels"),
    "Shimmer:APQ3":  ("Shimmer APQ3",         0.0, 0.3,   0.016,  "3-point amplitude perturbation"),
    "Shimmer:APQ5":  ("Shimmer APQ5",         0.0, 0.3,   0.018,  "5-point amplitude perturbation"),
    "Shimmer:APQ11": ("Shimmer APQ11",        0.0, 0.3,   0.024,  "11-point amplitude perturbation"),
    "Shimmer:DDA":   ("Shimmer DDA",          0.0, 0.9,   0.047,  "Average absolute amplitude diff"),
    "NHR":           ("NHR",                  0.0, 0.5,   0.025,  "Noise-to-Harmonics Ratio"),
    "HNR":           ("HNR (dB)",             0.0, 40.0,  21.0,   "Harmonics-to-Noise Ratio"),
    "RPDE":          ("RPDE",                 0.0, 1.0,   0.5,    "Recurrence period density entropy"),
    "DFA":           ("DFA",                  0.5, 1.0,   0.72,   "Detrended fluctuation analysis"),
    "PPE":           ("PPE",                  0.0, 1.0,   0.21,   "Pitch period entropy"),
    "age":           ("Age (years)",          30,  90,    65,     "Patient age"),
    "sex":           ("Sex (0=M, 1=F)",       0,   1,     0,      "Biological sex"),
    "test_time":     ("Test time (days)",     0,   200,   100,    "Days since recruitment"),
}


@st.cache_resource
def load_model_and_scaler():
    scaler_path = MODELS_DIR / "scaler.pkl"
    model_path  = MODELS_DIR / "feature-attention_mlp.pt"

    if not scaler_path.exists() or not model_path.exists():
        return None, None

    scaler = joblib.load(scaler_path)
    n_features = len(FEATURE_INFO)
    model = FeatureAttentionMLP(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler


def severity_color(category: str) -> str:
    return {"Mild": "#2ecc71", "Moderate": "#f39c12", "Severe": "#e74c3c"}.get(category, "#888")


def main():
    st.set_page_config(
        page_title="Parkinson's UPDRS Estimator",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 Parkinson's UPDRS Severity Estimator")
    st.markdown(
        "Predict the **UPDRS motor severity score** from voice biomarkers using an "
        "explainable Feature-Attention MLP. Adjust the sliders to simulate a patient's "
        "voice profile and see the predicted severity with feature importance."
    )
    st.markdown("---")

    model, scaler = load_model_and_scaler()

    if model is None:
        st.warning(
            "⚠️ Trained model not found. Please run `python3 src/train.py --model attention` first."
        )
        st.info("Meanwhile, you can still explore the feature sliders below.")

    # ── Sidebar: feature inputs ──
    st.sidebar.header("Voice Biomarker Features")
    st.sidebar.markdown("Adjust values to simulate a patient profile.")

    feature_vals = {}
    for key, (label, vmin, vmax, default, description) in FEATURE_INFO.items():
        feature_vals[key] = st.sidebar.slider(
            label,
            min_value=float(vmin),
            max_value=float(vmax),
            value=float(default),
            help=description,
        )

    # ── Predict ──
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Prediction")

        if model is not None and st.button("🔍 Predict UPDRS Score", type="primary"):
            x_raw = np.array([[feature_vals[k] for k in FEATURE_INFO.keys()]], dtype=np.float32)
            x_scaled = scaler.transform(x_raw)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

            with torch.no_grad():
                score, attn_weights = model(x_tensor, return_attention=True)
                score = score.item()
                attn = attn_weights[0].numpy()

            severity = severity_bin(score)
            color = severity_color(severity)

            st.markdown(
                f"""
                <div style='background:{color}22;border-left:4px solid {color};
                     padding:16px;border-radius:8px;margin-bottom:12px'>
                    <h2 style='color:{color};margin:0'>UPDRS Score: {score:.1f}</h2>
                    <p style='margin:4px 0 0;color:{color};font-weight:500'>
                        Severity: {severity}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("UPDRS scale: 0 = no impairment, 108 = severe impairment")

            # ── Feature importance via input × attention ──
            st.subheader("Feature Importance for This Prediction")
            st.markdown("Contribution of each feature to this prediction (input magnitude × attention):")

            feature_names = list(FEATURE_INFO.keys())

            # Use input magnitude × attention weight as proxy importance
            importance = np.abs(x_scaled[0]) * attn
            sorted_idx = np.argsort(importance)[::-1][:10]

            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#1f77b4" if i == 0 else "#aec7e8" for i in range(10)]
            ax.barh(
                range(10),
                importance[sorted_idx][::-1],
                color=colors[::-1], edgecolor="none",
            )
            ax.set_yticks(range(10))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1], fontsize=10)
            ax.set_xlabel("Feature Contribution (|input| × attention weight)")
            ax.set_title("Top 10 Features Driving This Prediction")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        elif model is None:
            st.info("Train the model first, then predictions will appear here.")

    with col2:
        st.subheader("About This Model")
        st.markdown("""
        **Architecture:** Feature-Attention MLP

        **Novel contribution:**
        - Predicts *continuous* UPDRS score (not binary PD/healthy)
        - Learnable attention layer reveals which biomarkers drive severity
        - SHAP explanations for clinical interpretability

        **Dataset:** UCI Parkinson's Telemonitoring (5,875 recordings, 42 subjects)

        **Key voice biomarkers:**
        | Feature | Clinical meaning |
        |---------|-----------------|
        | Jitter (%) | Vocal frequency instability |
        | Shimmer | Amplitude variation |
        | HNR | Voice clarity |
        | RPDE | Nonlinear dynamics |
        | PPE | Pitch regularity |

        ---
        **Severity scale:**
        - 🟢 Mild: < 20
        - 🟡 Moderate: 20 – 36
        - 🔴 Severe: > 36
        """)

    st.markdown("---")
    st.caption(
        "Research project | MIT-WPU Final Year BTech (AI & Data Science) | "
        "Target: IEEE Access / EMBC 2026"
    )


if __name__ == "__main__":
    main()

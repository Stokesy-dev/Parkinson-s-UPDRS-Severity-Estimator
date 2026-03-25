"""
explain.py
----------
SHAP-based explainability pipeline for the Feature-Attention MLP.

Generates:
  - SHAP summary beeswarm plot (global feature importance)
  - SHAP bar chart (mean |SHAP| per feature)
  - SHAP force plot for 3 sample patients
  - Attention weight heatmap (global, from the attention layer)

Run after training:
    python3 src/explain.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import shap
import joblib
from pathlib import Path

from dataset import prepare_data
from model import FeatureAttentionMLP

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

MODELS_DIR = Path(__file__).parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")   # SHAP runs on CPU

# Clinical descriptions for top PD biomarkers (for paper discussion section)
BIOMARKER_NOTES = {
    "Jitter(%)":      "Vocal frequency instability — key PD motor symptom",
    "Jitter(Abs)":    "Absolute jitter — pitch variation per period",
    "Jitter:RAP":     "Relative average perturbation of pitch",
    "Jitter:PPQ5":    "5-point period perturbation quotient",
    "Shimmer":        "Amplitude variation — linked to muscle rigidity",
    "Shimmer(dB)":    "Shimmer in decibels",
    "Shimmer:APQ3":   "3-point amplitude perturbation quotient",
    "Shimmer:APQ11":  "11-point amplitude perturbation quotient",
    "NHR":            "Noise-to-harmonics ratio — breathiness indicator",
    "HNR":            "Harmonics-to-noise ratio — voice clarity",
    "RPDE":           "Recurrence period density entropy — nonlinear dynamics",
    "DFA":            "Detrended fluctuation analysis — signal complexity",
    "PPE":            "Pitch period entropy — vocal dysregulation",
}


def load_model(n_features: int) -> FeatureAttentionMLP:
    model_path = MODELS_DIR / "feature-attention_mlp.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    model = FeatureAttentionMLP(n_features=n_features)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def model_predict(X: np.ndarray, model: FeatureAttentionMLP) -> np.ndarray:
    """Wrapper for SHAP: numpy in → numpy out."""
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return model(t).numpy()


def compute_shap_values(model, X_train: np.ndarray, X_test: np.ndarray, n_background: int = 100):
    """
    Use SHAP KernelExplainer (model-agnostic, works with any PyTorch model).
    Background = subset of training data for expectation estimation.
    """
    print("Computing SHAP values (this takes ~2–5 min on CPU)...")
    background = shap.sample(X_train, n_background, random_state=42)
    predict_fn = lambda x: model_predict(x, model)
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test, nsamples=200)
    print(f"SHAP values shape: {shap_values.shape}")
    return shap_values, explainer


def plot_shap_summary(shap_values, X_test, feature_names, save_name="shap_summary.png"):
    """Beeswarm plot — shows distribution of SHAP impact per feature."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        show=False, plot_size=None,
    )
    plt.title("SHAP Feature Impact on UPDRS Severity Prediction", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / save_name)
    plt.close()
    print(f"Saved → {FIGURES_DIR / save_name}")


def plot_shap_bar(shap_values, feature_names, save_name="shap_bar.png"):
    """Bar chart of mean absolute SHAP values (global importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1f77b4" if i == 0 else "#aec7e8" for i in range(top_n)]
    ax.barh(
        range(top_n),
        mean_abs[sorted_idx[:top_n]][::-1],
        color=colors[::-1],
        edgecolor="none",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]][::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / save_name)
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / save_name}")

    # Print top features with clinical notes
    print("\nTop features and clinical significance:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        fname = feature_names[idx]
        note = BIOMARKER_NOTES.get(fname, "")
        print(f"  {rank:2}. {fname:<20}  SHAP={mean_abs[idx]:.4f}  | {note}")


def plot_attention_weights(model, X_test_tensor, feature_names, save_name="attention_weights.png"):
    """Heatmap of mean attention weights from the attention layer."""
    weights = model.get_attention_weights(X_test_tensor).numpy()
    mean_weights = weights.mean(axis=0)

    sorted_idx = np.argsort(mean_weights)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        range(top_n),
        mean_weights[sorted_idx[:top_n]][::-1],
        color="#5e81ac", edgecolor="none",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]][::-1], fontsize=10)
    ax.set_xlabel("Mean Attention Weight")
    ax.set_title("Feature Attention Weights (Learned by Model)", fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / save_name)
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / save_name}")


def main():
    print("Loading data...")
    data = prepare_data()
    feature_names = data["feature_cols"]
    n_features = data["n_features"]

    print("Loading trained model...")
    model = load_model(n_features)

    # SHAP on test set (or subset for speed)
    X_test = data["X_test"]
    X_train = data["X_train"]

    # Limit test samples for speed if large
    if len(X_test) > 200:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), 200, replace=False)
        X_test_shap = X_test[idx]
    else:
        X_test_shap = X_test

    shap_values, _ = compute_shap_values(model, X_train, X_test_shap)

    plot_shap_summary(shap_values, X_test_shap, feature_names)
    plot_shap_bar(shap_values, feature_names)

    X_test_tensor = torch.tensor(X_test_shap, dtype=torch.float32)
    plot_attention_weights(model, X_test_tensor, feature_names)

    print("\nAll explainability figures saved to results/figures/")


if __name__ == "__main__":
    main()

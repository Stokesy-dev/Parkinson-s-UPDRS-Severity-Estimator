"""
utils.py
--------
Shared metrics computation and publication-ready plotting helpers.
All figures saved to results/figures/ at 300 DPI.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
METRICS_DIR = Path(__file__).parent.parent / "results" / "metrics"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── Metrics ───────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """Compute MAE, RMSE, R² and return as dict."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    metrics = {
        f"{prefix}mae": round(float(mae), 4),
        f"{prefix}rmse": round(float(rmse), 4),
        f"{prefix}r2": round(float(r2), 4),
    }
    return metrics


def save_metrics(metrics: dict, filename: str):
    """Save metrics dict to JSON."""
    path = METRICS_DIR / filename
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")


def load_metrics(filename: str) -> dict:
    path = METRICS_DIR / filename
    with open(path) as f:
        return json.load(f)


# ─────────────────────────── Plots ───────────────────────────

def plot_loss_curves(train_losses: list, val_losses: list, save_name: str = "loss_curves.png"):
    """Training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss", color="#1f77b4", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val loss", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.savefig(FIGURES_DIR / save_name)
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / save_name}")


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_name: str = "pred_vs_actual.png",
):
    """Scatter plot of predicted vs actual UPDRS scores."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#1f77b4", edgecolors="none", s=20)
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel("Actual UPDRS Score")
    ax.set_ylabel("Predicted UPDRS Score")
    ax.set_title(f"{model_name}  |  MAE={mae:.2f}  R²={r2:.3f}")
    ax.legend()
    fig.savefig(FIGURES_DIR / save_name)
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / save_name}")


def plot_baseline_comparison(results: dict, save_name: str = "baseline_comparison.png"):
    """
    Bar chart comparing MAE and R² across all models.
    results = {"Model A": {"mae": 3.2, "r2": 0.75}, ...}
    """
    models = list(results.keys())
    maes = [results[m]["mae"] for m in models]
    r2s = [results[m]["r2"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    colors = ["#aec7e8"] * (len(models) - 1) + ["#1f77b4"]   # highlight last (ours)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, maes, width, color=colors, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylabel("MAE (lower is better)")
    ax1.set_title("MAE Comparison")
    for bar, val in zip(bars1, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(x, r2s, width, color=colors, edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15, ha="right")
    ax2.set_ylabel("R² (higher is better)")
    ax2.set_title("R² Comparison")
    for bar, val in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — UPDRS Severity Regression", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name)
    plt.close(fig)
    print(f"Saved → {FIGURES_DIR / save_name}")


def severity_bin(score: float) -> str:
    """Map UPDRS score to clinical severity category."""
    if score < 20:
        return "Mild"
    elif score < 36:
        return "Moderate"
    else:
        return "Severe"

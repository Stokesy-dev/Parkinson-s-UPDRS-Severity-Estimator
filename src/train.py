"""
train.py
--------
Training pipeline for all models:
  - Random Forest baseline
  - SVR baseline
  - Plain MLP baseline (ablation)
  - Feature-Attention MLP (proposed model)

Usage:
    python3 src/train.py --model all          # train everything
    python3 src/train.py --model attention    # proposed model only
    python3 src/train.py --model rf           # RF baseline only
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import joblib

from dataset import prepare_data
from model import FeatureAttentionMLP, PlainMLP, count_parameters
from utils import (
    compute_metrics, save_metrics,
    plot_loss_curves, plot_pred_vs_actual, plot_baseline_comparison,
)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


# ───────────────────────── Classical Baselines ─────────────────────────

def train_rf(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestRegressor
    print("\n── Random Forest ──")
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    joblib.dump(rf, MODELS_DIR / "rf.pkl")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")
    plot_pred_vs_actual(y_test, y_pred, "Random Forest", "rf_pred_vs_actual.png")
    return y_pred, metrics


def train_svr(X_train, y_train, X_test, y_test):
    from sklearn.svm import SVR
    from sklearn.pipeline import Pipeline
    print("\n── SVR ──")
    svr = SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    joblib.dump(svr, MODELS_DIR / "svr.pkl")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")
    plot_pred_vs_actual(y_test, y_pred, "SVR", "svr_pred_vs_actual.png")
    return y_pred, metrics


# ───────────────────────── PyTorch Training Loop ─────────────────────────

def make_loaders(data: dict, batch_size: int = 64):
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor(data["X_train"]), to_tensor(data["y_train"]))
    val_ds = TensorDataset(to_tensor(data["X_val"]),   to_tensor(data["y_val"]))
    test_ds = TensorDataset(to_tensor(data["X_test"]), to_tensor(data["y_test"]))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size),
        DataLoader(test_ds,  batch_size=batch_size),
    )


def train_nn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_test: np.ndarray,
    model_name: str,
    epochs: int = 150,
    lr: float = 1e-3,
    patience: int = 20,
):
    print(f"\n── {model_name} ({count_parameters(model):,} params) ──")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in tqdm(range(1, epochs + 1), desc=model_name):
        # ── Train ──
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)

        # ── Validate ──
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_batch_losses.append(criterion(pred, yb).item())
        val_loss = np.mean(val_batch_losses)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # ── Evaluate on test set ──
    model.load_state_dict(best_state)
    torch.save(best_state, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pt")

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")

    safe_name = model_name.lower().replace(" ", "_")
    plot_loss_curves(train_losses, val_losses, f"{safe_name}_loss_curves.png")
    plot_pred_vs_actual(y_test, y_pred, model_name, f"{safe_name}_pred_vs_actual.png")

    return y_pred, metrics, model


# ───────────────────────── Main ─────────────────────────

def main(args):
    print(f"Device: {DEVICE}")
    data = prepare_data(seed=42)
    train_loader, val_loader, test_loader = make_loaders(data, batch_size=64)

    all_results = {}

    if args.model in ("rf", "all"):
        _, rf_metrics = train_rf(data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        all_results["Random Forest"] = {"mae": rf_metrics["test_mae"], "r2": rf_metrics["test_r2"]}
        save_metrics(rf_metrics, "rf_metrics.json")

    if args.model in ("svr", "all"):
        _, svr_metrics = train_svr(data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        all_results["SVR"] = {"mae": svr_metrics["test_mae"], "r2": svr_metrics["test_r2"]}
        save_metrics(svr_metrics, "svr_metrics.json")

    if args.model in ("mlp", "all"):
        plain_mlp = PlainMLP(n_features=data["n_features"])
        _, mlp_metrics, _ = train_nn(
            plain_mlp, train_loader, val_loader, test_loader,
            data["y_test"], "Plain MLP", epochs=args.epochs,
        )
        all_results["Plain MLP"] = {"mae": mlp_metrics["test_mae"], "r2": mlp_metrics["test_r2"]}
        save_metrics(mlp_metrics, "plain_mlp_metrics.json")

    if args.model in ("attention", "all"):
        attn_mlp = FeatureAttentionMLP(n_features=data["n_features"], dropout=args.dropout)
        _, attn_metrics, trained_model = train_nn(
            attn_mlp, train_loader, val_loader, test_loader,
            data["y_test"], "Feature-Attention MLP", epochs=args.epochs,
        )
        all_results["Attention MLP (Ours)"] = {
            "mae": attn_metrics["test_mae"], "r2": attn_metrics["test_r2"]
        }
        save_metrics(attn_metrics, "attention_mlp_metrics.json")

    if len(all_results) > 1:
        plot_baseline_comparison(all_results, "baseline_comparison.png")
        with open(MODELS_DIR.parent / "results" / "metrics" / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\n── Summary ──")
        print(f"{'Model':<25} {'MAE':>8} {'R²':>8}")
        print("─" * 45)
        for name, m in all_results.items():
            print(f"{name:<25} {m['mae']:>8.4f} {m['r2']:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rf", "svr", "mlp", "attention", "all"],
        default="all",
        help="Which model(s) to train",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    main(args)

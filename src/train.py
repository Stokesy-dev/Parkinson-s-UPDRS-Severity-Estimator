"""
train.py
--------
Training pipeline supporting two evaluation protocols:

  --protocol random   : naive random split (replicates prior work — shows inflated metrics)
  --protocol subject  : subject-wise split (correct protocol — shows true generalization)
  --protocol both     : runs both, produces the paper's key comparison table

Usage:
    python3 src/train.py --model all --protocol both
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dataset import prepare_data, load_raw, get_feature_cols, TARGET_COL, SUBJECT_COL
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


def prepare_random_split(test_size=0.20, seed=42):
    """Naive random split — replicates prior work. Causes subject leakage."""
    df = load_raw()
    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.125, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"\nRandom split (LEAKY — for comparison only):")
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }


def train_rf(X_train, y_train, X_test, y_test, tag=""):
    from sklearn.ensemble import RandomForestRegressor
    print(f"\n── Random Forest {tag}──")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")
    plot_pred_vs_actual(y_test, y_pred, f"Random Forest {tag}", f"rf_{tag.strip()}_pred_vs_actual.png")
    return y_pred, metrics


def train_svr(X_train, y_train, X_test, y_test, tag=""):
    from sklearn.svm import SVR
    print(f"\n── SVR {tag}──")
    svr = SVR(kernel="rbf", C=10, epsilon=0.5, gamma="scale")
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")
    plot_pred_vs_actual(y_test, y_pred, f"SVR {tag}", f"svr_{tag.strip()}_pred_vs_actual.png")
    return y_pred, metrics


def make_loaders(data, batch_size=64):
    def tt(arr): return torch.tensor(arr, dtype=torch.float32)
    train_ds = TensorDataset(tt(data["X_train"]), tt(data["y_train"]))
    val_ds   = TensorDataset(tt(data["X_val"]),   tt(data["y_val"]))
    test_ds  = TensorDataset(tt(data["X_test"]),  tt(data["y_test"]))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size),
            DataLoader(test_ds,  batch_size=batch_size))


def train_nn(model, train_loader, val_loader, test_loader,
             y_test, model_name, tag="", epochs=150, lr=1e-3, patience=25):
    full_name = f"{model_name} {tag}".strip()
    print(f"\n── {full_name} ({count_parameters(model):,} params) ──")
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in tqdm(range(1, epochs + 1), desc=full_name):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_batch_losses.append(criterion(model(xb), yb).item())

        train_losses.append(np.mean(batch_losses))
        val_losses.append(np.mean(val_batch_losses))
        scheduler.step(val_losses[-1])

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    safe = full_name.lower().replace(" ", "_").replace("[","").replace("]","")
    torch.save(best_state, MODELS_DIR / f"{safe}.pt")

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    y_pred = np.concatenate(preds)

    metrics = compute_metrics(y_test, y_pred, prefix="test_")
    print(f"  MAE={metrics['test_mae']}  RMSE={metrics['test_rmse']}  R²={metrics['test_r2']}")
    safe_name = full_name.lower().replace(" ","_").replace("[","").replace("]","")
    plot_loss_curves(train_losses, val_losses, f"{safe_name}_loss_curves.png")
    plot_pred_vs_actual(y_test, y_pred, full_name, f"{safe_name}_pred_vs_actual.png")
    return y_pred, metrics, model


def run_protocol(args, data, protocol_tag):
    tag = f"[{protocol_tag}]"
    results = {}
    train_loader, val_loader, test_loader = make_loaders(data)

    if args.model in ("rf", "all"):
        _, m = train_rf(data["X_train"], data["y_train"], data["X_test"], data["y_test"], tag)
        results["Random Forest"] = {"mae": m["test_mae"], "r2": m["test_r2"]}
        save_metrics(m, f"rf_{protocol_tag}_metrics.json")

    if args.model in ("svr", "all"):
        _, m = train_svr(data["X_train"], data["y_train"], data["X_test"], data["y_test"], tag)
        results["SVR"] = {"mae": m["test_mae"], "r2": m["test_r2"]}
        save_metrics(m, f"svr_{protocol_tag}_metrics.json")

    if args.model in ("mlp", "all"):
        plain = PlainMLP(n_features=data["n_features"])
        _, m, _ = train_nn(plain, train_loader, val_loader, test_loader,
                           data["y_test"], "Plain MLP", tag, epochs=args.epochs)
        results["Plain MLP"] = {"mae": m["test_mae"], "r2": m["test_r2"]}
        save_metrics(m, f"plain_mlp_{protocol_tag}_metrics.json")

    if args.model in ("attention", "all"):
        attn = FeatureAttentionMLP(n_features=data["n_features"], dropout=args.dropout)
        _, m, _ = train_nn(attn, train_loader, val_loader, test_loader,
                           data["y_test"], "Feature-Attention MLP", tag, epochs=args.epochs)
        results["Attention MLP (Ours)"] = {"mae": m["test_mae"], "r2": m["test_r2"]}
        save_metrics(m, f"attention_mlp_{protocol_tag}_metrics.json")

    return results


def main(args):
    print(f"Device: {DEVICE}")
    results_subject = {}
    results_random  = {}

    if args.protocol in ("subject", "both"):
        print("\n" + "="*55)
        print("PROTOCOL 1: Subject-wise split (correct, no leakage)")
        print("="*55)
        data_subject = prepare_data(seed=42)
        results_subject = run_protocol(args, data_subject, "subject")

    if args.protocol in ("random", "both"):
        print("\n" + "="*55)
        print("PROTOCOL 2: Random split (replicates prior work — LEAKY)")
        print("="*55)
        data_random = prepare_random_split(seed=42)
        results_random = run_protocol(args, data_random, "random")

    if args.protocol == "both":
        print("\n" + "="*65)
        print("COMPARISON TABLE — paper's key result")
        print("="*65)
        print(f"{'Model':<25} {'Subject R²':>12} {'Random R²':>12} {'Gap':>10}")
        print("─" * 65)
        for name in results_subject:
            r2_s = results_subject[name]["r2"]
            r2_r = results_random.get(name, {}).get("r2", "—")
            gap  = round(r2_r - r2_s, 4) if isinstance(r2_r, float) else "—"
            print(f"{name:<25} {r2_s:>12} {r2_r:>12} {str(gap):>10}")

        combined = {"subject_wise": results_subject, "random_split": results_random}
        out = Path(__file__).parent.parent / "results" / "metrics" / "protocol_comparison.json"
        with open(out, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["rf","svr","mlp","attention","all"], default="all")
    parser.add_argument("--protocol", choices=["random","subject","both"],          default="both")
    parser.add_argument("--epochs",   type=int,   default=150)
    parser.add_argument("--dropout",  type=float, default=0.3)
    args = parser.parse_args()
    main(args)

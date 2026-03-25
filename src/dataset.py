"""
dataset.py
----------
Data loading, preprocessing, and subject-wise splitting for the
UCI Parkinson's Telemonitoring dataset.

CRITICAL: Subject-wise splitting prevents data leakage (multiple
recordings per patient). Random splitting inflates metrics and
gets flagged immediately by IEEE reviewers.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib

DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
    "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "Shimmer:APQ11", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA", "PPE",
    "age", "sex", "test_time",
    "Jitter:DDP",   # included separately in UCI cols — handled below
]

TARGET_COL = "total_UPDRS"
SUBJECT_COL = "subject#"


def download_dataset():
    """Download UCI Parkinson's Telemonitoring dataset via ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo
        print("Downloading UCI Parkinson's Telemonitoring dataset...")
        dataset = fetch_ucirepo(id=189)
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        # ucimlrepo may not include subject# in features — add from ids if present
        if hasattr(dataset.data, 'ids') and dataset.data.ids is not None:
            df[SUBJECT_COL] = dataset.data.ids.values.flatten()
        DATA_DIR.mkdir(exist_ok=True)
        out_path = DATA_DIR / "parkinsons_updrs.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}  |  Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Auto-download failed: {e}")
        print("Manual download:")
        print("  https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring")
        print(f"  Place 'parkinsons_updrs.data' inside {DATA_DIR}/")
        return None


def load_raw() -> pd.DataFrame:
    """Load raw CSV. Falls back to .data file if CSV not found."""
    csv_path = DATA_DIR / "parkinsons_updrs.csv"
    data_path = DATA_DIR / "parkinsons_updrs.data"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif data_path.exists():
        df = pd.read_csv(data_path)
        df.to_csv(csv_path, index=False)
        print(f"Converted .data → {csv_path}")
    else:
        raise FileNotFoundError(
            f"Dataset not found in {DATA_DIR}. Run: python3 src/dataset.py --download"
        )
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature columns present in df (excludes target, subject ID, motor_UPDRS)."""
    exclude = {TARGET_COL, "motor_UPDRS", SUBJECT_COL}
    return [c for c in df.columns if c not in exclude]


def subject_wise_split(df: pd.DataFrame, val_size=0.10, test_size=0.20, seed=42):
    """
    Split by subject ID to prevent data leakage.
    Returns (train_df, val_df, test_df).
    """
    subjects = df[SUBJECT_COL].values
    features = get_feature_cols(df)

    # First split: train+val vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=subjects))

    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Second split: train vs val
    val_ratio_adjusted = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio_adjusted, random_state=seed)
    trainval_subjects = trainval_df[SUBJECT_COL].values
    train_idx, val_idx = next(gss_val.split(trainval_df, groups=trainval_subjects))

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    # Verify no subject overlap
    train_subj = set(train_df[SUBJECT_COL])
    val_subj = set(val_df[SUBJECT_COL])
    test_subj = set(test_df[SUBJECT_COL])
    assert train_subj.isdisjoint(val_subj), "Subject leakage: train/val overlap!"
    assert train_subj.isdisjoint(test_subj), "Subject leakage: train/test overlap!"
    assert val_subj.isdisjoint(test_subj), "Subject leakage: val/test overlap!"

    print(f"Split complete (subject-wise, no leakage):")
    print(f"  Train: {len(train_df)} rows | {len(train_subj)} subjects")
    print(f"  Val  : {len(val_df)} rows | {len(val_subj)} subjects")
    print(f"  Test : {len(test_df)} rows | {len(test_subj)} subjects")

    return train_df, val_df, test_df


def fit_scaler(train_df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    """Fit StandardScaler on training set only. Save to models/ for inference."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Scaler saved to {MODELS_DIR / 'scaler.pkl'}")
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler) -> pd.DataFrame:
    """Apply fitted scaler to a split."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def get_numpy(df: pd.DataFrame, feature_cols: list):
    """Extract X (features) and y (UPDRS) as numpy arrays."""
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def prepare_data(seed=42):
    """
    Full pipeline: load → split → scale → return numpy arrays.
    Returns dict with train/val/test X, y arrays + metadata.
    """
    df = load_raw()
    feature_cols = get_feature_cols(df)

    print(f"\nDataset loaded: {df.shape[0]} rows, {len(feature_cols)} features")
    print(f"Target range: {df[TARGET_COL].min():.2f} – {df[TARGET_COL].max():.2f}")
    print(f"Subjects: {df[SUBJECT_COL].nunique()}")

    train_df, val_df, test_df = subject_wise_split(df, seed=seed)
    scaler = fit_scaler(train_df, feature_cols)

    train_df = apply_scaler(train_df, feature_cols, scaler)
    val_df = apply_scaler(val_df, feature_cols, scaler)
    test_df = apply_scaler(test_df, feature_cols, scaler)

    X_train, y_train = get_numpy(train_df, feature_cols)
    X_val, y_val = get_numpy(val_df, feature_cols)
    X_test, y_test = get_numpy(test_df, feature_cols)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val,   "y_val": y_val,
        "X_test": X_test,  "y_test": y_test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "n_features": len(feature_cols),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download dataset from UCI")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    else:
        data = prepare_data()
        print(f"\nReady for training:")
        print(f"  X_train: {data['X_train'].shape}")
        print(f"  X_val  : {data['X_val'].shape}")
        print(f"  X_test : {data['X_test'].shape}")

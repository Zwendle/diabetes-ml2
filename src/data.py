"""Data loading and preprocessing for the diabetes MLP. NumPy/pandas only."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"


def load_raw(path=DATA_FILE, drop_duplicates=True):
    """load cvs and return (X, y, feature_names) as float arrays"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the CSV from the Kaggle link on the UCI page "
            "and place it in data/raw/ (or update DATA_FILE)."
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not in file. Columns: {list(df.columns)}")
    if drop_duplicates:
        # survey extracts sometimes contain duplicate rows, drop them to avoid inflating train/test scores
        df = df.drop_duplicates().reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=np.float64)
    features = df.drop(columns=[TARGET])
    return features.to_numpy(dtype=np.float64), y, list(features.columns)


def stratified_split(y, fractions=(0.7, 0.15, 0.15), seed=42):
    """return index arrays (train, val, test) preserving the class ratio in each"""
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        parts[0].append(idx[:n_train])
        parts[1].append(idx[n_train:n_train + n_val])
        parts[2].append(idx[n_train + n_val:])
    return tuple(rng.permutation(np.concatenate(p)) for p in parts)


class Standardizer:
    """z score non-binary columns, fit on training data only to avoid leakage"""

    def fit(self, X):
        self.cols = np.array([len(np.unique(X[:, j])) > 2 for j in range(X.shape[1])])
        self.mean = X[:, self.cols].mean(axis=0)
        self.std = X[:, self.cols].std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X = X.copy()
        X[:, self.cols] = (X[:, self.cols] - self.mean) / self.std
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def get_splits(path=DATA_FILE, seed=42, drop_duplicates=True):
    """full pipeline: load -> stratified split -> standardize (fit on train only)"""
    X, y, names = load_raw(path, drop_duplicates)
    tr, va, te = stratified_split(y, seed=seed)
    scaler = Standardizer().fit(X[tr])
    return {
        "X_train": scaler.transform(X[tr]), "y_train": y[tr],
        "X_val": scaler.transform(X[va]), "y_val": y[va],
        "X_test": scaler.transform(X[te]), "y_test": y[te],
        "feature_names": names, "scaler": scaler,
    }


if __name__ == "__main__":
    d = get_splits()
    for k in ("train", "val", "test"):
        X, y = d[f"X_{k}"], d[f"y_{k}"]
        print(f"{k:5s} X={X.shape}  positive rate={y.mean():.3f}")
    print("features:", d["feature_names"])
    print("standardized columns:", [n for n, c in zip(d["feature_names"], d["scaler"].cols) if c])

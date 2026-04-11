"""
C-MAPSS data loading and preprocessing for STAR replication.

Sensor selection: 14 sensors (s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21)
Normalization: min-max on train set only
RUL labels: piecewise linear with cap=125
Windows: sliding window with stride=1, left-pad short engines
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.cluster import KMeans

DATA_ROOT = Path("/home/sagemaker-user/IndustrialJEPA/datasets/data/cmapss/6. Turbofan Engine Degradation Simulation Data Set")

# 0-indexed column positions for the 14 selected sensors in the 26-column raw format:
# [engine_id(0), cycle(1), op1(2), op2(3), op3(4), s1(5)..s21(25)]
# s2=col5+1=col6? No: s1=col5, s2=col6, ..., sN=col(4+N)
# So sK is at column index 4+K
SENSOR_COLS = [4+2, 4+3, 4+4, 4+7, 4+8, 4+9, 4+11, 4+12, 4+13, 4+14, 4+15, 4+17, 4+20, 4+21]
# = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]

OP_COLS = [2, 3, 4]  # op_setting1, op_setting2, op_setting3

COL_NAMES = (
    ["engine_id", "cycle"] +
    ["op1", "op2", "op3"] +
    [f"s{i}" for i in range(1, 22)]
)

RUL_CAP = 125
N_SENSORS = 14


def load_raw(subset: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load raw train, test, and RUL files for a given subset (e.g. 'FD001')."""
    def _read(path):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
        return df

    train_df = _read(DATA_ROOT / f"train_{subset}.txt")
    test_df = _read(DATA_ROOT / f"test_{subset}.txt")
    rul_arr = np.loadtxt(DATA_ROOT / f"RUL_{subset}.txt")
    return train_df, test_df, rul_arr


def compute_rul_labels(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add a 'rul' column to a training dataframe using piecewise linear capped RUL."""
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["max_cycle"] = df["engine_id"].map(max_cycles)
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=cap)
    return df


def fit_normalizer(train_df: pd.DataFrame) -> dict:
    """Compute per-sensor min-max statistics from training data."""
    sensors = train_df.iloc[:, SENSOR_COLS].values.astype(np.float32)
    return {
        "min": sensors.min(axis=0),
        "max": sensors.max(axis=0),
    }


def normalize(data: np.ndarray, stats: dict) -> np.ndarray:
    """Apply min-max normalization using pre-computed stats."""
    return (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)


def fit_condition_normalizer(train_df: pd.DataFrame, n_conditions: int = 6) -> dict:
    """Per-operating-condition normalizer for multi-condition subsets (FD002/FD004).

    Uses KMeans on op_setting columns to assign conditions, then computes
    per-condition min-max stats.
    """
    op_data = train_df.iloc[:, OP_COLS].values.astype(np.float32)
    kmeans = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
    kmeans.fit(op_data)

    train_df = train_df.copy()
    train_df["condition"] = kmeans.predict(op_data)

    sensors = train_df.iloc[:, SENSOR_COLS].values.astype(np.float32)
    cond_labels = train_df["condition"].values

    per_cond_min = np.zeros((n_conditions, N_SENSORS), dtype=np.float32)
    per_cond_max = np.zeros((n_conditions, N_SENSORS), dtype=np.float32)

    for c in range(n_conditions):
        mask = cond_labels == c
        per_cond_min[c] = sensors[mask].min(axis=0)
        per_cond_max[c] = sensors[mask].max(axis=0)

    return {
        "kmeans": kmeans,
        "per_cond_min": per_cond_min,
        "per_cond_max": per_cond_max,
    }


def normalize_by_condition(data: np.ndarray, op_data: np.ndarray, cond_stats: dict) -> np.ndarray:
    """Apply per-condition min-max normalization."""
    kmeans = cond_stats["kmeans"]
    conditions = kmeans.predict(op_data.astype(np.float32))
    result = np.zeros_like(data, dtype=np.float32)
    for c in range(cond_stats["per_cond_min"].shape[0]):
        mask = conditions == c
        if mask.sum() > 0:
            result[mask] = (data[mask] - cond_stats["per_cond_min"][c]) / (
                cond_stats["per_cond_max"][c] - cond_stats["per_cond_min"][c] + 1e-8
            )
    return result


def make_windows(
    df: pd.DataFrame,
    window_length: int,
    stats: dict,
    is_train: bool = True,
    rul_cap: int = RUL_CAP,
    cond_stats: dict = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for training or test data.

    For train: all windows per engine, stride=1.
    For test: only the LAST window per engine.

    Returns:
        X: (N, window_length, N_SENSORS) float32
        y: (N,) float32, RUL values
    """
    X_list = []
    y_list = []

    if is_train:
        df = compute_rul_labels(df, cap=rul_cap)

    engines = df["engine_id"].unique()

    for eid in engines:
        eng = df[df["engine_id"] == eid].reset_index(drop=True)
        sensors = eng.iloc[:, SENSOR_COLS].values.astype(np.float32)
        op = eng.iloc[:, OP_COLS].values.astype(np.float32)

        # Normalize
        if cond_stats is not None:
            sensors_norm = normalize_by_condition(sensors, op, cond_stats)
        else:
            sensors_norm = normalize(sensors, stats)

        T = len(eng)

        # Left-pad if shorter than window_length
        if T < window_length:
            pad_len = window_length - T
            sensors_norm = np.concatenate([np.tile(sensors_norm[:1], (pad_len, 1)), sensors_norm], axis=0)
            if is_train:
                # Also pad RUL
                rul_vals = eng["rul"].values
                rul_vals = np.concatenate([np.tile(rul_vals[:1], pad_len), rul_vals])
            T = window_length
        else:
            if is_train:
                rul_vals = eng["rul"].values

        if is_train:
            # All windows with stride=1
            for start in range(T - window_length + 1):
                X_list.append(sensors_norm[start:start + window_length])
                y_list.append(rul_vals[start + window_length - 1])
        else:
            # Only last window
            X_list.append(sensors_norm[T - window_length:T])
            # For test engines, RUL is the "time to failure" remaining at the last recorded cycle
            # This is stored separately in RUL_FDxxx.txt
            y_list.append(np.nan)  # placeholder, replaced by caller

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


class CMAPSSDataset(Dataset):
    """PyTorch dataset wrapping windowed C-MAPSS data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(
    subset: str,
    window_length: int,
    batch_size: int,
    val_fraction: float = 0.15,
    seed: int = 42,
    rul_cap: int = RUL_CAP,
    use_cond_norm: bool = False,
) -> dict:
    """
    Full data preparation pipeline for one C-MAPSS subset.

    Returns a dict with:
        train_loader, val_loader, test_loader,
        stats, n_train_engines, n_val_engines, n_test_engines
    """
    train_df_raw, test_df_raw, rul_test = load_raw(subset)

    # Train/val split by engine
    all_engines = train_df_raw["engine_id"].unique()
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(all_engines) * val_fraction))
    val_engines = rng.choice(all_engines, size=n_val, replace=False)
    train_engines = np.array([e for e in all_engines if e not in val_engines])

    train_df = train_df_raw[train_df_raw["engine_id"].isin(train_engines)].reset_index(drop=True)
    val_df = train_df_raw[train_df_raw["engine_id"].isin(val_engines)].reset_index(drop=True)

    # Fit normalizer on training engines only
    stats = fit_normalizer(train_df)

    cond_stats = None
    if use_cond_norm:
        cond_stats = fit_condition_normalizer(train_df, n_conditions=6)

    # Create windows
    X_train, y_train = make_windows(train_df, window_length, stats, is_train=True, rul_cap=rul_cap, cond_stats=cond_stats)
    X_val, y_val = make_windows(val_df, window_length, stats, is_train=True, rul_cap=rul_cap, cond_stats=cond_stats)
    X_test, y_test_placeholder = make_windows(test_df_raw, window_length, stats, is_train=False, rul_cap=rul_cap, cond_stats=cond_stats)

    # Fill in actual test RUL values from RUL_FDxxx.txt
    # rul_test[i] = RUL for test engine i+1 at last recorded cycle (not capped)
    # We cap at RUL_CAP for consistency
    y_test = np.minimum(rul_test, rul_cap).astype(np.float32)
    assert len(y_test) == X_test.shape[0], (
        f"Test RUL count mismatch: {len(y_test)} vs {X_test.shape[0]}"
    )

    train_ds = CMAPSSDataset(X_train, y_train)
    val_ds = CMAPSSDataset(X_val, y_val)
    test_ds = CMAPSSDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "stats": stats,
        "cond_stats": cond_stats,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "n_test_engines": len(test_df_raw["engine_id"].unique()),
        "n_train_windows": len(X_train),
        "n_val_windows": len(X_val),
        "n_test_windows": len(X_test),
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    # Sanity check on FD001
    print("=== FD001 Sanity Check ===")
    d = prepare_data("FD001", window_length=32, batch_size=32, seed=42)
    print(f"Train engines: {d['n_train_engines']}, Val engines: {d['n_val_engines']}, Test engines: {d['n_test_engines']}")
    print(f"Train windows: {d['n_train_windows']}, Val windows: {d['n_val_windows']}, Test windows: {d['n_test_windows']}")

    X_b, y_b = next(iter(d["train_loader"]))
    print(f"Batch X shape: {X_b.shape}, y shape: {y_b.shape}")
    print(f"X range: [{X_b.min():.4f}, {X_b.max():.4f}]")
    print(f"Train RUL range: [{y_b.min():.1f}, {y_b.max():.1f}]")
    print(f"Test RUL range: [{d['y_test'].min():.1f}, {d['y_test'].max():.1f}]")
    print(f"Test X shape: {d['X_test'].shape}")
    print("FD001 OK")

    print()
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        wlen = {"FD001": 32, "FD002": 64, "FD003": 48, "FD004": 64}[subset]
        bs = {"FD001": 32, "FD002": 64, "FD003": 32, "FD004": 64}[subset]
        d = prepare_data(subset, window_length=wlen, batch_size=bs, seed=42)
        print(f"{subset}: train_engines={d['n_train_engines']}, val_engines={d['n_val_engines']}, "
              f"test_engines={d['n_test_engines']}, train_windows={d['n_train_windows']}, "
              f"test_windows={d['n_test_windows']}, y_test_range=[{d['y_test'].min():.1f},{d['y_test'].max():.1f}]")
    print("All subsets OK")

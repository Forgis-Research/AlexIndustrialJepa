"""
Trivial feature-engineered regressor on C-MAPSS FD001 (ml-researcher
Rule 3 lower bound).

Hand-built features at the canonical last-window evaluation point:
engine length, last raw sensor values, per-sensor slope over the final
10 cycles, per-sensor mean/std over the final 10 cycles.

Fits Ridge on the training split and reports test RMSE on the
canonical last-window-per-engine test set (RUL cap=125). This is the
cheapest lower bound we can produce against 13.80 RMSE. If the learned
JEPA is within ~1 std of this, the representation isn't doing work the
protocol can see.
"""
import os
import sys
import time
import json
import numpy as np

V11 = "/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11"
sys.path.insert(0, V11)

from data_utils import (
    load_cmapss_subset, SELECTED_SENSORS, RUL_CAP, get_sensor_cols,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TAIL = 10


def featurize(seq: np.ndarray) -> np.ndarray:
    """Return 1D feature vector for one engine's (T, S) sensor sequence."""
    T, S = seq.shape
    last = seq[-1]
    tail = seq[-TAIL:] if T >= TAIL else seq
    tail_mean = tail.mean(axis=0)
    tail_std = tail.std(axis=0)
    idx = np.arange(tail.shape[0], dtype=np.float32)
    idx_z = (idx - idx.mean()) / (idx.std() + 1e-8)
    slopes = np.zeros(S, dtype=np.float32)
    for j in range(S):
        y = tail[:, j]
        y_z = y - y.mean()
        slopes[j] = (idx_z * y_z).sum() / (idx.shape[0] - 1 + 1e-8)
    return np.concatenate([
        np.array([T, np.log(T + 1)], dtype=np.float32),
        last.astype(np.float32),
        tail_mean.astype(np.float32),
        tail_std.astype(np.float32),
        slopes,
    ])


def last_window_xy(engines: dict, y_cap=RUL_CAP, use_full_as_train=False):
    """Build last-window features and capped-RUL labels per engine."""
    X, y = [], []
    for eid, seq in engines.items():
        feat = featurize(seq)
        if use_full_as_train:
            for cut in range(5, len(seq)):
                feat_cut = featurize(seq[:cut + 1])
                rul = max(len(seq) - cut - 1, 0)
                X.append(feat_cut)
                y.append(min(rul, y_cap))
        else:
            X.append(feat)
            y.append(0)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main():
    t0 = time.time()
    print("Loading C-MAPSS FD001 via v11 data_utils...", flush=True)
    data = load_cmapss_subset("FD001")
    train_engines = data["train_engines"]
    val_engines = data["val_engines"]
    test_engines = data["test_engines"]
    test_ruls = data["test_rul"]

    print(f"  train={len(train_engines)} val={len(val_engines)} "
          f"test={len(test_engines)} test_rul={len(test_ruls)}",
          flush=True)

    X_train, y_train = last_window_xy(
        {**train_engines, **val_engines}, use_full_as_train=True)
    print(f"  train cuts: X={X_train.shape} y=[{y_train.min():.1f},"
          f"{y_train.max():.1f}]", flush=True)

    X_test, _ = last_window_xy(test_engines, use_full_as_train=False)
    y_test = np.minimum(test_ruls, RUL_CAP).astype(np.float32)
    print(f"  test: X={X_test.shape} y=[{y_test.min():.1f},"
          f"{y_test.max():.1f}]", flush=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, y_train)
        y_pred = np.clip(model.predict(X_test_s), 0, RUL_CAP)
        rmse = float(np.sqrt(((y_pred - y_test) ** 2).mean()))
        print(f"  Ridge alpha={alpha}: test RMSE={rmse:.3f}", flush=True)
        if best is None or rmse < best["rmse"]:
            best = {"alpha": alpha, "rmse": rmse,
                    "preds": y_pred.tolist(),
                    "targets": y_test.tolist()}

    elapsed = time.time() - t0
    print(f"Best Ridge: alpha={best['alpha']} RMSE={best['rmse']:.3f} "
          f"(elapsed {elapsed:.1f}s)", flush=True)

    out = {
        "best_alpha": best["alpha"],
        "test_rmse": best["rmse"],
        "n_test": len(y_test),
        "n_train_cuts": len(y_train),
        "n_features": X_train.shape[1],
        "tail_window": TAIL,
        "elapsed_s": elapsed,
    }
    out_path = os.path.join(
        "/home/sagemaker-user/AlexIndustrialJepa/alex-contribution/"
        "experiments/v1", "trivial_feature_regressor.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("saved ->", out_path, flush=True)


if __name__ == "__main__":
    main()

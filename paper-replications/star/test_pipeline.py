"""
Quick pipeline validation for STAR replication.

Checks:
  1. Forward pass on random (2, 32, 14) input -> output shape (2,), no NaN
  2. 5 epochs on real FD001 data -> training loss strictly decreases (first to last)
  3. Parameter count printed (should be 1k-1M range reasonably)
  4. Final eval returns numeric RMSE and Score
"""

import sys
import torch
import numpy as np

from models import build_model, count_parameters
from data_utils import prepare_data
from train_utils import train_one_epoch, evaluate


def test_forward_pass():
    print("=== Test 1: Forward pass ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model("FD001", device)
    n_params = count_parameters(model)
    print(f"  FD001 parameters: {n_params:,}")

    x = torch.randn(2, 32, 14, device=device)
    out = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    print(f"  Output values: {out.tolist()}")
    print("  PASS")


def test_loss_decreases():
    print("\n=== Test 2: Loss decreases over 5 epochs ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model("FD001", device)

    data = prepare_data("FD001", window_length=32, batch_size=32, seed=42)
    train_loader = data["train_loader"]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    losses = []
    for ep in range(5):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        losses.append(loss)
        print(f"  Epoch {ep+1}: loss={loss:.5f}")

    # Check that loss at end < loss at start (not necessarily monotone but should trend down)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.5f} -> {losses[-1]:.5f}"
    )
    print(f"  Loss decreased: {losses[0]:.5f} -> {losses[-1]:.5f}")
    print("  PASS")
    return model, data


def test_evaluation(model, data):
    print("\n=== Test 3: Evaluation returns numeric metrics ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rmse, score, preds, trues = evaluate(model, data["test_loader"], device)
    print(f"  Test RMSE: {rmse:.3f}")
    print(f"  PHM Score: {score:.1f}")
    print(f"  Predictions range: [{preds.min():.1f}, {preds.max():.1f}]")
    print(f"  True RUL range: [{trues.min():.1f}, {trues.max():.1f}]")
    assert np.isfinite(rmse), f"RMSE is not finite: {rmse}"
    assert np.isfinite(score), f"Score is not finite: {score}"
    print("  PASS")


def test_all_subsets_forward():
    print("\n=== Test 4: All subsets forward pass ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    configs = {
        "FD001": dict(T=32),
        "FD002": dict(T=64),
        "FD003": dict(T=48),
        "FD004": dict(T=64),
    }
    for subset, cfg in configs.items():
        model = build_model(subset, device)
        n_params = count_parameters(model)
        x = torch.randn(2, cfg["T"], 14, device=device)
        out = model(x)
        assert out.shape == (2,), f"{subset}: Expected (2,), got {out.shape}"
        assert not torch.isnan(out).any(), f"{subset}: NaN in output!"
        print(f"  {subset}: {n_params:,} params, output OK")
    print("  PASS")


if __name__ == "__main__":
    print("STAR Pipeline Test")
    print("==================")

    try:
        test_forward_pass()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    try:
        model, data = test_loss_decreases()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    try:
        test_evaluation(model, data)
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    try:
        test_all_subsets_forward()
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("\n==================")
    print("ALL TESTS PASSED")

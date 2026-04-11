"""
Training loop and evaluation utilities for STAR replication.

Training:
  - Adam optimizer
  - CosineAnnealingLR scheduler
  - Early stopping on val RMSE (patience=20)
  - MSE loss on normalized RUL (target / 125)

Evaluation:
  - RMSE in cycles (denormalized)
  - PHM 2008 asymmetric score
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

RUL_CAP = 125.0


def compute_phm_score(pred: np.ndarray, true: np.ndarray) -> float:
    """PHM 2008 asymmetric scoring function. d = pred - true."""
    d = pred - true
    return float(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1).sum())


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.
    Returns (rmse, phm_score, preds_cycles, trues_cycles).
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            out = model(X)  # (B,) in [0, 1]
            pred_cycles = out.cpu().numpy() * RUL_CAP
            preds.append(pred_cycles)
            trues.append(y.numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    # Clamp predictions to [0, RUL_CAP]
    preds = np.clip(preds, 0, RUL_CAP)
    rmse = compute_rmse(preds, trues)
    score = compute_phm_score(preds, trues)
    return rmse, score, preds, trues


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.MSELoss()
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)  # (B,) in [0, 1]
        target_norm = y / RUL_CAP
        loss = criterion(pred, target_norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
        n += len(X)
    return total_loss / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    checkpoint_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Full training loop with early stopping.

    Returns dict with:
        best_val_rmse, train_losses, val_rmses, best_epoch, total_time
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_rmse = float("inf")
    best_epoch = 0
    no_improve = 0
    train_losses = []
    val_rmses = []
    t0 = time.time()

    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_rmse, val_score, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            no_improve = 0
            # Save best state in memory
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1 or no_improve == 0):
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.5f}, val_rmse={val_rmse:.3f}, "
                  f"best={best_val_rmse:.3f} (ep{best_epoch}), elapsed={elapsed:.0f}s")

        if no_improve >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "epochs_run": len(train_losses),
        "train_losses": train_losses,
        "val_rmses": val_rmses,
        "total_time": total_time,
    }

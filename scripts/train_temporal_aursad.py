#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Train and evaluate Temporal Self-Prediction on AURSAD dataset.

This script:
1. Trains temporal predictor on healthy data only
2. Evaluates anomaly detection on test set (healthy + faults)
3. Computes ROC-AUC, PR-AUC, and per-fault-type metrics
4. Optionally compares with static prediction baseline

Usage:
    python scripts/train_temporal_aursad.py --epochs 50
    python scripts/train_temporal_aursad.py --epochs 50 --compare-baseline
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
    setpoint_masks = torch.stack([item[2]["setpoint_mask"] for item in batch])
    effort_masks = torch.stack([item[2]["effort_mask"] for item in batch])

    return {
        "setpoint": setpoints,
        "effort": efforts,
        "setpoint_mask": setpoint_masks,
        "effort_mask": effort_masks,
        "is_anomaly": [item[2]["is_anomaly"] for item in batch],
        "fault_type": [item[2]["fault_type"] for item in batch],
        "phase": [item[2]["phase"] for item in batch],
    }


def create_dataloaders(args):
    """Create train/val/test dataloaders."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=args.window_size,
        stride=args.stride,
        normalize=True,
        norm_mode="global",  # Preserve magnitude for anomaly detection
        train_healthy_only=True,  # One-class: train on healthy only
        aursad_phase_handling="tightening_only",  # Focus on where faults occur
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    train_ds = FactoryNetDataset(config, split="train")
    val_ds = FactoryNetDataset(config, split="val")
    test_ds = FactoryNetDataset(config, split="test")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def create_temporal_model(args):
    """Create temporal predictor model."""
    from industrialjepa.baselines import TemporalPredictor, TemporalConfig

    config = TemporalConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=args.window_size,
        patch_size=16,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=8,
        dropout=0.1,
        context_ratio=args.context_ratio,
        prediction_mode=args.mode,
        ema_decay=0.996,
    )

    return TemporalPredictor(config), config


def create_baseline_model(args):
    """Create static SetpointToEffort baseline for comparison."""
    from industrialjepa.baselines import SetpointToEffort, AutoencoderConfig

    config = AutoencoderConfig(
        setpoint_dim=14,
        effort_dim=13,
        seq_len=args.window_size,
        patch_size=16,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=8,
        dropout=0.1,
    )

    return SetpointToEffort(config), config


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)
        setpoint_mask = batch["setpoint_mask"].to(device)
        effort_mask = batch["effort_mask"].to(device)

        optimizer.zero_grad()
        output = model(setpoint, effort, setpoint_mask, effort_mask)
        loss = output["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update for temporal predictor
        if hasattr(model, 'update_ema'):
            model.update_ema()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())

    return total_loss / num_batches


@torch.no_grad()
def evaluate_anomaly_detection(model, test_loader, device):
    """
    Evaluate anomaly detection performance.

    Returns:
        Dict with ROC-AUC, PR-AUC, per-fault metrics
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_fault_types = []

    for batch in tqdm(test_loader, desc="Evaluating"):
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)
        setpoint_mask = batch["setpoint_mask"].to(device)
        effort_mask = batch["effort_mask"].to(device)

        # Compute anomaly scores
        scores = model.compute_anomaly_score(
            setpoint, effort, setpoint_mask, effort_mask
        )

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend([1 if a else 0 for a in batch["is_anomaly"]])
        all_fault_types.extend(batch["fault_type"])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Overall metrics
    results = {}

    if len(np.unique(all_labels)) > 1:
        results["roc_auc"] = roc_auc_score(all_labels, all_scores)
        results["pr_auc"] = average_precision_score(all_labels, all_scores)

        # Find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        results["optimal_threshold"] = thresholds[best_idx]
        results["tpr_at_optimal"] = tpr[best_idx]
        results["fpr_at_optimal"] = fpr[best_idx]
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    # Per-fault-type analysis
    fault_types = set(all_fault_types)
    results["per_fault"] = {}

    for fault in fault_types:
        mask = np.array([ft == fault for ft in all_fault_types])
        fault_scores = all_scores[mask]
        fault_labels = all_labels[mask]

        results["per_fault"][fault] = {
            "count": int(mask.sum()),
            "mean_score": float(fault_scores.mean()),
            "std_score": float(fault_scores.std()),
            "anomaly_rate": float(fault_labels.mean()) if len(fault_labels) > 0 else 0,
        }

    # Score statistics
    results["score_stats"] = {
        "normal_mean": float(all_scores[all_labels == 0].mean()) if (all_labels == 0).any() else None,
        "normal_std": float(all_scores[all_labels == 0].std()) if (all_labels == 0).any() else None,
        "anomaly_mean": float(all_scores[all_labels == 1].mean()) if (all_labels == 1).any() else None,
        "anomaly_std": float(all_scores[all_labels == 1].std()) if (all_labels == 1).any() else None,
    }

    # Label distribution
    results["label_distribution"] = {
        "total": len(all_labels),
        "normal": int((all_labels == 0).sum()),
        "anomaly": int((all_labels == 1).sum()),
    }

    return results


def print_results(results, model_name):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")

    print(f"\nLabel Distribution:")
    print(f"  Total samples: {results['label_distribution']['total']}")
    print(f"  Normal: {results['label_distribution']['normal']}")
    print(f"  Anomaly: {results['label_distribution']['anomaly']}")

    print(f"\nOverall Metrics:")
    if results["roc_auc"] is not None:
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  PR-AUC:  {results['pr_auc']:.4f}")
        print(f"  Optimal threshold: {results['optimal_threshold']:.4f}")
        print(f"  TPR at optimal: {results['tpr_at_optimal']:.4f}")
        print(f"  FPR at optimal: {results['fpr_at_optimal']:.4f}")
    else:
        print("  (Cannot compute - only one class present)")

    print(f"\nScore Statistics:")
    stats = results["score_stats"]
    if stats["normal_mean"] is not None:
        print(f"  Normal:  {stats['normal_mean']:.4f} ± {stats['normal_std']:.4f}")
    if stats["anomaly_mean"] is not None:
        print(f"  Anomaly: {stats['anomaly_mean']:.4f} ± {stats['anomaly_std']:.4f}")

    print(f"\nPer-Fault Analysis:")
    for fault, data in sorted(results["per_fault"].items()):
        print(f"  {fault}:")
        print(f"    Count: {data['count']}, Score: {data['mean_score']:.4f} ± {data['std_score']:.4f}")


def main():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--mode", type=str, default="jepa",
                        choices=["jepa", "direct"],
                        help="Temporal prediction mode")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--context-ratio", type=float, default=0.5)

    # Data
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Comparison
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Also train and compare with static S2E baseline")

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"temporal_aursad_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    logger.info("Loading AURSAD dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(args)
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # =========================================================================
    # Train Temporal Predictor
    # =========================================================================
    logger.info(f"\nTraining Temporal Predictor ({args.mode} mode)...")
    model, config = create_temporal_model(args)
    model = model.to(device)
    logger.info(f"Parameters: {model.get_num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                setpoint = batch["setpoint"].to(device)
                effort = batch["effort"].to(device)
                setpoint_mask = batch["setpoint_mask"].to(device)
                effort_mask = batch["effort_mask"].to(device)
                output = model(setpoint, effort, setpoint_mask, effort_mask)
                val_loss += output["loss"].item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "config": config,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best_temporal.pt")

    # Evaluate temporal predictor
    logger.info("\nEvaluating Temporal Predictor on test set...")
    checkpoint = torch.load(output_dir / "best_temporal.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    temporal_results = evaluate_anomaly_detection(model, test_loader, device)
    temporal_results["train_losses"] = train_losses
    temporal_results["val_losses"] = val_losses
    temporal_results["best_epoch"] = checkpoint["epoch"]

    print_results(temporal_results, f"Temporal Predictor ({args.mode})")

    # Save results
    with open(output_dir / "temporal_results.json", "w") as f:
        json.dump(temporal_results, f, indent=2)

    # =========================================================================
    # Compare with Baseline (optional)
    # =========================================================================
    if args.compare_baseline:
        logger.info("\n" + "="*60)
        logger.info("Training Static Baseline (SetpointToEffort) for comparison...")
        logger.info("="*60)

        baseline_model, baseline_config = create_baseline_model(args)
        baseline_model = baseline_model.to(device)
        logger.info(f"Baseline parameters: {baseline_model.get_num_params():,}")

        baseline_optimizer = torch.optim.AdamW(
            baseline_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            baseline_optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        best_baseline_val = float("inf")

        for epoch in range(args.epochs):
            # Train baseline
            baseline_model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Baseline Epoch {epoch}"):
                setpoint = batch["setpoint"].to(device)
                effort = batch["effort"].to(device)
                setpoint_mask = batch["setpoint_mask"].to(device)
                effort_mask = batch["effort_mask"].to(device)

                baseline_optimizer.zero_grad()
                output = baseline_model(setpoint, effort, setpoint_mask, effort_mask)
                loss = output["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
                baseline_optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)

            # Validate
            baseline_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    setpoint = batch["setpoint"].to(device)
                    effort = batch["effort"].to(device)
                    setpoint_mask = batch["setpoint_mask"].to(device)
                    effort_mask = batch["effort_mask"].to(device)
                    output = baseline_model(setpoint, effort, setpoint_mask, effort_mask)
                    val_loss += output["loss"].item()
            val_loss /= len(val_loader)

            baseline_scheduler.step()

            logger.info(f"Baseline Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

            if val_loss < best_baseline_val:
                best_baseline_val = val_loss
                torch.save({
                    "config": baseline_config,
                    "state_dict": baseline_model.state_dict(),
                }, output_dir / "best_baseline.pt")

        # Evaluate baseline
        logger.info("\nEvaluating Baseline on test set...")
        checkpoint = torch.load(output_dir / "best_baseline.pt", map_location=device, weights_only=False)
        baseline_model.load_state_dict(checkpoint["state_dict"])

        baseline_results = evaluate_anomaly_detection(baseline_model, test_loader, device)
        print_results(baseline_results, "Static Baseline (SetpointToEffort)")

        with open(output_dir / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2)

        # Print comparison
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<25} {'Temporal':>15} {'Baseline':>15} {'Δ':>10}")
        print("-"*60)

        if temporal_results["roc_auc"] and baseline_results["roc_auc"]:
            t_roc = temporal_results["roc_auc"]
            b_roc = baseline_results["roc_auc"]
            print(f"{'ROC-AUC':<25} {t_roc:>15.4f} {b_roc:>15.4f} {t_roc-b_roc:>+10.4f}")

            t_pr = temporal_results["pr_auc"]
            b_pr = baseline_results["pr_auc"]
            print(f"{'PR-AUC':<25} {t_pr:>15.4f} {b_pr:>15.4f} {t_pr-b_pr:>+10.4f}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

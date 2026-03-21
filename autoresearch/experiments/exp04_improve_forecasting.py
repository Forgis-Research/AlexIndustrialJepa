#!/usr/bin/env python
"""
Experiment 04: Improve Many-to-1 Forecasting Transfer

Problem: AURSAD held-out transfer ratio is 1.71 (target ≤ 1.5).
When training on Voraus → AURSAD, model undertransfers.

Systematic ablation:
A) Baseline: reproduce exp02 settings (15 epochs, 128 hidden, 500 episodes)
B) More data: 1000 Voraus episodes (was 500)
C) More epochs: 40 epochs (was 15)
D) Larger model: 256 hidden, 4 layers (was 128, 3)
E) Combined: best of B+C+D
F) No RevIN ablation: test if RevIN helps or hurts

Focus: AURSAD held-out (hardest case). Voraus held-out already passes.
"""

import sys
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "autoresearch" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Data Loading
# ============================================================

def load_source(source_name, split="train", window_size=256, stride=128,
                norm_mode="episode", max_episodes=None):
    """Load a single data source."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    config = FactoryNetConfig(
        data_source=source_name,
        max_episodes=max_episodes,
        window_size=window_size,
        stride=stride,
        normalize=True,
        norm_mode=norm_mode,
        effort_signals=["voltage", "current"],
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12,
        unified_effort_dim=6,
    )

    ds = FactoryNetDataset(config, split=split)
    return ds


def load_source_with_shared(source_name, splits=("train", "val", "test"), **kwargs):
    """Load a source across splits, sharing data for memory efficiency."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    datasets = {}
    shared = None
    for split in splits:
        try:
            if shared is None:
                ds = load_source(source_name, split=split, **kwargs)
                shared = ds.get_shared_data()
            else:
                config = FactoryNetConfig(
                    data_source=source_name,
                    max_episodes=kwargs.get("max_episodes"),
                    window_size=kwargs.get("window_size", 256),
                    stride=kwargs.get("stride", 128),
                    normalize=True,
                    norm_mode=kwargs.get("norm_mode", "episode"),
                    effort_signals=["voltage", "current"],
                    setpoint_signals=["position", "velocity"],
                    unified_setpoint_dim=12,
                    unified_effort_dim=6,
                )
                ds = FactoryNetDataset(config, split=split, shared_data=shared)
            datasets[split] = ds
            logger.info(f"  {source_name}/{split}: {len(ds)} windows")
        except Exception as e:
            logger.warning(f"  Failed to load {source_name}/{split}: {e}")
            datasets[split] = None
    return datasets


def collate_fn(batch):
    """Custom collate function."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
    metadata = {
        "is_anomaly": [item[2].get("is_anomaly", False) for item in batch],
    }
    return setpoints, efforts, metadata


def make_loader(dataset, batch_size=64, shuffle=False):
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )


def make_combined_loader(datasets_list, batch_size=64, shuffle=True):
    """Create a combined loader from multiple datasets."""
    valid = [d for d in datasets_list if d is not None and len(d) > 0]
    if not valid:
        return None
    combined = ConcatDataset(valid)
    return DataLoader(
        combined, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )


# ============================================================
# Models (same as exp02 but parameterized)
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class MultiSourceForecaster(nn.Module):
    """Forecaster with RevIN for cross-domain generalization."""

    def __init__(self, input_dim=18, hidden_dim=128, num_layers=3,
                 num_heads=4, context_len=128, forecast_len=128,
                 use_revin=True, dropout=0.1):
        super().__init__()
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.input_dim = input_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(input_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(
            torch.randn(1, context_len, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_full):
        B, T, D = x_full.shape

        if self.use_revin:
            x_full = self.revin(x_full, mode='norm')

        context = x_full[:, :self.context_len, :]
        target = x_full[:, self.context_len:self.context_len + self.forecast_len, :]

        h = self.input_proj(context) + self.pos_enc[:, :context.shape[1], :]
        h = self.transformer(h)
        h = self.norm(h)
        pred = self.forecast_head(h)

        if pred.shape[1] > self.forecast_len:
            pred = pred[:, -self.forecast_len:, :]
        elif pred.shape[1] < self.forecast_len:
            pad = pred[:, -1:, :].expand(-1, self.forecast_len - pred.shape[1], -1)
            pred = torch.cat([pred, pad], dim=1)

        if self.use_revin:
            pred_denorm = self.revin(pred, mode='denorm')
            target_denorm = self.revin(target, mode='denorm')
            loss = F.mse_loss(pred_denorm, target_denorm)
        else:
            loss = F.mse_loss(pred, target)

        return {"loss": loss, "pred": pred, "target": target}


# ============================================================
# Training
# ============================================================

def train_forecaster(model, train_loader, val_loader, epochs, lr=1e-4,
                     device='cuda', warmup_epochs=5):
    """Train forecaster with warmup + cosine annealing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10  # Early stopping patience
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for setpoint, effort, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            setpoint = setpoint.to(device)
            effort = effort.to(device)
            x = torch.cat([setpoint, effort], dim=-1)

            optimizer.zero_grad()
            result = model(x)
            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for setpoint, effort, _ in val_loader:
                setpoint = setpoint.to(device)
                effort = effort.to(device)
                x = torch.cat([setpoint, effort], dim=-1)
                result = model(x)
                val_loss += result['loss'].item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}, lr={current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch > warmup_epochs + 5:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_loss


@torch.no_grad()
def evaluate_forecasting(model, test_loader, device='cuda'):
    """Evaluate forecasting MSE."""
    model.eval()
    total_mse = 0
    n_batches = 0

    for setpoint, effort, _ in test_loader:
        setpoint = setpoint.to(device)
        effort = effort.to(device)
        x = torch.cat([setpoint, effort], dim=-1)
        result = model(x)
        total_mse += result['loss'].item()
        n_batches += 1

    return total_mse / max(n_batches, 1)


# ============================================================
# Ablation Configs
# ============================================================

ABLATIONS = {
    "A_baseline": {
        "desc": "Reproduce exp02 baseline",
        "epochs": 15,
        "hidden_dim": 128,
        "num_layers": 3,
        "max_episodes_voraus": 500,
        "max_episodes_cnc": None,
        "use_revin": True,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
    "B_more_data": {
        "desc": "More Voraus episodes (1000 vs 500)",
        "epochs": 15,
        "hidden_dim": 128,
        "num_layers": 3,
        "max_episodes_voraus": 1000,
        "max_episodes_cnc": None,
        "use_revin": True,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
    "C_more_epochs": {
        "desc": "More training (40 epochs with warmup)",
        "epochs": 40,
        "hidden_dim": 128,
        "num_layers": 3,
        "max_episodes_voraus": 500,
        "max_episodes_cnc": None,
        "use_revin": True,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
    "D_larger_model": {
        "desc": "Larger model (256 hidden, 4 layers)",
        "epochs": 15,
        "hidden_dim": 256,
        "num_layers": 4,
        "max_episodes_voraus": 500,
        "max_episodes_cnc": None,
        "use_revin": True,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
    "E_combined": {
        "desc": "Best of all: more data + epochs + larger model",
        "epochs": 40,
        "hidden_dim": 256,
        "num_layers": 4,
        "max_episodes_voraus": 1000,
        "max_episodes_cnc": None,
        "use_revin": True,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
    "F_no_revin": {
        "desc": "No RevIN (test if it helps or hurts)",
        "epochs": 40,
        "hidden_dim": 256,
        "num_layers": 4,
        "max_episodes_voraus": 1000,
        "max_episodes_cnc": None,
        "use_revin": False,
        "lr": 1e-4,
        "batch_size": 64,
        "dropout": 0.1,
    },
}


def run_ablation(ablation_name, config, seed=42, held_out="aursad"):
    """Run a single ablation for AURSAD held-out."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION {ablation_name}: {config['desc']}")
    logger.info(f"{'='*60}")

    # Load data
    # Training sources: everything except held-out
    train_sources_config = {
        "voraus": config["max_episodes_voraus"],
        "cnc": config["max_episodes_cnc"],
    }
    if held_out != "aursad":
        train_sources_config["aursad"] = 500
        if held_out in train_sources_config:
            del train_sources_config[held_out]

    logger.info(f"Held-out: {held_out}")
    logger.info(f"Training sources: {list(train_sources_config.keys())}")

    # Load all needed sources
    all_data = {}
    for source_name, max_ep in train_sources_config.items():
        logger.info(f"Loading {source_name} (max_episodes={max_ep})...")
        try:
            all_data[source_name] = load_source_with_shared(
                source_name, max_episodes=max_ep,
                window_size=256, stride=128, norm_mode="episode",
            )
        except Exception as e:
            logger.warning(f"Failed to load {source_name}: {e}")

    # Load held-out source
    logger.info(f"Loading held-out {held_out}...")
    held_out_max = 500 if held_out in ["aursad", "voraus"] else None
    all_data[held_out] = load_source_with_shared(
        held_out, max_episodes=held_out_max,
        window_size=256, stride=128, norm_mode="episode",
    )

    # Build training loaders
    train_sources = [s for s in train_sources_config if s in all_data
                     and all_data[s].get('train') is not None
                     and len(all_data[s]['train']) > 0]
    logger.info(f"Successfully loaded train sources: {train_sources}")

    if not train_sources:
        logger.error("No training sources available!")
        return None

    train_datasets = [all_data[s]['train'] for s in train_sources]
    val_datasets = [all_data[s].get('val') or all_data[s]['train'] for s in train_sources]

    combined_train = make_combined_loader(train_datasets, batch_size=config["batch_size"])
    combined_val = make_combined_loader(val_datasets, batch_size=config["batch_size"], shuffle=False)

    # Test loaders
    target_test = make_loader(all_data[held_out].get('test'), batch_size=config["batch_size"])
    source_test_loaders = {}
    for s in train_sources:
        loader = make_loader(all_data[s].get('test'), batch_size=config["batch_size"])
        if loader is not None:
            source_test_loaders[s] = loader

    if target_test is None:
        logger.error(f"No test data for held-out {held_out}!")
        return None

    # Train forecaster
    forecaster = MultiSourceForecaster(
        input_dim=18, hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], num_heads=4,
        context_len=128, forecast_len=128,
        use_revin=config["use_revin"],
        dropout=config["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in forecaster.parameters())
    logger.info(f"Model params: {n_params:,}")

    history, best_val = train_forecaster(
        forecaster, combined_train, combined_val,
        epochs=config["epochs"], lr=config["lr"],
        device=device, warmup_epochs=min(5, config["epochs"] // 4),
    )

    # Evaluate
    source_mses = {}
    for s, loader in source_test_loaders.items():
        mse = evaluate_forecasting(forecaster, loader, device)
        source_mses[s] = mse
        logger.info(f"  Source {s} MSE: {mse:.6f}")

    avg_source_mse = np.mean(list(source_mses.values())) if source_mses else 0
    target_mse = evaluate_forecasting(forecaster, target_test, device)
    transfer_ratio = target_mse / max(avg_source_mse, 1e-8)

    logger.info(f"  Target {held_out} MSE: {target_mse:.6f}")
    logger.info(f"  Avg Source MSE: {avg_source_mse:.6f}")
    logger.info(f"  Transfer Ratio: {transfer_ratio:.4f}")
    logger.info(f"  {'PASS' if transfer_ratio <= 1.5 else 'FAIL'} (target ≤ 1.5)")

    return {
        "ablation": ablation_name,
        "config": config,
        "held_out": held_out,
        "train_sources": train_sources,
        "n_params": n_params,
        "history": history,
        "best_val_loss": best_val,
        "source_mses": source_mses,
        "target_mse": target_mse,
        "avg_source_mse": avg_source_mse,
        "transfer_ratio": transfer_ratio,
        "passed": transfer_ratio <= 1.5,
    }


def plot_ablation_results(all_results, seed, held_out):
    """Plot ablation comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    names = []
    ratios = []
    val_losses = []
    for name, result in all_results.items():
        if result is None:
            continue
        names.append(name.split('_')[0])  # Just the letter
        ratios.append(result['transfer_ratio'])
        val_losses.append(result['best_val_loss'])

    # 1. Transfer ratios
    ax = axes[0]
    colors = ['green' if r <= 1.5 else 'red' for r in ratios]
    bars = ax.bar(names, ratios, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='Target ≤ 1.5')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Transfer Ratio')
    ax.set_title(f'AURSAD Held-Out Transfer Ratio\n(seed={seed})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation losses
    ax = axes[1]
    ax.bar(names, val_losses, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Model Fit (lower is better)')
    ax.grid(True, alpha=0.3)

    # 3. Training curves for all ablations
    ax = axes[2]
    for name, result in all_results.items():
        if result is None:
            continue
        short_name = name.split('_')[0]
        ax.plot(result['history']['val_loss'], label=short_name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Exp04: Forecasting Ablation Study (held-out={held_out})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = FIGURES_DIR / f"exp04_ablation_seed{seed}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def run_multi_seed(best_ablation_name, best_config, seeds=[42, 123, 456]):
    """Run the best ablation across 3 seeds for statistical significance."""
    results = []
    for seed in seeds:
        logger.info(f"\n{'#'*60}")
        logger.info(f"MULTI-SEED: {best_ablation_name}, seed={seed}")
        logger.info(f"{'#'*60}")
        r = run_ablation(best_ablation_name, best_config, seed=seed, held_out="aursad")
        if r is not None:
            results.append(r)

    if not results:
        return None

    ratios = [r['transfer_ratio'] for r in results]
    logger.info(f"\nMulti-seed results for {best_ablation_name}:")
    logger.info(f"  Transfer ratios: {ratios}")
    logger.info(f"  Mean: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    logger.info(f"  All pass: {all(r <= 1.5 for r in ratios)}")

    return {
        "ablation": best_ablation_name,
        "seeds": seeds,
        "ratios": ratios,
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "all_pass": all(r <= 1.5 for r in ratios),
        "per_seed": results,
    }


def run_full_loo(config, seed=42):
    """Run full leave-one-out with best config on both held-outs."""
    results = {}
    for held_out in ["aursad", "voraus"]:
        logger.info(f"\n{'#'*60}")
        logger.info(f"FULL LOO: held-out={held_out}, seed={seed}")
        logger.info(f"{'#'*60}")
        r = run_ablation("best", config, seed=seed, held_out=held_out)
        if r is not None:
            results[held_out] = r
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", type=str, default=None,
                        help="Run specific ablation (A,B,C,D,E,F) or 'all'")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run best config with 3 seeds")
    parser.add_argument("--full-loo", action="store_true",
                        help="Run full LOO with best config")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.multi_seed:
        # Run multi-seed on combined config (E)
        ms_results = run_multi_seed("E_combined", ABLATIONS["E_combined"])
        with open(RESULTS_DIR / f"exp04_multiseed_{timestamp}.json", 'w') as f:
            json.dump(ms_results, f, indent=2, default=str)

    elif args.full_loo:
        # Run full LOO with combined config
        loo_results = run_full_loo(ABLATIONS["E_combined"], seed=args.seed)
        with open(RESULTS_DIR / f"exp04_loo_seed{args.seed}_{timestamp}.json", 'w') as f:
            json.dump(loo_results, f, indent=2, default=str)

    elif args.ablation:
        # Run specific ablation(s)
        if args.ablation.lower() == 'all':
            ablations_to_run = ABLATIONS
        else:
            key = [k for k in ABLATIONS if k.startswith(args.ablation.upper())]
            if key:
                ablations_to_run = {key[0]: ABLATIONS[key[0]]}
            else:
                logger.error(f"Unknown ablation: {args.ablation}")
                sys.exit(1)

        all_results = {}
        for name, config in ablations_to_run.items():
            result = run_ablation(name, config, seed=args.seed, held_out="aursad")
            all_results[name] = result

        # Plot
        plot_ablation_results(all_results, args.seed, "aursad")

        # Save
        with open(RESULTS_DIR / f"exp04_ablation_seed{args.seed}_{timestamp}.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Summary
        print("\n" + "=" * 80)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Ablation':<20} {'Ratio':<10} {'Val Loss':<12} {'Pass?':<8} {'Description'}")
        print("-" * 80)
        for name, result in all_results.items():
            if result is None:
                print(f"{name:<20} {'FAILED':<10}")
                continue
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{name:<20} {result['transfer_ratio']:<10.4f} "
                  f"{result['best_val_loss']:<12.6f} {status:<8} {result['config']['desc']}")
        print("=" * 80)

    else:
        # Default: run all ablations
        all_results = {}
        for name, config in ABLATIONS.items():
            result = run_ablation(name, config, seed=args.seed, held_out="aursad")
            all_results[name] = result

        plot_ablation_results(all_results, args.seed, "aursad")

        with open(RESULTS_DIR / f"exp04_ablation_seed{args.seed}_{timestamp}.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # Summary
        print("\n" + "=" * 80)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Ablation':<20} {'Ratio':<10} {'Val Loss':<12} {'Pass?':<8} {'Description'}")
        print("-" * 80)
        for name, result in all_results.items():
            if result is None:
                print(f"{name:<20} {'FAILED':<10}")
                continue
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{name:<20} {result['transfer_ratio']:<10.4f} "
                  f"{result['best_val_loss']:<12.6f} {status:<8} {result['config']['desc']}")
        print("=" * 80)

        # Find best and report
        best_name = min(
            [k for k, v in all_results.items() if v is not None],
            key=lambda k: all_results[k]['transfer_ratio']
        )
        best = all_results[best_name]
        print(f"\nBest: {best_name} (ratio={best['transfer_ratio']:.4f})")
        if best['passed']:
            print("OBJECTIVE ACHIEVED! Running multi-seed validation...")

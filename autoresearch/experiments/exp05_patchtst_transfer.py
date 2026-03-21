#!/usr/bin/env python
"""
Experiment 05: PatchTST-Style Channel-Independent Forecasting

Key insight from exp04: bigger model + more epochs didn't help (ratio still 1.72).
The problem is fundamental: joint forecasting of all 18 channels couples
domain-specific cross-channel correlations.

PatchTST approach:
1. Channel-independent: each channel is forecast independently
2. Patch embedding: local temporal patches capture transferable dynamics
3. Per-channel RevIN: normalize each channel independently

Hypothesis: By forecasting channels independently, the model learns
per-channel temporal dynamics (which are more universal across robots)
rather than cross-channel correlations (which are task-specific).

Also testing:
- Smaller stride (64 vs 128) for more training data
- Stronger regularization (dropout=0.2, weight_decay=0.05)
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

def load_source(source_name, split="train", window_size=256, stride=64,
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
    return FactoryNetDataset(config, split=split)


def load_source_with_shared(source_name, splits=("train", "val", "test"), **kwargs):
    """Load a source across splits, sharing data."""
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
                    stride=kwargs.get("stride", 64),
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
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
    metadata = {"is_anomaly": [item[2].get("is_anomaly", False) for item in batch]}
    return setpoints, efforts, metadata


def make_loader(dataset, batch_size=64, shuffle=False):
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False)


def make_combined_loader(datasets_list, batch_size=64, shuffle=True):
    valid = [d for d in datasets_list if d is not None and len(d) > 0]
    if not valid:
        return None
    combined = ConcatDataset(valid)
    return DataLoader(combined, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)


# ============================================================
# PatchTST-Style Model
# ============================================================

class PatchTSTForecaster(nn.Module):
    """
    Channel-independent patch-based forecaster inspired by PatchTST.

    Key differences from MultiSourceForecaster (exp02):
    1. Channel-independent: Each of D channels is processed independently
    2. Patch embedding: Time series is split into non-overlapping patches
    3. Shared backbone: Same transformer weights applied to all channels
    4. Per-channel RevIN: Normalize each channel independently

    This forces the model to learn per-channel temporal dynamics rather than
    cross-channel correlations, which should transfer better across robots.
    """

    def __init__(self, n_channels=18, patch_len=16, stride_patch=8,
                 d_model=128, n_heads=4, n_layers=3, d_ff=512,
                 dropout=0.2, context_len=128, forecast_len=128,
                 use_revin=True):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride_patch = stride_patch
        self.d_model = d_model
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.use_revin = use_revin

        # Number of patches in context
        self.n_patches = (context_len - patch_len) // stride_patch + 1

        # Per-channel RevIN
        if use_revin:
            self.revin_mean = None
            self.revin_std = None

        # Patch embedding (shared across channels)
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Shared transformer backbone
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Forecast head: map from patch representations to forecast
        self.forecast_head = nn.Sequential(
            nn.Flatten(start_dim=-2),  # (B*C, n_patches * d_model)
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, forecast_len),
        )

    def _revin_norm(self, x):
        """Per-channel RevIN normalization. x: (B, T, C)"""
        self.revin_mean = x.mean(dim=1, keepdim=True).detach()  # (B, 1, C)
        self.revin_std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (x - self.revin_mean) / self.revin_std

    def _revin_denorm(self, x):
        """Per-channel RevIN denormalization. x: (B, T, C)"""
        return x * self.revin_std + self.revin_mean

    def _create_patches(self, x):
        """
        Create patches from time series.
        x: (B*C, T) → (B*C, n_patches, patch_len)
        """
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride_patch)
        return patches  # (B*C, n_patches, patch_len)

    def forward(self, x_full):
        """
        x_full: (B, T, D) - full sequence (context + forecast)
        """
        B, T, D = x_full.shape

        # RevIN normalize per-channel
        if self.use_revin:
            x_full = self._revin_norm(x_full)

        # Split context and target
        context = x_full[:, :self.context_len, :]  # (B, ctx_len, D)
        target = x_full[:, self.context_len:self.context_len + self.forecast_len, :]  # (B, fct_len, D)

        # Channel-independent processing
        # Reshape: (B, T, D) → (B*D, T)
        context_ci = context.permute(0, 2, 1).reshape(B * D, self.context_len)

        # Create patches: (B*D, n_patches, patch_len)
        patches = self._create_patches(context_ci)

        # Embed patches: (B*D, n_patches, d_model)
        h = self.patch_embed(patches) + self.pos_enc

        # Transformer: (B*D, n_patches, d_model)
        h = self.transformer(h)
        h = self.norm(h)

        # Forecast: (B*D, forecast_len)
        pred_flat = self.forecast_head(h)

        # Reshape back: (B, D, forecast_len) → (B, forecast_len, D)
        pred = pred_flat.reshape(B, D, self.forecast_len).permute(0, 2, 1)

        # Compute loss
        if self.use_revin:
            pred_denorm = self._revin_denorm(pred)
            target_denorm = self._revin_denorm(target)
            loss = F.mse_loss(pred_denorm, target_denorm)
        else:
            loss = F.mse_loss(pred, target)

        return {"loss": loss, "pred": pred, "target": target}


class JointForecasterWithReg(nn.Module):
    """
    Joint forecaster (like exp02) but with stronger regularization.
    Used as a control to separate the effect of regularization from
    the effect of channel independence.
    """

    def __init__(self, input_dim=18, hidden_dim=128, num_layers=3,
                 num_heads=4, context_len=128, forecast_len=128,
                 use_revin=True, dropout=0.2):
        super().__init__()
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.input_dim = input_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin_mean = None
            self.revin_std = None

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, context_len, hidden_dim) * 0.02)

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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def _revin_norm(self, x):
        self.revin_mean = x.mean(dim=1, keepdim=True).detach()
        self.revin_std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (x - self.revin_mean) / self.revin_std

    def _revin_denorm(self, x):
        return x * self.revin_std + self.revin_mean

    def forward(self, x_full):
        B, T, D = x_full.shape

        if self.use_revin:
            x_full = self._revin_norm(x_full)

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
            pred_denorm = self._revin_denorm(pred)
            target_denorm = self._revin_denorm(target)
            loss = F.mse_loss(pred_denorm, target_denorm)
        else:
            loss = F.mse_loss(pred, target)

        return {"loss": loss, "pred": pred, "target": target}


# ============================================================
# Training
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr=1e-4,
                device='cuda', warmup_epochs=5, weight_decay=0.05):
    """Train with warmup + cosine schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 12
    history = {'train_loss': [], 'val_loss': []}

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
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

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
# Experiment Configurations
# ============================================================

CONFIGS = {
    "patchtst_base": {
        "desc": "PatchTST channel-independent (base)",
        "model": "patchtst",
        "patch_len": 16,
        "stride_patch": 8,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.2,
        "epochs": 40,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "stride": 64,  # data stride
        "max_episodes_voraus": 1000,
    },
    "patchtst_larger": {
        "desc": "PatchTST channel-independent (larger patches)",
        "model": "patchtst",
        "patch_len": 32,
        "stride_patch": 16,
        "d_model": 128,
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.2,
        "epochs": 40,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "stride": 64,
        "max_episodes_voraus": 1000,
    },
    "joint_regularized": {
        "desc": "Joint model with stronger regularization (control)",
        "model": "joint",
        "hidden_dim": 128,
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.2,
        "epochs": 40,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "stride": 64,
        "max_episodes_voraus": 1000,
    },
}


def run_experiment(config_name, config, seed=42, held_out="aursad"):
    """Run a single experiment configuration."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"\n{'='*60}")
    logger.info(f"CONFIG {config_name}: {config['desc']}")
    logger.info(f"Seed: {seed}, Held-out: {held_out}")
    logger.info(f"{'='*60}")

    # Determine training sources
    all_sources = ["aursad", "voraus", "cnc"]
    train_source_names = [s for s in all_sources if s != held_out]

    max_episodes_map = {
        "voraus": config["max_episodes_voraus"],
        "aursad": 500,
        "cnc": None,
    }

    # Load data
    all_data = {}
    for source in train_source_names + [held_out]:
        max_ep = max_episodes_map.get(source)
        logger.info(f"Loading {source} (max_episodes={max_ep}, stride={config['stride']})...")
        try:
            all_data[source] = load_source_with_shared(
                source, max_episodes=max_ep,
                window_size=256, stride=config["stride"], norm_mode="episode",
            )
        except Exception as e:
            logger.warning(f"Failed to load {source}: {e}")

    train_sources = [s for s in train_source_names if s in all_data
                     and all_data[s].get('train') is not None
                     and len(all_data[s]['train']) > 0]
    logger.info(f"Active train sources: {train_sources}")

    if not train_sources or held_out not in all_data:
        logger.error("Insufficient data!")
        return None

    # Build loaders
    train_datasets = [all_data[s]['train'] for s in train_sources]
    val_datasets = [all_data[s].get('val') or all_data[s]['train'] for s in train_sources]

    combined_train = make_combined_loader(train_datasets, batch_size=64)
    combined_val = make_combined_loader(val_datasets, batch_size=64, shuffle=False)
    target_test = make_loader(all_data[held_out].get('test'), batch_size=64)

    source_test_loaders = {}
    for s in train_sources:
        loader = make_loader(all_data[s].get('test'), batch_size=64)
        if loader is not None:
            source_test_loaders[s] = loader

    if combined_train is None or target_test is None:
        logger.error("Missing train or test data!")
        return None

    total_train = sum(len(all_data[s]['train']) for s in train_sources)
    logger.info(f"Total training windows: {total_train}")

    # Create model
    if config["model"] == "patchtst":
        model = PatchTSTForecaster(
            n_channels=18,
            patch_len=config["patch_len"],
            stride_patch=config["stride_patch"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            dropout=config["dropout"],
            context_len=128,
            forecast_len=128,
            use_revin=True,
        ).to(device)
    else:
        model = JointForecasterWithReg(
            input_dim=18,
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config["n_layers"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            context_len=128,
            forecast_len=128,
            use_revin=True,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config['model']}, params: {n_params:,}")

    # Train
    history, best_val = train_model(
        model, combined_train, combined_val,
        epochs=config["epochs"], lr=config["lr"],
        device=device, warmup_epochs=5,
        weight_decay=config["weight_decay"],
    )

    # Evaluate
    source_mses = {}
    for s, loader in source_test_loaders.items():
        mse = evaluate_forecasting(model, loader, device)
        source_mses[s] = mse
        logger.info(f"  Source {s} MSE: {mse:.6f}")

    avg_source_mse = np.mean(list(source_mses.values())) if source_mses else 0
    target_mse = evaluate_forecasting(model, target_test, device)
    transfer_ratio = target_mse / max(avg_source_mse, 1e-8)

    logger.info(f"  Target {held_out} MSE: {target_mse:.6f}")
    logger.info(f"  Avg Source MSE: {avg_source_mse:.6f}")
    logger.info(f"  Transfer Ratio: {transfer_ratio:.4f}")
    passed = transfer_ratio <= 1.5
    logger.info(f"  {'PASS' if passed else 'FAIL'} (target ≤ 1.5)")

    return {
        "config_name": config_name,
        "config": config,
        "seed": seed,
        "held_out": held_out,
        "train_sources": train_sources,
        "total_train_windows": total_train,
        "n_params": n_params,
        "history": history,
        "best_val_loss": best_val,
        "source_mses": source_mses,
        "target_mse": target_mse,
        "avg_source_mse": avg_source_mse,
        "transfer_ratio": transfer_ratio,
        "passed": passed,
    }


def plot_comparison(all_results, seed):
    """Plot comparison of all configurations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    names = []
    ratios = []
    target_mses = []
    for name, result in all_results.items():
        if result is None:
            continue
        names.append(name.replace('_', '\n'))
        ratios.append(result['transfer_ratio'])
        target_mses.append(result['target_mse'])

    # Transfer ratios
    ax = axes[0]
    colors = ['green' if r <= 1.5 else 'orange' if r <= 1.7 else 'red' for r in ratios]
    bars = ax.bar(range(len(names)), ratios, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.axhline(y=1.5, color='r', linestyle='--', linewidth=2, label='Target ≤ 1.5')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Transfer Ratio')
    ax.set_title('AURSAD Held-Out Transfer Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Target MSE
    ax = axes[1]
    ax.bar(range(len(names)), target_mses, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Target MSE')
    ax.set_title('AURSAD Test MSE (lower is better)')
    ax.grid(True, alpha=0.3)

    # Training curves
    ax = axes[2]
    for name, result in all_results.items():
        if result is None:
            continue
        ax.plot(result['history']['val_loss'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Exp05: PatchTST vs Joint Forecasting (seed={seed})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = FIGURES_DIR / f"exp05_patchtst_comparison_seed{seed}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def run_multi_seed(config_name, config, seeds=[42, 123, 456]):
    """Run best config with multiple seeds."""
    results = []
    for seed in seeds:
        r = run_experiment(config_name, config, seed=seed, held_out="aursad")
        if r is not None:
            results.append(r)

    if not results:
        return None

    ratios = [r['transfer_ratio'] for r in results]
    logger.info(f"\n{'='*60}")
    logger.info(f"MULTI-SEED RESULTS: {config_name}")
    logger.info(f"  Ratios: {[f'{r:.4f}' for r in ratios]}")
    logger.info(f"  Mean: {np.mean(ratios):.4f} ± {np.std(ratios):.4f}")
    logger.info(f"  All pass: {all(r <= 1.5 for r in ratios)}")
    logger.info(f"{'='*60}")

    return {
        "config_name": config_name,
        "seeds": seeds,
        "ratios": ratios,
        "mean_ratio": float(np.mean(ratios)),
        "std_ratio": float(np.std(ratios)),
        "all_pass": all(r <= 1.5 for r in ratios),
        "results": results,
    }


def run_full_loo(config_name, config, seed=42):
    """Run full leave-one-out with best config."""
    results = {}
    for held_out in ["aursad", "voraus"]:
        r = run_experiment(config_name, config, seed=seed, held_out=held_out)
        if r is not None:
            results[held_out] = r

    logger.info(f"\n{'='*60}")
    logger.info(f"FULL LOO RESULTS (seed={seed}):")
    for ho, r in results.items():
        logger.info(f"  {ho}: ratio={r['transfer_ratio']:.4f} {'PASS' if r['passed'] else 'FAIL'}")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None,
                        help="Run specific config or 'all'")
    parser.add_argument("--multi-seed", action="store_true")
    parser.add_argument("--full-loo", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.multi_seed:
        # Find best config from single run first, or use patchtst_base
        config_name = args.config or "patchtst_base"
        config = CONFIGS[config_name]
        ms_results = run_multi_seed(config_name, config)
        outfile = RESULTS_DIR / f"exp05_multiseed_{config_name}_{timestamp}.json"
        with open(outfile, 'w') as f:
            json.dump(ms_results, f, indent=2, default=str)
        logger.info(f"Saved: {outfile}")

    elif args.full_loo:
        config_name = args.config or "patchtst_base"
        config = CONFIGS[config_name]
        loo_results = run_full_loo(config_name, config, seed=args.seed)
        outfile = RESULTS_DIR / f"exp05_loo_{config_name}_seed{args.seed}_{timestamp}.json"
        with open(outfile, 'w') as f:
            json.dump(loo_results, f, indent=2, default=str)
        logger.info(f"Saved: {outfile}")

    else:
        # Run all configs
        configs_to_run = CONFIGS
        if args.config and args.config != 'all':
            configs_to_run = {args.config: CONFIGS[args.config]}

        all_results = {}
        for name, config in configs_to_run.items():
            result = run_experiment(name, config, seed=args.seed, held_out="aursad")
            all_results[name] = result

        plot_comparison(all_results, args.seed)

        outfile = RESULTS_DIR / f"exp05_seed{args.seed}_{timestamp}.json"
        with open(outfile, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Saved: {outfile}")

        # Summary
        print("\n" + "=" * 80)
        print("EXP05 RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Config':<22} {'Model':<10} {'Ratio':<10} {'Target MSE':<12} {'Params':<12} {'Pass?'}")
        print("-" * 80)
        for name, r in all_results.items():
            if r is None:
                print(f"{name:<22} {'FAILED'}")
                continue
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{name:<22} {r['config']['model']:<10} {r['transfer_ratio']:<10.4f} "
                  f"{r['target_mse']:<12.6f} {r['n_params']:<12,} {status}")
        print("=" * 80)

        best_name = min([k for k, v in all_results.items() if v is not None],
                        key=lambda k: all_results[k]['transfer_ratio'])
        best = all_results[best_name]
        print(f"\nBest: {best_name} (ratio={best['transfer_ratio']:.4f})")

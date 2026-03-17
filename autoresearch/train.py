#!/usr/bin/env python3
"""
IndustrialJEPA Training Script for Autoresearch
This file IS modified by the agent.

Run: python train.py

Target: minimize val_loss (JEPA prediction error)
Time limit: ~5 minutes per experiment
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prepare import get_dataloaders

# ============================================================================
# HYPERPARAMETERS (Agent can modify these)
# ============================================================================

EPOCHS = 5              # Keep low for 5-min runs
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

# Model architecture
LATENT_DIM = 256
HIDDEN_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_PREDICTOR_LAYERS = 2
NUM_HEADS = 8
DROPOUT = 0.1

# JEPA specific
EMA_MOMENTUM = 0.996
RECONSTRUCTION_WEIGHT = 0.1

# ============================================================================
# MODEL DEFINITION (Agent can modify architecture)
# ============================================================================

class StateEncoder(nn.Module):
    """Encodes observation sequence to latent representation."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int,
                 num_layers: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        # x: (B, T, obs_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)
        return self.norm(x)


class StatePredictor(nn.Module):
    """Predicts next latent state from current state + command."""

    def __init__(self, latent_dim: int, cmd_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        layers = []
        input_dim = latent_dim + cmd_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, out_dim),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ])

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, cmd):
        # z: (B, latent_dim), cmd: (B, cmd_dim)
        x = torch.cat([z, cmd.mean(dim=1)], dim=-1)  # Pool command over time
        return self.norm(self.mlp(x))


class Decoder(nn.Module):
    """Decodes latent state back to observation (optional reconstruction)."""

    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z):
        return self.mlp(z)


class JEPAModel(nn.Module):
    """JEPA World Model for time series."""

    def __init__(self, obs_dim: int, cmd_dim: int, latent_dim: int, hidden_dim: int,
                 num_encoder_layers: int, num_predictor_layers: int, num_heads: int,
                 ema_momentum: float = 0.996, recon_weight: float = 0.1):
        super().__init__()

        self.ema_momentum = ema_momentum
        self.recon_weight = recon_weight

        # Online encoder
        self.encoder = StateEncoder(obs_dim, latent_dim, hidden_dim,
                                    num_encoder_layers, num_heads)

        # Target encoder (EMA updated)
        self.target_encoder = StateEncoder(obs_dim, latent_dim, hidden_dim,
                                           num_encoder_layers, num_heads)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self._copy_encoder_to_target()

        # Predictor
        self.predictor = StatePredictor(latent_dim, cmd_dim, hidden_dim,
                                        num_predictor_layers)

        # Decoder (optional)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)

    def _copy_encoder_to_target(self):
        for p_online, p_target in zip(self.encoder.parameters(),
                                      self.target_encoder.parameters()):
            p_target.data.copy_(p_online.data)

    @torch.no_grad()
    def update_ema(self):
        for p_online, p_target in zip(self.encoder.parameters(),
                                      self.target_encoder.parameters()):
            p_target.data = self.ema_momentum * p_target.data + \
                           (1 - self.ema_momentum) * p_online.data

    def forward(self, obs_t, cmd_t, obs_t1):
        """
        Args:
            obs_t: (B, T, obs_dim) - current observation
            cmd_t: (B, T, cmd_dim) - command/setpoint
            obs_t1: (B, T, obs_dim) - next observation

        Returns:
            dict with losses
        """
        # Encode current state
        z_t = self.encoder(obs_t)

        # Predict next state
        z_pred = self.predictor(z_t, cmd_t)

        # Target: encode next state with EMA encoder
        with torch.no_grad():
            z_target = self.target_encoder(obs_t1)

        # JEPA loss: predict in latent space
        jepa_loss = F.mse_loss(z_pred, z_target)

        # Optional reconstruction loss
        obs_recon = self.decoder(z_t)
        recon_loss = F.mse_loss(obs_recon, obs_t.mean(dim=1))

        total_loss = jepa_loss + self.recon_weight * recon_loss

        return {
            'total': total_loss,
            'jepa': jepa_loss,
            'recon': recon_loss,
        }

    def get_anomaly_score(self, obs_t, cmd_t, obs_t1):
        """Compute anomaly score as prediction error."""
        z_t = self.encoder(obs_t)
        z_pred = self.predictor(z_t, cmd_t)

        with torch.no_grad():
            z_target = self.target_encoder(obs_t1)

        return F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_jepa = 0
    n_batches = 0

    for batch in loader:
        obs_t = batch['obs_t'].to(device)
        cmd_t = batch['cmd_t'].to(device)
        obs_t1 = batch['obs_t1'].to(device)

        losses = model(obs_t, cmd_t, obs_t1)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema()

        total_loss += losses['total'].item()
        total_jepa += losses['jepa'].item()
        n_batches += 1

    return total_loss / n_batches, total_jepa / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_jepa = 0
    n_batches = 0

    for batch in loader:
        obs_t = batch['obs_t'].to(device)
        cmd_t = batch['cmd_t'].to(device)
        obs_t1 = batch['obs_t1'].to(device)

        losses = model(obs_t, cmd_t, obs_t1)

        total_loss += losses['total'].item()
        total_jepa += losses['jepa'].item()
        n_batches += 1

    return total_loss / n_batches, total_jepa / n_batches


def main():
    start_time = time.time()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, info = get_dataloaders(BATCH_SIZE)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    model = JEPAModel(
        obs_dim=info['obs_dim'],
        cmd_dim=info['cmd_dim'],
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_predictor_layers=NUM_PREDICTOR_LAYERS,
        num_heads=NUM_HEADS,
        ema_momentum=EMA_MOMENTUM,
        recon_weight=RECONSTRUCTION_WEIGHT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    best_val_loss = float('inf')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_jepa = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_jepa = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), results_dir / 'best_model.pt')

    # Final evaluation
    test_loss, test_jepa = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time

    # Print metrics in standard format
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"val_loss: {best_val_loss:.6f}")
    print(f"test_loss: {test_loss:.6f}")
    print(f"test_jepa: {test_jepa:.6f}")
    print(f"elapsed_seconds: {elapsed:.1f}")
    print(f"parameters: {n_params}")
    print("=" * 50)

    # Save results
    results = {
        'val_loss': best_val_loss,
        'test_loss': test_loss,
        'test_jepa': test_jepa,
        'elapsed_seconds': elapsed,
        'parameters': n_params,
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'latent_dim': LATENT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'num_predictor_layers': NUM_PREDICTOR_LAYERS,
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'latest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return best_val_loss


if __name__ == "__main__":
    main()

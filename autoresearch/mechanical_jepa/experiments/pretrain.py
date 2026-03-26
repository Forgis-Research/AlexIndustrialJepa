#!/usr/bin/env python3
"""
Mechanical-JEPA Pretraining Script

Run this AFTER viability_test.py passes.

Usage:
    python pretrain.py --config small --epochs 50
    python pretrain.py --config base --epochs 100
"""

import argparse
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Model Configurations
# ============================================================================

CONFIGS = {
    'tiny': {
        'd_model': 64,
        'n_heads': 4,
        'n_layers': 2,
        'predictor_layers': 1,
    },
    'small': {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'predictor_layers': 2,
    },
    'base': {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'predictor_layers': 2,
    },
    'large': {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 8,
        'predictor_layers': 3,
    },
}


# ============================================================================
# Full Model Implementation
# ============================================================================

class StateEncoder(nn.Module):
    """Transformer encoder for robot state sequences."""

    def __init__(self, input_dim, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :seq_len]
        x = self.transformer(x)
        return self.norm(x)


class ActionEncoder(nn.Module):
    """MLP encoder for action sequences."""

    def __init__(self, action_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, actions):
        # actions: (batch, seq_len, action_dim)
        return self.net(actions)


class Predictor(nn.Module):
    """Predicts target embeddings from context + actions."""

    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, context, target_queries):
        # context: (batch, context_len, d_model)
        # target_queries: (batch, pred_len, d_model)
        out = self.transformer(target_queries, context)
        return self.norm(out)


class MechanicalJEPA(nn.Module):
    """Full Mechanical-JEPA model."""

    def __init__(
        self,
        state_dim=7,
        action_dim=7,
        d_model=128,
        n_heads=4,
        n_layers=4,
        predictor_layers=2,
        mask_ratio=0.3,
        ema_decay=0.996,
    ):
        super().__init__()

        self.state_encoder = StateEncoder(state_dim, d_model, n_heads, n_layers)
        self.target_encoder = StateEncoder(state_dim, d_model, n_heads, n_layers)
        self.action_encoder = ActionEncoder(action_dim, d_model)
        self.predictor = Predictor(d_model, n_heads, predictor_layers)

        # Initialize target encoder as copy
        self.target_encoder.load_state_dict(self.state_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Learnable query tokens for prediction
        self.query_tokens = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)

        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

    def create_temporal_mask(self, seq_len, context_len):
        """Create mask: context is first context_len, rest is target."""
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[context_len:] = True
        return mask

    def forward(self, states, actions=None, context_ratio=0.7):
        """
        Forward pass for training.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim) or None
            context_ratio: fraction of sequence to use as context
        """
        batch_size, seq_len, state_dim = states.shape
        context_len = int(seq_len * context_ratio)

        # Split into context and target
        context_states = states[:, :context_len]
        target_states = states[:, context_len:]

        # Encode context
        z_context = self.state_encoder(context_states)

        # Encode actions if provided
        if actions is not None:
            action_embed = self.action_encoder(actions[:, :context_len])
            z_context = z_context + action_embed

        # Get target embeddings (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(target_states)

        # Predict target embeddings
        pred_len = seq_len - context_len
        queries = self.query_tokens[:, :pred_len].expand(batch_size, -1, -1)
        z_pred = self.predictor(z_context, queries)

        # Loss: MSE in latent space
        loss = F.mse_loss(z_pred, z_target)

        return loss, {
            'z_context': z_context,
            'z_target': z_target,
            'z_pred': z_pred,
        }

    def encode(self, states):
        """Encode states (for evaluation)."""
        return self.state_encoder(states)

    def ema_update(self):
        """Update target encoder with EMA."""
        with torch.no_grad():
            for p_enc, p_tgt in zip(
                self.state_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                p_tgt.data = self.ema_decay * p_tgt.data + (1 - self.ema_decay) * p_enc.data


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        states = batch['states'].to(device)
        actions = batch.get('actions')
        if actions is not None:
            actions = actions.to(device)

        optimizer.zero_grad()
        loss, _ = model(states, actions)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        model.ema_update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        states = batch['states'].to(device)
        actions = batch.get('actions')
        if actions is not None:
            actions = actions.to(device)

        loss, _ = model(states, actions)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# Data Loading (Placeholder)
# ============================================================================

def create_synthetic_dataloader(n_samples, seq_len, state_dim, action_dim, batch_size):
    """Create synthetic dataloader for testing."""
    from torch.utils.data import DataLoader, TensorDataset

    states = torch.randn(n_samples, seq_len, state_dim)
    actions = torch.randn(n_samples, seq_len, action_dim)

    # Make states more realistic (smooth trajectories)
    for i in range(1, seq_len):
        states[:, i] = 0.9 * states[:, i-1] + 0.1 * states[:, i]

    dataset = TensorDataset(states, actions)

    def collate_fn(batch):
        states, actions = zip(*batch)
        return {
            'states': torch.stack(states),
            'actions': torch.stack(actions),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Mechanical-JEPA Pretraining')
    parser.add_argument('--config', type=str, default='small', choices=CONFIGS.keys())
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Model
    cfg = CONFIGS[args.config]
    model = MechanicalJEPA(
        state_dim=7,
        action_dim=7,
        **cfg
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.config} ({n_params:,} params)")

    # Data (placeholder - replace with real OXE loading)
    print("Loading data...")
    train_loader = create_synthetic_dataloader(
        n_samples=5000, seq_len=128, state_dim=7, action_dim=7, batch_size=args.batch_size
    )
    val_loader = create_synthetic_dataloader(
        n_samples=500, seq_len=128, state_dim=7, action_dim=7, batch_size=args.batch_size
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print(f"\nStarting training: {args.epochs} epochs")
    print("=" * 60)

    best_val_loss = float('inf')
    history = []

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr = scheduler.get_last_lr()[0]

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'time': epoch_time,
        })

        # Logging
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"lr={lr:.2e}, time={epoch_time:.1f}s")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': cfg,
            }, checkpoint_dir / 'best.pt')

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': cfg,
            }, checkpoint_dir / f'epoch_{epoch+1}.pt')

    print("=" * 60)
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

    # Save history
    with open(checkpoint_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

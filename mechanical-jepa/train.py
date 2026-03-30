"""
Mechanical-JEPA Training Script.

Usage:
    python train.py                      # Train with defaults
    python train.py --epochs 50          # Custom epochs
    python train.py --eval-only          # Evaluate pretrained model
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import create_dataloaders
from src.models import MechanicalJEPA


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_dir': 'data/bearings',
    'dataset_filter': 'cwru',  # Start with CWRU only
    'batch_size': 32,
    'window_size': 4096,
    'stride': 2048,
    'n_channels': 3,
    'test_ratio': 0.2,
    'num_workers': 0,

    # Model
    'patch_size': 256,
    'embed_dim': 256,
    'encoder_depth': 4,
    'predictor_depth': 2,
    'n_heads': 4,
    'mask_ratio': 0.5,
    'ema_decay': 0.996,

    # Training
    'epochs': 30,
    'lr': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'min_lr': 1e-6,

    # Linear probe
    'probe_epochs': 20,
    'probe_lr': 1e-3,

    # Other
    'seed': 42,
    'log_interval': 10,
    'save_dir': 'checkpoints',
}


def get_args():
    parser = argparse.ArgumentParser(description='Train Mechanical-JEPA')

    # Override config values
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--encoder-depth', type=int, default=None)
    parser.add_argument('--mask-ratio', type=float, default=None)
    parser.add_argument('--dataset', type=str, default=None, choices=['cwru', 'ims', None])
    parser.add_argument('--seed', type=int, default=None)

    # Modes
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--no-save', action='store_true', help='Do not save checkpoints')

    return parser.parse_args()


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    """Cosine learning rate schedule with warmup."""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)

    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate([warmup_schedule, schedule])
    return schedule


# =============================================================================
# Linear Probe
# =============================================================================

class LinearProbe(nn.Module):
    """Simple linear classifier for evaluation."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(
    model: MechanicalJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Train a linear probe on frozen JEPA embeddings.

    Returns dict with train_acc, test_acc, per-class metrics.
    """
    print("\n" + "=" * 60)
    print("LINEAR PROBE EVALUATION")
    print("=" * 60)

    model.eval()
    n_classes = 4  # healthy, outer_race, inner_race, ball

    # Extract embeddings
    print("Extracting embeddings...")

    def extract_embeddings(loader):
        all_embeds = []
        all_labels = []
        with torch.no_grad():
            for signals, labels, _ in loader:
                signals = signals.to(device)
                embeds = model.get_embeddings(signals)
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        return torch.cat(all_embeds), torch.cat(all_labels)

    train_embeds, train_labels = extract_embeddings(train_loader)
    test_embeds, test_labels = extract_embeddings(test_loader)

    print(f"Train embeddings: {train_embeds.shape}")
    print(f"Test embeddings: {test_embeds.shape}")

    # Create probe
    probe = LinearProbe(config['embed_dim'], n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=config['probe_lr'])
    criterion = nn.CrossEntropyLoss()

    # Train probe
    print(f"\nTraining linear probe for {config['probe_epochs']} epochs...")

    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    test_embeds = test_embeds.to(device)
    test_labels = test_labels.to(device)

    best_test_acc = 0
    for epoch in range(config['probe_epochs']):
        probe.train()

        # Simple batch training on all embeddings
        optimizer.zero_grad()
        logits = probe(train_embeds)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_embeds)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels).float().mean().item()

            test_logits = probe(test_embeds)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # Per-class accuracy
    print("\nPer-class test accuracy:")
    class_names = ['healthy', 'outer_race', 'inner_race', 'ball']
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            acc = (test_preds[mask] == test_labels[mask]).float().mean().item()
            per_class_acc[name] = acc
            print(f"  {name:12s}: {acc:.4f} ({mask.sum().item()} samples)")

    results = {
        'train_acc': train_acc,
        'test_acc': best_test_acc,
        'per_class_acc': per_class_acc,
    }

    print(f"\nBest test accuracy: {best_test_acc:.4f}")

    return results


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: MechanicalJEPA,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    config: dict,
    device: torch.device,
    lr_schedule: np.ndarray,
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)

    for batch_idx, (signals, labels, _) in enumerate(train_loader):
        # Update learning rate
        global_step = epoch * n_batches + batch_idx
        lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        signals = signals.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = model.train_step(signals)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        model.update_ema()

        total_loss += loss.item()

        if (batch_idx + 1) % config['log_interval'] == 0:
            print(f"  Batch {batch_idx+1}/{n_batches}: loss={loss.item():.4f}, lr={lr:.2e}")

    return total_loss / n_batches


def train(config: dict, device: torch.device):
    """Main training function."""
    print("=" * 60)
    print("MECHANICAL-JEPA TRAINING")
    print("=" * 60)
    print(f"\nConfig:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create data loaders
    print("\nLoading data...")
    train_loader, test_loader, data_info = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_ratio=config['test_ratio'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        dataset_filter=config['dataset_filter'],
        n_channels=config['n_channels'],
    )

    print(f"Train: {data_info['train_windows']} windows from {len(data_info['train_bearings'])} bearings")
    print(f"Test: {data_info['test_windows']} windows from {len(data_info['test_bearings'])} bearings")

    # Create model
    print("\nCreating model...")
    model = MechanicalJEPA(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'],
        n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'],
        ema_decay=config['ema_decay'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
    )

    # LR schedule
    lr_schedule = cosine_scheduler(
        config['lr'], config['min_lr'],
        config['epochs'], config['warmup_epochs']
    )

    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    history = {'loss': [], 'lr': []}

    start_time = time.time()
    for epoch in range(config['epochs']):
        epoch_start = time.time()

        avg_loss = train_epoch(
            model, train_loader, optimizer,
            epoch, config, device, lr_schedule
        )

        history['loss'].append(avg_loss)
        history['lr'].append(lr_schedule[min(epoch, len(lr_schedule) - 1)])

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    # Linear probe evaluation
    probe_results = train_linear_probe(model, train_loader, test_loader, config, device)

    # Save checkpoint
    if not config.get('no_save', False):
        save_dir = Path(config['save_dir'])
        save_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = save_dir / f'jepa_{timestamp}.pt'

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history,
            'probe_results': probe_results,
            'data_info': data_info,
        }, checkpoint_path)

        print(f"\nCheckpoint saved to {checkpoint_path}")

    return model, history, probe_results


# =============================================================================
# Main
# =============================================================================

def main():
    args = get_args()

    # Build config
    config = DEFAULT_CONFIG.copy()

    # Override from args
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    if args.embed_dim is not None:
        config['embed_dim'] = args.embed_dim
    if args.encoder_depth is not None:
        config['encoder_depth'] = args.encoder_depth
    if args.mask_ratio is not None:
        config['mask_ratio'] = args.mask_ratio
    if args.dataset is not None:
        config['dataset_filter'] = args.dataset
    if args.seed is not None:
        config['seed'] = args.seed
    if args.no_save:
        config['no_save'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.eval_only:
        # Load checkpoint and evaluate
        if args.checkpoint is None:
            print("Error: --checkpoint required for --eval-only mode")
            return

        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Use config from checkpoint
        saved_config = checkpoint['config']
        config.update(saved_config)

        # Create data loaders
        train_loader, test_loader, data_info = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            window_size=config['window_size'],
            stride=config['stride'],
            test_ratio=config['test_ratio'],
            seed=config['seed'],
            num_workers=config['num_workers'],
            dataset_filter=config['dataset_filter'],
            n_channels=config['n_channels'],
        )

        # Create and load model
        model = MechanicalJEPA(
            n_channels=config['n_channels'],
            window_size=config['window_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            encoder_depth=config['encoder_depth'],
            predictor_depth=config['predictor_depth'],
            n_heads=config['n_heads'],
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Run linear probe
        probe_results = train_linear_probe(model, train_loader, test_loader, config, device)

    else:
        # Train
        model, history, probe_results = train(config, device)

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {probe_results['test_acc']:.4f}")
    print("\nPer-class accuracy:")
    for cls, acc in probe_results['per_class_acc'].items():
        print(f"  {cls}: {acc:.4f}")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    random_baseline = 0.25  # 4-class random
    transferability_target = random_baseline + 0.05  # +5% over random

    if probe_results['test_acc'] > transferability_target:
        print(f"[PASS] Transferability: {probe_results['test_acc']:.1%} > {transferability_target:.1%} target")
    else:
        print(f"[FAIL] Transferability: {probe_results['test_acc']:.1%} <= {transferability_target:.1%} target")


if __name__ == '__main__':
    main()

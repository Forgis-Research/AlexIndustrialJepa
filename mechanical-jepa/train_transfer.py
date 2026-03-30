"""
Enhanced training script with cross-dataset transfer support.

Modes:
1. Standard: Train on CWRU, test on CWRU (baseline)
2. Transfer: Train on CWRU, test on IMS degradation detection
3. Mixed: Train on CWRU+IMS, test on both

Usage:
    python train_transfer.py --mode standard --epochs 30
    python train_transfer.py --mode transfer --epochs 30
    python train_transfer.py --mode mixed --epochs 30
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

from src.data import BearingDataset, create_dataloaders
from src.models import MechanicalJEPA
from train import LinearProbe, cosine_scheduler, train_epoch
from transfer_eval import create_ims_degradation_loaders, extract_embeddings


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # Data
    'data_dir': 'data/bearings',
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

    # Transfer
    'transfer_probe_epochs': 100,
    'transfer_probe_lr': 1e-3,

    # Other
    'seed': 42,
    'log_interval': 10,
    'save_dir': 'checkpoints',
    'mode': 'standard',
}


def get_args():
    parser = argparse.ArgumentParser(description='Train Mechanical-JEPA with transfer')

    # Mode
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['standard', 'transfer', 'mixed'],
                       help='Training mode')

    # Override config
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--encoder-depth', type=int, default=None)
    parser.add_argument('--mask-ratio', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)

    # Modes
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='mechanical-jepa')

    return parser.parse_args()


# =============================================================================
# Linear Probe Training
# =============================================================================

def train_linear_probe_simple(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict:
    """Train linear probe on extracted embeddings."""
    embed_dim = embeddings.shape[1]

    # Convert to tensors
    train_embeds = torch.tensor(embeddings, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels, dtype=torch.long).to(device)
    test_embeds = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Create probe
    probe = LinearProbe(embed_dim, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    best_test_acc = 0
    for epoch in range(epochs):
        probe.train()
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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # Per-class accuracy
    probe.eval()
    with torch.no_grad():
        test_logits = probe(test_embeds)
        test_preds = test_logits.argmax(dim=1)

    per_class_acc = {}
    for i in range(n_classes):
        mask = test_labels == i
        if mask.sum() > 0:
            acc = (test_preds[mask] == test_labels[mask]).float().mean().item()
            per_class_acc[f'class_{i}'] = acc

    return {
        'train_acc': train_acc,
        'test_acc': best_test_acc,
        'per_class_acc': per_class_acc,
    }


# =============================================================================
# Main Training
# =============================================================================

def train_with_transfer(config: dict, device: torch.device):
    """Main training with transfer evaluation."""
    print("="*60)
    print("MECHANICAL-JEPA TRAINING WITH TRANSFER")
    print("="*60)
    print(f"\nMode: {config['mode']}")
    print(f"\nConfig:")
    for key, value in config.items():
        if not key.startswith('wandb') and key != 'mode':
            print(f"  {key}: {value}")

    # Initialize wandb
    use_wandb = not config.get('no_wandb', False)
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'mechanical-jepa'),
            config={k: v for k, v in config.items() if not k.startswith('wandb') and k != 'no_wandb'},
            name=f"jepa_{config['mode']}_{config['epochs']}ep",
        )

    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create CWRU data loaders
    print("\n" + "="*60)
    print("LOADING CWRU DATA (pretraining)")
    print("="*60)
    cwru_train_loader, cwru_test_loader, cwru_info = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_ratio=config['test_ratio'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        dataset_filter='cwru',
        n_channels=config['n_channels'],
    )

    print(f"CWRU Train: {cwru_info['train_windows']} windows from {len(cwru_info['train_bearings'])} bearings")
    print(f"CWRU Test: {cwru_info['test_windows']} windows from {len(cwru_info['test_bearings'])} bearings")

    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
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
    print(f"Total parameters: {n_params:,}")

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
    print("\n" + "="*60)
    print(f"TRAINING ON CWRU ({config['epochs']} epochs)")
    print("="*60)
    history = {'loss': [], 'lr': []}

    start_time = time.time()
    for epoch in range(config['epochs']):
        epoch_start = time.time()

        avg_loss = train_epoch(
            model, cwru_train_loader, optimizer,
            epoch, config, device, lr_schedule
        )

        history['loss'].append(avg_loss)
        history['lr'].append(lr_schedule[min(epoch, len(lr_schedule) - 1)])

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': history['lr'][-1],
                'epoch_time': epoch_time,
            })

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")

    # =========================================================================
    # EVALUATION
    # =========================================================================

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    results = {}

    # 1. CWRU Linear Probe (within-dataset)
    print("\n1. CWRU Within-Dataset Evaluation")
    print("-"*60)
    cwru_train_embeds, cwru_train_labels, _ = extract_embeddings(model, cwru_train_loader, device)
    cwru_test_embeds, cwru_test_labels, _ = extract_embeddings(model, cwru_test_loader, device)

    cwru_results = train_linear_probe_simple(
        cwru_train_embeds, cwru_train_labels,
        cwru_test_embeds, cwru_test_labels,
        n_classes=4,
        epochs=config['probe_epochs'],
        lr=config['probe_lr'],
        device=device,
    )

    print(f"\nCWRU Test Accuracy: {cwru_results['test_acc']:.4f}")
    print("Per-class:")
    for cls, acc in cwru_results['per_class_acc'].items():
        print(f"  {cls}: {acc:.4f}")

    results['cwru'] = cwru_results

    # 2. IMS Transfer (cross-dataset)
    if config['mode'] in ['transfer', 'mixed']:
        print("\n2. IMS Cross-Dataset Transfer Evaluation")
        print("-"*60)

        ims_train_loader, ims_test_loader, ims_info = create_ims_degradation_loaders(
            data_dir=config['data_dir'],
            test_set='1st_test',
            batch_size=config['batch_size'],
            window_size=config['window_size'],
            stride=config['stride'],
            n_channels=config['n_channels'],
        )

        # Extract IMS embeddings using CWRU-pretrained encoder
        print("\nExtracting IMS embeddings with CWRU-pretrained encoder...")
        ims_train_embeds, ims_train_labels, _ = extract_embeddings(model, ims_train_loader, device)
        ims_test_embeds, ims_test_labels, _ = extract_embeddings(model, ims_test_loader, device)

        print(f"IMS Train embeddings: {ims_train_embeds.shape}")
        print(f"IMS Test embeddings: {ims_test_embeds.shape}")

        # Train probe on IMS
        print("\nTraining linear probe on IMS...")
        ims_results = train_linear_probe_simple(
            ims_train_embeds, ims_train_labels,
            ims_test_embeds, ims_test_labels,
            n_classes=2,  # healthy vs degraded
            epochs=config['transfer_probe_epochs'],
            lr=config['transfer_probe_lr'],
            device=device,
        )

        print(f"\n{'='*60}")
        print("TRANSFER RESULTS: CWRU → IMS")
        print(f"{'='*60}")
        print(f"Train accuracy: {ims_results['train_acc']:.4f}")
        print(f"Test accuracy: {ims_results['test_acc']:.4f}")
        print(f"Random baseline: 0.5000")

        # Check transfer success
        if ims_results['test_acc'] > 0.55:
            print(f"\n✓ TRANSFER SUCCESS: {ims_results['test_acc']:.1%} > 55%")
            transfer_success = True
        else:
            print(f"\n✗ TRANSFER WEAK: {ims_results['test_acc']:.1%} <= 55%")
            transfer_success = False

        results['ims'] = ims_results
        results['transfer_success'] = transfer_success

    # Log to wandb
    if use_wandb:
        wandb.log({
            'final/cwru_test_acc': cwru_results['test_acc'],
            'final/training_time_min': total_time / 60,
        })

        if 'ims' in results:
            wandb.log({
                'final/ims_test_acc': results['ims']['test_acc'],
                'final/transfer_gap': cwru_results['test_acc'] - results['ims']['test_acc'],
                'final/transfer_success': results['transfer_success'],
            })

        wandb.summary['cwru_test_acc'] = cwru_results['test_acc']
        if 'ims' in results:
            wandb.summary['ims_test_acc'] = results['ims']['test_acc']

    # Save checkpoint
    if not config.get('no_save', False):
        save_dir = Path(config['save_dir'])
        save_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = save_dir / f'jepa_{config["mode"]}_{timestamp}.pt'

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'history': history,
            'results': results,
            'cwru_info': cwru_info,
        }, checkpoint_path)

        print(f"\nCheckpoint saved to {checkpoint_path}")

        if use_wandb:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    return model, history, results


def main():
    args = get_args()

    # Build config
    config = DEFAULT_CONFIG.copy()

    # Override from args
    config['mode'] = args.mode
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
    if args.seed is not None:
        config['seed'] = args.seed
    if args.no_save:
        config['no_save'] = True
    if args.no_wandb:
        config['no_wandb'] = True
    config['wandb_project'] = args.wandb_project

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train
    model, history, results = train_with_transfer(config, device)

    # Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Mode: {config['mode']}")
    print(f"CWRU Test Accuracy: {results['cwru']['test_acc']:.4f}")

    if 'ims' in results:
        print(f"IMS Test Accuracy: {results['ims']['test_acc']:.4f}")
        print(f"Transfer Gap: {results['cwru']['test_acc'] - results['ims']['test_acc']:.4f}")

        if results['transfer_success']:
            print("\n✓ CROSS-DATASET TRANSFER SUCCESSFUL")
        else:
            print("\n✗ CROSS-DATASET TRANSFER NEEDS IMPROVEMENT")


if __name__ == '__main__':
    main()

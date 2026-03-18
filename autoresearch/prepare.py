#!/usr/bin/env python3
"""
Data preparation for autoresearch experiments.
This file is NOT modified by the agent.

Usage:
    python prepare.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from industrialjepa.data import FactoryNetConfig, WorldModelDataConfig, create_world_model_dataloaders


def prepare_data(
    data_source: str = "aursad",
    window_size: int = 256,
    batch_size: int = 64,
    max_episodes: int = 100,  # Limit for fast iteration
):
    """Prepare dataloaders for autoresearch experiments."""

    print(f"Preparing {data_source} data...")
    print(f"  Window size: {window_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max episodes: {max_episodes}")

    factorynet_config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        config_name="normalized",
        data_source=data_source,
        window_size=window_size,
        effort_signals=["torque", "current", "velocity"],
        train_healthy_only=True,
        max_episodes=max_episodes,
    )

    data_config = WorldModelDataConfig(
        factorynet_config=factorynet_config,
        obs_mode="effort",
        cmd_mode="setpoint",
    )

    train_loader, val_loader, test_loader, info = create_world_model_dataloaders(
        data_config,
        batch_size=batch_size,
        num_workers=4,  # Linux (SageMaker)
    )

    print(f"Data loaded:")
    print(f"  obs_dim: {info['obs_dim']}")
    print(f"  cmd_dim: {info['cmd_dim']}")
    print(f"  Train windows: {info['train_size']}")
    print(f"  Val windows: {info['val_size']}")
    print(f"  Test windows: {info['test_size']}")

    # Save info for train.py
    torch.save({
        'info': info,
        'factorynet_config': factorynet_config,
        'data_config': data_config,
    }, 'data_info.pt')

    print("✓ Data prepared. Saved info to data_info.pt")

    return train_loader, val_loader, test_loader, info


def get_dataloaders(batch_size: int = 64):
    """Load prepared dataloaders (call after prepare_data)."""

    data_info = torch.load('data_info.pt', weights_only=False)
    info = data_info['info']
    factorynet_config = data_info['factorynet_config']
    data_config = data_info['data_config']

    # Recreate dataloaders
    train_loader, val_loader, test_loader, _ = create_world_model_dataloaders(
        data_config,
        batch_size=batch_size,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    prepare_data()

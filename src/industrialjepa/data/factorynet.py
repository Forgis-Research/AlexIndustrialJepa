# SPDX-FileCopyrightText: 2025-2026 Forgis AG
# SPDX-License-Identifier: MIT

"""FactoryNet dataset loader for IndustrialJEPA.

FactoryNet provides causally-structured industrial time series data with:
- Setpoint: What the controller commanded (position, velocity)
- Effort: What the machine expended (motor current)
- Feedback: What actually happened (measured position)

This loader supports both single-machine and multi-machine configurations
for JEPA training and cross-machine transfer experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

logger = logging.getLogger(__name__)


# Column name patterns for FactoryNet schema
SETPOINT_COLS = {
    "position": [f"setpoint_pos_{i}" for i in range(6)],
    "velocity": [f"setpoint_vel_{i}" for i in range(6)],
    "acceleration": [f"setpoint_acc_{i}" for i in range(6)],
}

EFFORT_COLS = {
    "current": [f"effort_current_{i}" for i in range(6)],
    "voltage": [f"effort_voltage_{i}" for i in range(6)],
}

FEEDBACK_COLS = {
    "position": [f"feedback_pos_{i}" for i in range(6)],
    "velocity": [f"feedback_vel_{i}" for i in range(6)],
}

METADATA_COLS = ["dataset_source", "machine_type", "episode_id", "ctx_anomaly_label"]


@dataclass
class FactoryNetConfig:
    """Configuration for FactoryNet dataset loading."""

    # Dataset source
    dataset_name: str = "Forgis/factorynet-hackathon"
    subset: Optional[str] = None  # None = all, or "AURSAD", "voraus-AD", etc.

    # Sequence parameters
    window_size: int = 256
    stride: int = 128  # Overlap between windows

    # Normalization
    normalize: bool = True
    norm_mode: Literal["episode", "global", "none"] = "episode"

    # Column selection
    setpoint_signals: list[str] = field(default_factory=lambda: ["position", "velocity"])
    effort_signals: list[str] = field(default_factory=lambda: ["current"])

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # For fault detection: only train on healthy data
    train_healthy_only: bool = True


class FactoryNetDataset(Dataset):
    """PyTorch Dataset for FactoryNet data.

    Loads industrial time series with Setpoint/Effort/Feedback structure
    for JEPA training. Returns (setpoint_window, effort_window) pairs.

    Example:
        >>> config = FactoryNetConfig(subset="AURSAD", window_size=256)
        >>> dataset = FactoryNetDataset(config, split="train")
        >>> setpoint, effort, metadata = dataset[0]
        >>> print(setpoint.shape)  # (window_size, num_setpoint_features)
    """

    def __init__(
        self,
        config: FactoryNetConfig,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.config = config
        self.split = split

        # Load dataset from HuggingFace
        logger.info(f"Loading FactoryNet from {config.dataset_name}")
        self._load_data()

        # Build column lists
        self._setup_columns()

        # Create episode index and windows
        self._build_episode_index()
        self._create_windows()

        # Compute normalization statistics
        if config.normalize and config.norm_mode == "global":
            self._compute_global_stats()

        logger.info(
            f"FactoryNetDataset initialized: {len(self)} windows, "
            f"{len(self.episode_ids)} episodes, split={split}"
        )

    def _load_data(self):
        """Load data from HuggingFace datasets."""
        try:
            # Try loading with subset filter
            self.hf_dataset = load_dataset(
                self.config.dataset_name,
                split="train",  # FactoryNet typically has single split
            )
        except Exception as e:
            logger.warning(f"Failed to load {self.config.dataset_name}: {e}")
            logger.info("Falling back to karimm6/FactoryNet_Dataset")
            self.hf_dataset = load_dataset(
                "karimm6/FactoryNet_Dataset",
                "normalized",
                split="train",
            )

        # Convert to pandas for easier manipulation
        self.df = self.hf_dataset.to_pandas()

        # Filter by subset if specified
        if self.config.subset:
            if "dataset_source" in self.df.columns:
                self.df = self.df[self.df["dataset_source"] == self.config.subset]
            elif "machine_type" in self.df.columns:
                # Try filtering by machine_type if dataset_source not available
                self.df = self.df[
                    self.df["machine_type"].str.contains(self.config.subset, case=False)
                ]
            logger.info(f"Filtered to subset {self.config.subset}: {len(self.df)} rows")

    def _setup_columns(self):
        """Identify available columns for setpoint, effort, feedback."""
        available_cols = set(self.df.columns)

        # Build setpoint column list
        self.setpoint_cols = []
        for signal_type in self.config.setpoint_signals:
            if signal_type in SETPOINT_COLS:
                for col in SETPOINT_COLS[signal_type]:
                    if col in available_cols:
                        self.setpoint_cols.append(col)

        # Build effort column list
        self.effort_cols = []
        for signal_type in self.config.effort_signals:
            if signal_type in EFFORT_COLS:
                for col in EFFORT_COLS[signal_type]:
                    if col in available_cols:
                        self.effort_cols.append(col)

        # Validate we have the minimum required columns
        if not self.setpoint_cols:
            raise ValueError(
                f"No setpoint columns found. Available: {available_cols}"
            )
        if not self.effort_cols:
            raise ValueError(
                f"No effort columns found. Available: {available_cols}"
            )

        logger.info(f"Setpoint columns ({len(self.setpoint_cols)}): {self.setpoint_cols}")
        logger.info(f"Effort columns ({len(self.effort_cols)}): {self.effort_cols}")

    def _build_episode_index(self):
        """Build index of episodes and split into train/val/test."""
        # Get unique episode IDs
        if "episode_id" in self.df.columns:
            self.episode_ids = self.df["episode_id"].unique()
        else:
            # If no episode_id, treat entire dataset as one episode
            self.df["episode_id"] = "episode_0"
            self.episode_ids = ["episode_0"]

        # Get anomaly labels per episode
        self.episode_labels = {}
        for ep_id in self.episode_ids:
            ep_data = self.df[self.df["episode_id"] == ep_id]
            if "ctx_anomaly_label" in ep_data.columns:
                # Take most common label in episode
                labels = ep_data["ctx_anomaly_label"].dropna()
                if len(labels) > 0:
                    self.episode_labels[ep_id] = labels.mode().iloc[0] if len(labels.mode()) > 0 else None
                else:
                    self.episode_labels[ep_id] = None
            else:
                self.episode_labels[ep_id] = None

        # Identify healthy vs fault episodes
        healthy_episodes = [
            ep for ep, label in self.episode_labels.items()
            if label is None or str(label).lower() in ["none", "null", "healthy", "normal", ""]
        ]
        fault_episodes = [
            ep for ep in self.episode_ids if ep not in healthy_episodes
        ]

        logger.info(f"Episodes: {len(healthy_episodes)} healthy, {len(fault_episodes)} fault")

        # Split episodes (not rows) into train/val/test
        np.random.seed(42)  # Reproducibility

        if self.config.train_healthy_only:
            # Train only on healthy, test on both
            n_healthy = len(healthy_episodes)
            n_train = int(n_healthy * self.config.train_ratio)
            n_val = int(n_healthy * self.config.val_ratio)

            shuffled_healthy = np.random.permutation(healthy_episodes)
            train_eps = list(shuffled_healthy[:n_train])
            val_eps = list(shuffled_healthy[n_train:n_train + n_val])
            test_eps = list(shuffled_healthy[n_train + n_val:]) + fault_episodes
        else:
            # Standard split including faults in all splits
            all_episodes = list(self.episode_ids)
            np.random.shuffle(all_episodes)
            n_total = len(all_episodes)
            n_train = int(n_total * self.config.train_ratio)
            n_val = int(n_total * self.config.val_ratio)

            train_eps = all_episodes[:n_train]
            val_eps = all_episodes[n_train:n_train + n_val]
            test_eps = all_episodes[n_train + n_val:]

        # Select episodes for this split
        if self.split == "train":
            self.split_episodes = train_eps
        elif self.split == "val":
            self.split_episodes = val_eps
        else:
            self.split_episodes = test_eps

        logger.info(f"Split '{self.split}': {len(self.split_episodes)} episodes")

    def _create_windows(self):
        """Create sliding windows from episodes."""
        self.windows = []  # List of (episode_id, start_idx, end_idx)

        for ep_id in self.split_episodes:
            ep_data = self.df[self.df["episode_id"] == ep_id]
            ep_len = len(ep_data)

            # Skip episodes shorter than window size
            if ep_len < self.config.window_size:
                continue

            # Create windows with stride
            start = 0
            while start + self.config.window_size <= ep_len:
                self.windows.append({
                    "episode_id": ep_id,
                    "start_idx": ep_data.index[start],
                    "end_idx": ep_data.index[start + self.config.window_size - 1],
                    "label": self.episode_labels.get(ep_id),
                })
                start += self.config.stride

        logger.info(f"Created {len(self.windows)} windows")

    def _compute_global_stats(self):
        """Compute global mean/std for normalization."""
        setpoint_data = self.df[self.setpoint_cols].values
        effort_data = self.df[self.effort_cols].values

        self.setpoint_mean = np.nanmean(setpoint_data, axis=0)
        self.setpoint_std = np.nanstd(setpoint_data, axis=0) + 1e-8
        self.effort_mean = np.nanmean(effort_data, axis=0)
        self.effort_std = np.nanstd(effort_data, axis=0) + 1e-8

    def _normalize_window(
        self,
        setpoint: np.ndarray,
        effort: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a window of data."""
        if not self.config.normalize:
            return setpoint, effort

        if self.config.norm_mode == "episode":
            # Per-window normalization (z-score)
            setpoint = (setpoint - np.nanmean(setpoint, axis=0)) / (np.nanstd(setpoint, axis=0) + 1e-8)
            effort = (effort - np.nanmean(effort, axis=0)) / (np.nanstd(effort, axis=0) + 1e-8)
        elif self.config.norm_mode == "global":
            setpoint = (setpoint - self.setpoint_mean) / self.setpoint_std
            effort = (effort - self.effort_mean) / self.effort_std

        return setpoint, effort

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Get a (setpoint, effort) window pair.

        Returns:
            setpoint: Tensor of shape (window_size, num_setpoint_features)
            effort: Tensor of shape (window_size, num_effort_features)
            metadata: Dict with episode_id, label, etc.
        """
        window = self.windows[idx]

        # Get window data
        start_idx = window["start_idx"]
        end_idx = window["end_idx"]

        # Slice dataframe by index range
        window_data = self.df.loc[start_idx:end_idx]

        # Extract setpoint and effort
        setpoint = window_data[self.setpoint_cols].values.astype(np.float32)
        effort = window_data[self.effort_cols].values.astype(np.float32)

        # Handle NaN values
        setpoint = np.nan_to_num(setpoint, nan=0.0)
        effort = np.nan_to_num(effort, nan=0.0)

        # Normalize
        setpoint, effort = self._normalize_window(setpoint, effort)

        # Convert to tensors
        setpoint_tensor = torch.from_numpy(setpoint)
        effort_tensor = torch.from_numpy(effort)

        metadata = {
            "episode_id": window["episode_id"],
            "label": window["label"],
            "is_anomaly": window["label"] is not None and str(window["label"]).lower() not in ["none", "null", "healthy", "normal", ""],
        }

        return setpoint_tensor, effort_tensor, metadata


def create_dataloaders(
    config: FactoryNetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        config: FactoryNet configuration
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = FactoryNetDataset(config, split="train")
    val_dataset = FactoryNetDataset(config, split="val")
    test_dataset = FactoryNetDataset(config, split="test")

    def collate_fn(batch):
        setpoints, efforts, metadatas = zip(*batch)
        return (
            torch.stack(setpoints),
            torch.stack(efforts),
            metadatas,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Convenience function for quick testing
def load_aursad(
    window_size: int = 256,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load AURSAD dataset with default settings.

    Example:
        >>> train, val, test = load_aursad()
        >>> for setpoint, effort, meta in train:
        ...     print(setpoint.shape, effort.shape)
        ...     break
    """
    config = FactoryNetConfig(
        subset="AURSAD",
        window_size=window_size,
    )
    return create_dataloaders(config, batch_size=batch_size)

#!/usr/bin/env python
"""Test loading Voraus dataset."""

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

config = FactoryNetConfig(
    data_source='voraus',
    max_episodes=100,
    window_size=256,
    stride=128,
)
print('Loading...')
ds = FactoryNetDataset(config, split='train')
print(f'Success: {len(ds)} windows')

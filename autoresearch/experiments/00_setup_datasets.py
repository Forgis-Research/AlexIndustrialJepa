#!/usr/bin/env python
"""
Experiment 00: Setup and Validate All Datasets

This script validates that all required datasets can be loaded.
Run this first before starting overnight research.

Datasets:
- Track 1 (Bearings): CWRU, PHM2012, XJTU-SY, Paderborn
- Track 2 (Robots): AURSAD, Voraus, UR3 CobotOps, NIST UR5, Robot Failures
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_phmd():
    """Check if PHMD library is installed and list available datasets."""
    logger.info("=" * 60)
    logger.info("Checking PHMD library...")

    try:
        from phmd import datasets
        logger.info("✓ PHMD installed")

        # List available datasets
        available = datasets.list_datasets()
        logger.info(f"  Available datasets: {len(available)}")

        # Check for our target datasets
        targets = ['CWRU', 'PHM2012', 'XJTU-SY', 'XJTU']
        for t in targets:
            found = [d for d in available if t.lower() in d.lower()]
            if found:
                logger.info(f"  ✓ Found {t}: {found}")
            else:
                logger.warning(f"  ? {t} not found by name, may need different identifier")

        return True
    except ImportError:
        logger.error("✗ PHMD not installed. Run: pip install phmd")
        return False


def check_ucimlrepo():
    """Check if ucimlrepo is installed and test UR3 CobotOps."""
    logger.info("=" * 60)
    logger.info("Checking UCI ML Repository access...")

    try:
        from ucimlrepo import fetch_ucirepo
        logger.info("✓ ucimlrepo installed")

        # Test UR3 CobotOps
        logger.info("  Fetching UR3 CobotOps (id=963)...")
        ur3 = fetch_ucirepo(id=963)
        logger.info(f"  ✓ UR3 CobotOps: {ur3.data.features.shape}")

        # Test Robot Execution Failures
        logger.info("  Fetching Robot Execution Failures (id=138)...")
        failures = fetch_ucirepo(id=138)
        logger.info(f"  ✓ Robot Failures: {failures.data.features.shape}")

        return True
    except ImportError:
        logger.error("✗ ucimlrepo not installed. Run: pip install ucimlrepo")
        return False
    except Exception as e:
        logger.error(f"✗ Error fetching UCI data: {e}")
        return False


def check_factorynet():
    """Check if AURSAD and Voraus datasets load."""
    logger.info("=" * 60)
    logger.info("Checking FactoryNet datasets (AURSAD, Voraus)...")

    try:
        from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

        # Test AURSAD
        logger.info("  Loading AURSAD (10 episodes)...")
        config = FactoryNetConfig(data_source='aursad', max_episodes=10)
        ds = FactoryNetDataset(config, split='train')
        logger.info(f"  ✓ AURSAD: {len(ds)} windows")

        # Test Voraus
        logger.info("  Loading Voraus (10 episodes)...")
        config = FactoryNetConfig(data_source='voraus', max_episodes=10)
        ds = FactoryNetDataset(config, split='train')
        logger.info(f"  ✓ Voraus: {len(ds)} windows")

        return True
    except Exception as e:
        logger.error(f"✗ Error loading FactoryNet: {e}")
        return False


def check_nist_ur5():
    """Check if NIST UR5 data is available."""
    logger.info("=" * 60)
    logger.info("Checking NIST UR5 dataset...")

    # Check if data directory exists
    data_dir = Path("data/nist_ur5")
    if data_dir.exists():
        files = list(data_dir.glob("*.csv"))
        logger.info(f"  ✓ Found {len(files)} CSV files in {data_dir}")
        return True
    else:
        logger.warning(f"  ? NIST UR5 data not found at {data_dir}")
        logger.info("  Download from: https://data.nist.gov/pdr/lps/754A77D9DA1E771AE0532457068179851962")
        return False


def load_cwru_sample():
    """Try to load a sample from CWRU dataset."""
    logger.info("=" * 60)
    logger.info("Testing CWRU data loading...")

    try:
        from phmd import datasets

        cwru = datasets.Dataset("CWRU")
        logger.info(f"  Description: {cwru.describe()[:200]}...")

        # Try loading
        data = cwru.load()
        logger.info(f"  ✓ Loaded CWRU: {type(data)}")

        if hasattr(data, 'shape'):
            logger.info(f"  Shape: {data.shape}")
        elif isinstance(data, dict):
            logger.info(f"  Keys: {list(data.keys())[:5]}")

        return True
    except Exception as e:
        logger.error(f"  ✗ Error loading CWRU: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("MANY-TO-1 TRANSFER LEARNING: DATASET SETUP")
    logger.info("=" * 60)

    results = {}

    # Check all data sources
    results['phmd'] = check_phmd()
    results['uci'] = check_ucimlrepo()
    results['factorynet'] = check_factorynet()
    results['nist'] = check_nist_ur5()

    # Try loading CWRU as detailed test
    if results['phmd']:
        results['cwru_load'] = load_cwru_sample()

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for name, status in results.items():
        symbol = "✓" if status else "✗"
        logger.info(f"  {symbol} {name}")

    all_ok = all(results.values())

    if all_ok:
        logger.info("")
        logger.info("All datasets ready! You can start overnight research.")
        logger.info("Next: Run experiments/01_bearing_baseline.py")
    else:
        logger.info("")
        logger.info("Some datasets missing. Install required packages:")
        logger.info("  pip install phmd ucimlrepo")
        logger.info("  # Then download NIST UR5 manually")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

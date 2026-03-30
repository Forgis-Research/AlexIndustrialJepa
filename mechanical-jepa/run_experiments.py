"""
Autonomous experiment runner.

Runs a sequence of experiments:
1. Multi-seed baseline validation (seeds 42, 123, 456)
2. Cross-dataset transfer evaluation
3. Architecture variations
4. Masking strategy experiments
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✓ Success ({elapsed/60:.1f} min)")
        return True
    else:
        print(f"\n✗ Failed (exit code {result.returncode})")
        return False


def main():
    print("="*60)
    print("MECHANICAL-JEPA AUTONOMOUS EXPERIMENT RUNNER")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    experiments = []
    results = {}

    # ========================================================================
    # Phase 1: Multi-seed baseline validation
    # ========================================================================

    print("\n" + "="*60)
    print("PHASE 1: MULTI-SEED BASELINE VALIDATION")
    print("="*60)

    seeds = [42, 123, 456]

    for seed in seeds:
        cmd = [
            sys.executable, 'train.py',
            '--epochs', '30',
            '--seed', str(seed),
            '--no-wandb'
        ]

        success = run_command(cmd, f"Seed {seed} - Baseline (30 epochs)")
        results[f'baseline_seed_{seed}'] = 'success' if success else 'failed'

        if not success:
            print(f"\nWarning: Seed {seed} failed, continuing...")

    # ========================================================================
    # Phase 2: Cross-dataset transfer
    # ========================================================================

    print("\n" + "="*60)
    print("PHASE 2: CROSS-DATASET TRANSFER")
    print("="*60)

    # Find latest checkpoint
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob('*.pt'))
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"Using checkpoint: {latest_checkpoint.name}")

            # Mode A: Degradation detection
            cmd = [
                sys.executable, 'transfer_eval.py',
                '--checkpoint', str(latest_checkpoint),
                '--mode', 'degradation'
            ]
            success = run_command(cmd, "Cross-dataset transfer: CWRU → IMS (degradation)")
            results['transfer_degradation'] = 'success' if success else 'failed'

            # Mode B: Embedding visualization
            cmd = [
                sys.executable, 'transfer_eval.py',
                '--checkpoint', str(latest_checkpoint),
                '--mode', 'embedding'
            ]
            success = run_command(cmd, "Cross-dataset transfer: Embedding analysis")
            results['transfer_embedding'] = 'success' if success else 'failed'

        else:
            print("No checkpoints found, skipping transfer evaluation")
            results['transfer_degradation'] = 'skipped'
            results['transfer_embedding'] = 'skipped'
    else:
        print("No checkpoint directory, skipping transfer evaluation")
        results['transfer_degradation'] = 'skipped'
        results['transfer_embedding'] = 'skipped'

    # ========================================================================
    # Phase 3: Architecture variations (if time permits)
    # ========================================================================

    print("\n" + "="*60)
    print("PHASE 3: ARCHITECTURE VARIATIONS")
    print("="*60)

    variations = [
        (['--encoder-depth', '6'], "Deeper encoder (depth=6)"),
        (['--embed-dim', '512'], "Larger embeddings (dim=512)"),
        (['--mask-ratio', '0.7'], "Higher masking (ratio=0.7)"),
    ]

    for args, description in variations:
        cmd = [
            sys.executable, 'train.py',
            '--epochs', '30',
            '--seed', '42',
            '--no-wandb'
        ] + args

        success = run_command(cmd, description)
        results[f'variation_{description.split()[0].lower()}'] = 'success' if success else 'failed'

        if not success:
            print(f"\nWarning: {description} failed, continuing...")

    # ========================================================================
    # Phase 4: Enhanced training with transfer
    # ========================================================================

    print("\n" + "="*60)
    print("PHASE 4: ENHANCED TRAINING WITH BUILT-IN TRANSFER")
    print("="*60)

    cmd = [
        sys.executable, 'train_transfer.py',
        '--mode', 'transfer',
        '--epochs', '30',
        '--seed', '42',
        '--no-wandb'
    ]

    success = run_command(cmd, "Enhanced training with transfer mode")
    results['enhanced_transfer'] = 'success' if success else 'failed'

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for name, status in results.items():
        symbol = "✓" if status == "success" else "✗" if status == "failed" else "⊘"
        print(f"  {symbol} {name}: {status}")

    # Save results
    results_file = Path('results') / f'experiment_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExperiment runner interrupted by user")
        sys.exit(1)

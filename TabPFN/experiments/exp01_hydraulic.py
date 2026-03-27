"""
Experiment 01: TabPFN-TS on UCI Hydraulic System
================================================

Quick assessment of TabPFN-TS forecasting capability on hydraulic pressure sensors.

Usage:
    python exp01_hydraulic.py [--synthetic] [--cycles N]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

# Optional: suppress warnings
import warnings
warnings.filterwarnings('ignore')


def generate_synthetic_hydraulic(n_timesteps=600, n_cycles=5):
    """Generate synthetic hydraulic-like pressure data."""
    all_data = []
    for _ in range(n_cycles):
        t = np.linspace(0, 60, n_timesteps)
        # Pressure-like signal
        base = 100 + 20 * np.sin(2 * np.pi * 0.05 * t)
        transient = 30 * np.exp(-0.5 * t) * (t < 10)
        noise = 2 * np.random.randn(n_timesteps)
        all_data.append(base + transient + noise)
    return np.array(all_data)


def load_hydraulic_data(data_dir, sensor='PS1', n_cycles=None):
    """Load real hydraulic data if available."""
    file_path = data_dir / f'{sensor}.txt'
    if not file_path.exists():
        return None

    data = np.loadtxt(file_path)
    if n_cycles is not None:
        data = data[:n_cycles]
    return data


def naive_baseline(y_train, horizon):
    """Last value naive baseline."""
    return np.full(horizon, y_train[-1])


def seasonal_naive(y_train, horizon, period=20):
    """Seasonal naive baseline."""
    pattern = y_train[-period:]
    n_repeats = horizon // period + 1
    return np.tile(pattern, n_repeats)[:horizon]


def moving_average(y_train, horizon, window=20):
    """Moving average baseline."""
    return np.full(horizon, np.mean(y_train[-window:]))


def evaluate_forecasts(y_true, predictions_dict):
    """Calculate metrics for all methods."""
    results = {}
    for name, pred in predictions_dict.items():
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        results[name] = {'RMSE': rmse, 'MAE': mae}
    return results


def run_experiment(data, subsample=10, train_frac=0.8):
    """Run forecasting experiment on a single cycle."""

    # Subsample for speed
    y = data[::subsample]
    n = len(y)

    # Train/test split
    train_size = int(train_frac * n)
    y_train = y[:train_size]
    y_test = y[train_size:]
    horizon = len(y_test)

    if horizon == 0:
        return None

    predictions = {}

    # Baselines
    predictions['Naive (last)'] = naive_baseline(y_train, horizon)
    predictions['Seasonal naive'] = seasonal_naive(y_train, horizon)
    predictions['Moving avg'] = moving_average(y_train, horizon)

    # TabPFN-TS
    try:
        from tabpfn_ts import TabPFNForecaster
        forecaster = TabPFNForecaster(horizon=horizon)
        forecaster.fit(y_train)
        predictions['TabPFN-TS'] = forecaster.predict()
    except ImportError:
        print("TabPFN-TS not installed. Run: pip install tabpfn-time-series")
        predictions['TabPFN-TS'] = np.full(horizon, np.nan)
    except Exception as e:
        print(f"TabPFN-TS error: {e}")
        predictions['TabPFN-TS'] = np.full(horizon, np.nan)

    # Evaluate
    results = evaluate_forecasts(y_test, predictions)

    return {
        'y_train': y_train,
        'y_test': y_test,
        'predictions': predictions,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description='TabPFN-TS Hydraulic Experiment')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--cycles', type=int, default=5, help='Number of cycles to test')
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'datasets' / 'data' / 'hydraulic'

    # Load or generate data
    if args.synthetic or not data_dir.exists():
        print("Using synthetic hydraulic data...")
        data = generate_synthetic_hydraulic(n_cycles=args.cycles)
    else:
        print(f"Loading real hydraulic data from {data_dir}...")
        data = load_hydraulic_data(data_dir, n_cycles=args.cycles)
        if data is None:
            print("Could not load data. Using synthetic...")
            data = generate_synthetic_hydraulic(n_cycles=args.cycles)

    print(f"Data shape: {data.shape}")
    print(f"Running experiment on {len(data)} cycles...\n")

    # Run experiments
    all_results = []
    for i, cycle_data in enumerate(data):
        result = run_experiment(cycle_data)
        if result is not None:
            all_results.append(result)
            print(f"Cycle {i}: ", end='')
            for name, metrics in result['results'].items():
                print(f"{name}={metrics['RMSE']:.3f} ", end='')
            print()

    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATE RESULTS (mean RMSE across cycles)")
    print("="*60)

    methods = list(all_results[0]['results'].keys())
    for method in methods:
        rmses = [r['results'][method]['RMSE'] for r in all_results]
        rmses = [x for x in rmses if not np.isnan(x)]
        if rmses:
            print(f"{method:20s}: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Skill scores vs naive
    print("\n" + "="*60)
    print("SKILL SCORES (vs Naive)")
    print("="*60)

    naive_rmses = [r['results']['Naive (last)']['RMSE'] for r in all_results]
    for method in methods:
        if method == 'Naive (last)':
            continue
        method_rmses = [r['results'][method]['RMSE'] for r in all_results]
        skills = []
        for nr, mr in zip(naive_rmses, method_rmses):
            if not np.isnan(mr) and nr > 0:
                skills.append(1 - mr / nr)
        if skills:
            print(f"{method:20s}: {np.mean(skills):.3f} ± {np.std(skills):.3f}")

    # Plot last cycle
    if all_results:
        last = all_results[-1]
        n_train = len(last['y_train'])
        t_all = np.arange(n_train + len(last['y_test']))

        plt.figure(figsize=(12, 5))
        plt.plot(t_all[:n_train], last['y_train'], 'b-', label='Training', linewidth=0.8)
        plt.plot(t_all[n_train:], last['y_test'], 'g-', label='True', linewidth=1.5)

        for name, pred in last['predictions'].items():
            if not np.any(np.isnan(pred)):
                plt.plot(t_all[n_train:], pred, '--', label=name, linewidth=1.2)

        plt.axvline(x=n_train, color='gray', linestyle=':', alpha=0.7)
        plt.xlabel('Time (samples)')
        plt.ylabel('Pressure (bar)')
        plt.title('Hydraulic Pressure Forecasting - Last Cycle')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(__file__).parent / 'exp01_results.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
        plt.show()


if __name__ == '__main__':
    main()

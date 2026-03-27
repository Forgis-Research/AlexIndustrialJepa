"""
Baseline forecasting methods for comparison with TabPFN-TS.
"""

import numpy as np
from typing import Optional


class NaiveForecaster:
    """Last-value naive baseline."""

    def __init__(self):
        self.last_value = None

    def fit(self, y):
        self.last_value = y[-1]
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.last_value)


class SeasonalNaiveForecaster:
    """Seasonal naive baseline - repeats last period."""

    def __init__(self, period: int = 20):
        self.period = period
        self.pattern = None

    def fit(self, y):
        self.pattern = y[-self.period:]
        return self

    def predict(self, horizon: int):
        n_repeats = horizon // self.period + 1
        return np.tile(self.pattern, n_repeats)[:horizon]


class MovingAverageForecaster:
    """Moving average baseline."""

    def __init__(self, window: int = 20):
        self.window = window
        self.mean_value = None

    def fit(self, y):
        self.mean_value = np.mean(y[-self.window:])
        return self

    def predict(self, horizon: int):
        return np.full(horizon, self.mean_value)


class LinearTrendForecaster:
    """Linear extrapolation baseline."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.intercept = None
        self.slope = None

    def fit(self, y):
        recent = y[-self.lookback:]
        t = np.arange(len(recent))
        self.slope = np.polyfit(t, recent, 1)[0]
        self.intercept = recent[-1]
        return self

    def predict(self, horizon: int):
        return self.intercept + self.slope * np.arange(1, horizon + 1)


class ARIMAForecaster:
    """ARIMA baseline (requires statsmodels)."""

    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, y):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y, order=self.order).fit()
        except ImportError:
            print("statsmodels not installed. Run: pip install statsmodels")
            self.model = None
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            self.model = None
        return self

    def predict(self, horizon: int):
        if self.model is None:
            return np.full(horizon, np.nan)
        return self.model.forecast(steps=horizon)


def get_all_baselines(period: Optional[int] = 20) -> dict:
    """Get dictionary of all baseline forecasters."""
    return {
        'Naive': NaiveForecaster(),
        'Seasonal Naive': SeasonalNaiveForecaster(period=period),
        'Moving Average': MovingAverageForecaster(window=period),
        'Linear Trend': LinearTrendForecaster(lookback=period),
        'ARIMA': ARIMAForecaster(),
    }

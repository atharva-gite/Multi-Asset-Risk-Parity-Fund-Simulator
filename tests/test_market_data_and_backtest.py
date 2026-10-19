"""Unit tests for market-data validation, demo data, and backtesting."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_simulator.backtesting.engine import run_erc_backtest, run_fixed_weight_backtest
from risk_simulator.data.market_data import (
    generate_synthetic_demo_prices,
    validate_market_prices,
)
from risk_simulator.visualization.charts import cumulative_returns, drawdown_series


class MarketDataTests(unittest.TestCase):
    def test_validation_sorts_dates_and_preserves_missing_values(self) -> None:
        prices = pd.DataFrame(
            {
                "A": [101.0, 100.0, np.nan, 103.0],
                "B": [201.0, 200.0, 202.0, 203.0],
            },
            index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04"]),
        )
        validated = validate_market_prices(prices)
        self.assertTrue(validated.index.is_monotonic_increasing)
        self.assertTrue(pd.isna(validated.loc["2024-01-02", "A"]))
        self.assertEqual(len(validated), 4)

    def test_invalid_prices_are_rejected(self) -> None:
        duplicate_dates = pd.DataFrame(
            {"A": [1.0, 2.0]}, index=pd.to_datetime(["2024-01-01", "2024-01-01"])
        )
        with self.assertRaisesRegex(ValueError, "duplicate dates"):
            validate_market_prices(duplicate_dates)
        nonpositive = pd.DataFrame({"A": [1.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            validate_market_prices(nonpositive)

    def test_synthetic_prices_are_reproducible(self) -> None:
        first = generate_synthetic_demo_prices("2024-01-01", "2024-02-01", ["SPY", "IEF"])
        second = generate_synthetic_demo_prices("2024-01-01", "2024-02-01", ["SPY", "IEF"])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(list(first.columns), ["SPY", "IEF"])


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(43)
        self.index = pd.bdate_range("2024-01-01", periods=90)
        self.returns = pd.DataFrame(
            rng.multivariate_normal(
                [0.0001, 0.00005],
                [[0.00012, 0.00002], [0.00002, 0.00004]],
                size=len(self.index),
            ),
            index=self.index,
            columns=["SPY", "IEF"],
        )

    def test_erc_backtest_excludes_rebalance_day_return_from_weights(self) -> None:
        rebalance_date = pd.Timestamp("2024-02-01")
        changed = self.returns.copy()
        changed.loc[rebalance_date, "SPY"] = 0.75
        baseline = run_erc_backtest(
            self.returns,
            lookback=10,
            target_annual_volatility=0.10,
            transaction_cost_bps=0,
        )
        shocked = run_erc_backtest(
            changed,
            lookback=10,
            target_annual_volatility=0.10,
            transaction_cost_bps=0,
        )
        self.assertIn(rebalance_date, baseline.weights.index)
        np.testing.assert_allclose(
            baseline.weights.loc[rebalance_date].to_numpy(),
            shocked.weights.loc[rebalance_date].to_numpy(),
            atol=1e-10,
        )
        self.assertTrue((baseline.results["leverage"] <= 2.0 + 1e-10).all())

    def test_fixed_weight_backtest_charges_initial_turnover_cost(self) -> None:
        zero_returns = pd.DataFrame(0.0, index=self.index[:3], columns=["SPY", "IEF"])
        result = run_fixed_weight_backtest(
            zero_returns,
            {"SPY": 0.6, "IEF": 0.4},
            transaction_cost_bps=100,
        )
        self.assertAlmostEqual(result.results.iloc[0]["turnover"], 1.0)
        self.assertAlmostEqual(result.results.iloc[0]["net_return"], -0.01)
        self.assertAlmostEqual(result.results.iloc[1]["net_return"], 0.0)

    def test_invalid_fixed_weights_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            run_fixed_weight_backtest(self.returns.iloc[:3], {"SPY": 0.7, "IEF": 0.4})


class ChartHelperTests(unittest.TestCase):
    def test_cumulative_return_and_drawdown(self) -> None:
        index = pd.date_range("2024-01-01", periods=3)
        returns = pd.Series([0.10, -0.20, 0.10], index=index)
        curves = cumulative_returns({"strategy": returns})
        self.assertAlmostEqual(curves.iloc[-1, 0], 1.1 * 0.8 * 1.1 - 1.0)
        drawdowns = drawdown_series(returns)
        self.assertAlmostEqual(drawdowns.iloc[1], -0.20)
        self.assertLess(drawdowns.min(), 0.0)


if __name__ == "__main__":
    unittest.main()

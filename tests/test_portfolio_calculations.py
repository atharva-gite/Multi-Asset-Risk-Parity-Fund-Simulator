"""Unit tests for implemented return, risk, and ERC calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from risk_simulator.portfolio.optimization import solve_erc_weights
from risk_simulator.portfolio.returns import prices_to_returns
from risk_simulator.portfolio.risk import (
    estimate_covariance,
    portfolio_volatility,
    risk_contributions,
)


class ReturnCalculationTests(unittest.TestCase):
    def test_simple_and_log_returns(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        simple = prices_to_returns(prices)
        log = prices_to_returns(prices, method="log")
        self.assertAlmostEqual(simple.iloc[0, 0], 0.10)
        self.assertAlmostEqual(log.iloc[0, 0], np.log(1.10))
        self.assertEqual(len(simple), 2)

    def test_missing_prices_are_not_forward_filled(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 110.0, np.nan, 121.0]})
        returns = prices_to_returns(prices)
        self.assertAlmostEqual(returns.iloc[0, 0], 0.10)
        self.assertTrue(returns.iloc[1:, 0].isna().all())

    def test_invalid_return_inputs_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two"):
            prices_to_returns(pd.DataFrame({"A": [100.0]}))
        with self.assertRaisesRegex(ValueError, "method"):
            prices_to_returns(pd.DataFrame({"A": [100.0, 101.0]}), method="other")
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            prices_to_returns(pd.DataFrame({"A": [1.0, 0.0]}), method="log")


class RiskCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.covariance = pd.DataFrame(
            [[0.04, 0.006], [0.006, 0.01]],
            index=["A", "B"],
            columns=["A", "B"],
        )

    def test_covariance_annualization_and_complete_rows(self) -> None:
        returns = pd.DataFrame({"A": [0.01, 0.02, np.nan, -0.01], "B": [0.0, 0.01, 0.03, -0.02]})
        covariance = estimate_covariance(returns, periods_per_year=10, min_observations=3)
        complete_covariance = returns.dropna().cov(ddof=1) * 10
        pd.testing.assert_frame_equal(covariance, complete_covariance)

    def test_covariance_requires_enough_complete_rows(self) -> None:
        returns = pd.DataFrame({"A": [0.01, np.nan], "B": [0.02, 0.03]})
        with self.assertRaisesRegex(ValueError, "complete return observations"):
            estimate_covariance(returns, min_observations=2)

    def test_risk_contributions_sum_to_portfolio_volatility(self) -> None:
        weights = np.array([0.35, 0.65])
        volatility = portfolio_volatility(weights, self.covariance)
        contributions = risk_contributions(weights, self.covariance)
        self.assertAlmostEqual(contributions["risk_contribution"].sum(), volatility)
        self.assertAlmostEqual(contributions["risk_contribution_pct"].sum(), 1.0)

    def test_zero_volatility_and_invalid_weights(self) -> None:
        zero_covariance = np.zeros((2, 2))
        self.assertEqual(portfolio_volatility([0.5, 0.5], zero_covariance), 0.0)
        with self.assertRaisesRegex(ValueError, "zero-volatility"):
            risk_contributions([0.5, 0.5], zero_covariance)
        with self.assertRaisesRegex(ValueError, "matching"):
            portfolio_volatility([1.0], self.covariance)
        with self.assertRaisesRegex(ValueError, "finite"):
            portfolio_volatility([np.nan, 1.0], self.covariance)

    def test_non_positive_semidefinite_covariance_is_rejected(self) -> None:
        invalid = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            portfolio_volatility([0.5, 0.5], invalid)


class OptimizationTests(unittest.TestCase):
    def test_diagonal_covariance_has_inverse_volatility_erc_weights(self) -> None:
        covariance = np.diag([0.04, 0.01])
        result = solve_erc_weights(covariance)
        self.assertTrue(result.success, result.message)
        self.assertIsNotNone(result.weights)
        np.testing.assert_allclose(result.weights, [1.0 / 3.0, 2.0 / 3.0], atol=2e-3)
        self.assertAlmostEqual(float(result.weights.sum()), 1.0)
        self.assertTrue((result.weights >= 0).all())

    def test_singular_covariance_fails_without_fallback_weights(self) -> None:
        singular = np.array([[0.04, 0.04], [0.04, 0.04]])
        result = solve_erc_weights(singular)
        self.assertFalse(result.success)
        self.assertIsNone(result.weights)
        self.assertIn("positive-definite", result.message)

    def test_single_asset_is_fully_invested(self) -> None:
        result = solve_erc_weights(np.array([[0.04]]))
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.weights, [1.0])


if __name__ == "__main__":
    unittest.main()

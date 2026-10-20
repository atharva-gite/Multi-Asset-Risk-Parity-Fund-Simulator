"""Simple daily backtests with explicit rebalance and cost conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from risk_simulator.config import TRADING_DAYS_PER_YEAR
from risk_simulator.portfolio.optimization import solve_erc_weights
from risk_simulator.portfolio.risk import estimate_covariance, portfolio_volatility


@dataclass(frozen=True)
class BacktestResult:
    """Net/gross return series and target exposures used by the backtest."""

    results: pd.DataFrame
    weights: pd.DataFrame


def _complete_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty or returns.columns.has_duplicates:
        raise ValueError("Returns must be a non-empty DataFrame with unique columns.")
    values = returns.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Returns cannot contain infinite values.")
    if (values <= -1.0).any().any():
        raise ValueError("Simple returns must be greater than -100%.")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("Returns must use a DatetimeIndex.")
    if values.index.has_duplicates:
        raise ValueError("Returns contain duplicate dates.")
    values = values.sort_index().dropna(how="any")
    if values.empty:
        raise ValueError("No complete return observations are available.")
    return values


def _period(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    key = frequency.strip().lower()
    if key in {"monthly", "month", "m"}:
        return index.to_period("M")
    if key in {"quarterly", "quarter", "q"}:
        return index.to_period("Q")
    raise ValueError("Rebalancing frequency must be 'Monthly' or 'Quarterly'.")


def _finish_result(
    records: list[dict[str, float]],
    exposures: list[np.ndarray],
    dates: list[pd.Timestamp],
    columns: pd.Index,
) -> BacktestResult:
    if not records:
        raise ValueError("No backtest observations were produced; check date range and lookback.")
    result_frame = pd.DataFrame(records, index=pd.DatetimeIndex(dates))
    weight_frame = pd.DataFrame(exposures, index=pd.DatetimeIndex(dates), columns=columns)
    return BacktestResult(result_frame, weight_frame)


def run_erc_backtest(
    returns: pd.DataFrame,
    lookback: int = 63,
    rebalance_frequency: str = "Monthly",
    target_annual_volatility: float = 0.10,
    max_leverage: float = 2.0,
    transaction_cost_bps: float = 5.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    evaluation_start: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Backtest rolling ERC weights without using the rebalance-day return.

    At a rebalance date, the optimizer sees exactly the preceding lookback
    complete return rows. Exposures are set before that date's return. The
    volatility target scales the fully invested ERC sleeve and caps gross
    exposure at max_leverage. Costs are charged on gross traded exposure
    (the L1 change in exposures) at each rebalance. Financing and market impact
    are not modeled.
    """
    data = _complete_returns(returns)
    if lookback < 2:
        raise ValueError("lookback must be at least two observations.")
    if target_annual_volatility <= 0 or max_leverage <= 0:
        raise ValueError("Target volatility and max leverage must be positive.")
    if transaction_cost_bps < 0:
        raise ValueError("Transaction costs cannot be negative.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    periods = _period(data.index, rebalance_frequency)
    start = pd.Timestamp(evaluation_start) if evaluation_start is not None else data.index[0]
    current_exposure: np.ndarray | None = None
    previous_period: pd.Period | None = None
    records: list[dict[str, float]] = []
    exposures: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    tickers = data.columns

    for i, (date_value, row) in enumerate(data.iterrows()):
        period_value = periods[i]
        turnover = 0.0
        cost = 0.0
        if period_value != previous_period and i >= lookback:
            training = data.iloc[i - lookback : i]
            covariance = estimate_covariance(
                training, periods_per_year=periods_per_year, min_observations=lookback
            )
            optimized = solve_erc_weights(covariance)
            if not optimized.success or optimized.weights is None:
                raise RuntimeError(
                    f"ERC rebalance on {date_value.date()} failed: {optimized.message}"
                )
            base_vol = portfolio_volatility(optimized.weights, covariance)
            if base_vol <= 0:
                raise RuntimeError(
                    f"ERC rebalance on {date_value.date()} has zero estimated volatility."
                )
            leverage = min(target_annual_volatility / base_vol, max_leverage)
            new_exposure = optimized.weights * leverage
            turnover = (
                float(np.abs(new_exposure).sum())
                if current_exposure is None
                else float(np.abs(new_exposure - current_exposure).sum())
            )
            cost = turnover * transaction_cost_bps / 10_000.0
            current_exposure = new_exposure
        previous_period = period_value
        if current_exposure is None:
            continue
        asset_returns = row.to_numpy(dtype=float)
        gross_return = float(current_exposure @ asset_returns)
        if gross_return <= -1.0:
            raise RuntimeError(f"Portfolio NAV reached zero or below on {date_value.date()}.")
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        if date_value >= start:
            records.append(
                {
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "leverage": float(np.abs(current_exposure).sum()),
                    "turnover": turnover,
                    "transaction_cost": cost,
                }
            )
            exposures.append(current_exposure.copy())
            dates.append(date_value)
        # Drift holdings to the next close. Financing is excluded by convention.
        current_exposure = current_exposure * (1.0 + asset_returns) / (1.0 + gross_return)
    return _finish_result(records, exposures, dates, tickers)


def run_fixed_weight_backtest(
    returns: pd.DataFrame,
    target_weights: Mapping[str, float],
    rebalance_frequency: str = "Monthly",
    transaction_cost_bps: float = 5.0,
    evaluation_start: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Backtest a preset long-only portfolio with calendar rebalancing.

    The 60/40 comparison uses preset 60% SPY and 40% IEF weights and rebalances
    at the first available observation of each month. It has no estimated
    signals or look-ahead. Transaction cost uses gross L1 traded exposure.
    """
    data = _complete_returns(returns)
    if not target_weights:
        raise ValueError("At least one target weight is required.")
    if transaction_cost_bps < 0:
        raise ValueError("Transaction costs cannot be negative.")
    missing = sorted(set(target_weights) - set(data.columns))
    if missing:
        raise ValueError(f"Returns are missing benchmark instruments: {', '.join(missing)}.")
    target = np.asarray([target_weights[column] for column in target_weights], dtype=float)
    if not np.isfinite(target).all() or (target < 0).any() or not np.isclose(target.sum(), 1.0):
        raise ValueError("Fixed target weights must be finite, non-negative, and sum to one.")
    columns = pd.Index(list(target_weights))
    selected = data.loc[:, columns]
    periods = _period(selected.index, rebalance_frequency)
    start = pd.Timestamp(evaluation_start) if evaluation_start is not None else selected.index[0]
    current: np.ndarray | None = None
    previous_period: pd.Period | None = None
    records: list[dict[str, float]] = []
    exposures: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    for i, (date_value, row) in enumerate(selected.iterrows()):
        period_value = periods[i]
        turnover = 0.0
        cost = 0.0
        if period_value != previous_period:
            turnover = (
                float(np.abs(target).sum())
                if current is None
                else float(np.abs(target - current).sum())
            )
            cost = turnover * transaction_cost_bps / 10_000.0
            current = target.copy()
        previous_period = period_value
        assert current is not None
        asset_returns = row.to_numpy(dtype=float)
        gross_return = float(current @ asset_returns)
        if gross_return <= -1.0:
            raise RuntimeError(f"Benchmark NAV reached zero or below on {date_value.date()}.")
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0
        if date_value >= start:
            records.append(
                {
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "leverage": float(np.abs(current).sum()),
                    "turnover": turnover,
                    "transaction_cost": cost,
                }
            )
            exposures.append(current.copy())
            dates.append(date_value)
        current = current * (1.0 + asset_returns) / (1.0 + gross_return)
    return _finish_result(records, exposures, dates, columns)

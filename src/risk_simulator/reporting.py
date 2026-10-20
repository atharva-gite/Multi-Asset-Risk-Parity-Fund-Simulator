"""Reproducibility metadata and exports for simulator runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

from risk_simulator.config import TRADING_DAYS_PER_YEAR
from risk_simulator.portfolio.returns import prices_to_returns

_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "streamlit": "streamlit",
    "yfinance": "yfinance",
}


def price_data_sha256(prices: pd.DataFrame) -> str:
    """Return a SHA-256 fingerprint of the ordered date-indexed price table."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be a pandas DataFrame.")
    normalized = prices.copy()
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    normalized = normalized.sort_index()
    serialized = normalized.to_csv(
        index=True,
        index_label="date",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
        float_format="%.17g",
        na_rep="NaN",
        lineterminator="\n",
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for import_name, distribution_name in _DISTRIBUTIONS.items():
        try:
            versions[import_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            versions[import_name] = None
    return versions


def _application_version() -> str:
    try:
        return importlib.metadata.version("multi-asset-risk-parity-simulator")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def build_run_manifest(
    prices: pd.DataFrame,
    *,
    data_source: str,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    tickers: list[str] | tuple[str, ...],
    lookback: int,
    rebalance_frequency: str,
    target_annual_volatility: float,
    max_leverage: float,
    transaction_cost_bps: float,
    benchmark_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Build JSON-safe provenance for the exact inputs and selected assumptions."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must use a DatetimeIndex to create a run manifest.")

    missing_cells = int(prices.isna().sum().sum())
    evaluation_start_ts = pd.Timestamp(evaluation_start)
    evaluation_end_ts = pd.Timestamp(evaluation_end)
    complete_returns = prices_to_returns(prices, method="simple").dropna(how="any")
    evaluation_returns = complete_returns.loc[
        (complete_returns.index >= evaluation_start_ts)
        & (complete_returns.index <= evaluation_end_ts)
    ]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "application": {
            "name": "Multi-Asset Risk Parity Fund Simulator",
            "version": _application_version(),
            "python_version": platform.python_version(),
            "dependencies": _dependency_versions(),
        },
        "data": {
            "source": data_source,
            "requested_evaluation_start": pd.Timestamp(evaluation_start).date().isoformat(),
            "requested_evaluation_end": pd.Timestamp(evaluation_end).date().isoformat(),
            "observed_price_start": prices.index.min().date().isoformat(),
            "observed_price_end": prices.index.max().date().isoformat(),
            "price_rows": int(len(prices)),
            "complete_daily_return_observations": int(len(complete_returns)),
            "evaluation_observations": int(len(evaluation_returns)),
            "instruments": [str(ticker) for ticker in prices.columns],
            "missing_price_cells": missing_cells,
            "price_table_sha256": price_data_sha256(prices),
            "price_hash_format": "UTF-8 CSV; sorted dates; 17 significant float digits; NaN marker",
        },
        "strategy": {
            "requested_instruments": [str(ticker) for ticker in tickers],
            "return_method": "simple close-to-close return",
            "missing_data": "no price forward-fill; incomplete synchronized return rows excluded",
            "covariance_estimator": "annualized sample covariance of complete daily returns",
            "trading_days_per_year": TRADING_DAYS_PER_YEAR,
            "lookback_observations": int(lookback),
            "rebalance_frequency": rebalance_frequency,
            "target_annual_volatility": float(target_annual_volatility),
            "maximum_gross_exposure": float(max_leverage),
            "transaction_cost_bps_per_gross_traded_exposure": float(transaction_cost_bps),
            "transaction_cost_convention": "L1 exposure turnover multiplied by basis-point rate",
            "financing_costs_included": False,
        },
        "benchmark": (
            {
                "available_in_export": True,
                "weights": {str(key): float(value) for key, value in benchmark_weights.items()},
                "rebalance_frequency": rebalance_frequency,
                "transaction_cost_bps_per_gross_traded_exposure": float(transaction_cost_bps),
            }
            if benchmark_weights is not None
            else {"available_in_export": False}
        ),
    }


def manifest_json(manifest: Mapping[str, object]) -> str:
    """Serialize a manifest consistently for download and review."""
    return json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"

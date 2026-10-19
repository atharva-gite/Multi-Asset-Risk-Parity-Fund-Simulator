"""Covariance, volatility, and Euler risk-contribution calculations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from risk_simulator.config import TRADING_DAYS_PER_YEAR


def _covariance_array(
    covariance: pd.DataFrame | np.ndarray,
    *,
    require_positive_definite: bool = False,
) -> np.ndarray:
    matrix = covariance.to_numpy(dtype=float) if isinstance(covariance, pd.DataFrame) else np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Covariance must be a non-empty square matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance must contain only finite values.")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-10 * scale
    if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=tolerance):
        raise ValueError("Covariance must be symmetric.")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -tolerance:
        raise ValueError("Covariance must be positive semidefinite.")
    if require_positive_definite and eigenvalues.min() <= tolerance:
        raise ValueError("ERC optimization requires a positive-definite covariance matrix.")
    return matrix


def estimate_covariance(
    returns: pd.DataFrame,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    min_observations: int = 20,
) -> pd.DataFrame:
    """Estimate annualized sample covariance from complete return rows.

    Rows with any missing asset return are excluded rather than imputed.
    """
    if not isinstance(returns, pd.DataFrame) or returns.empty or returns.columns.has_duplicates:
        raise ValueError("Returns must be a non-empty DataFrame with unique columns.")
    if periods_per_year <= 0 or min_observations < 2:
        raise ValueError("Annualization and minimum observations must be positive.")
    values = returns.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Returns cannot contain infinite values.")
    complete = values.dropna(how="any")
    if len(complete) < min_observations:
        raise ValueError(
            f"At least {min_observations} complete return observations are required; "
            f"found {len(complete)}."
        )
    covariance = complete.cov(ddof=1) * periods_per_year
    _covariance_array(covariance)
    return covariance


def portfolio_volatility(
    weights: Sequence[float] | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
) -> float:
    """Return annualized portfolio volatility for a covariance matrix."""
    matrix = _covariance_array(covariance)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or len(weight_array) != matrix.shape[0]:
        raise ValueError("Weights must be a vector matching the covariance dimensions.")
    if not np.isfinite(weight_array).all():
        raise ValueError("Weights must be finite.")
    variance = float(weight_array @ matrix @ weight_array)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(matrix))) * float(weight_array @ weight_array))
    if variance < -tolerance:
        raise ValueError("Portfolio variance is negative; check the covariance matrix.")
    return float(np.sqrt(max(variance, 0.0)))


def risk_contributions(
    weights: Sequence[float] | np.ndarray,
    covariance: pd.DataFrame | np.ndarray,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return marginal and total Euler contributions in volatility units.

    Total contributions sum to portfolio volatility (Euler's theorem), subject
    to numerical tolerance. Weights may represent scaled exposures.
    """
    matrix = _covariance_array(covariance)
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim != 1 or len(weight_array) != matrix.shape[0]:
        raise ValueError("Weights must be a vector matching the covariance dimensions.")
    if not np.isfinite(weight_array).all():
        raise ValueError("Weights must be finite.")
    volatility = portfolio_volatility(weight_array, matrix)
    if volatility <= 1e-14:
        raise ValueError("Risk contributions are undefined for zero-volatility portfolios.")
    marginal = matrix @ weight_array / volatility
    total = weight_array * marginal
    if labels is None and isinstance(covariance, pd.DataFrame):
        labels = [str(label) for label in covariance.columns]
    if labels is None:
        labels = [str(i) for i in range(len(weight_array))]
    if len(labels) != len(weight_array):
        raise ValueError("Labels must match the number of weights.")
    return pd.DataFrame(
        {
            "weight": weight_array,
            "marginal_risk": marginal,
            "risk_contribution": total,
            "risk_contribution_pct": total / volatility,
        },
        index=pd.Index(labels, name="asset"),
    )

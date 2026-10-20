"""Long-only equal-risk-contribution allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from risk_simulator.portfolio.risk import _covariance_array


@dataclass(frozen=True)
class OptimizationResult:
    """Explicit optimizer outcome; failed solves never carry fallback weights."""

    success: bool
    weights: np.ndarray | None
    message: str
    max_risk_share_deviation: float | None = None


def solve_erc_weights(
    covariance: pd.DataFrame | np.ndarray,
    tolerance: float = 1e-4,
    max_iterations: int = 2_000,
) -> OptimizationResult:
    """Solve for non-negative, fully invested equal-risk-contribution weights.

    The objective minimizes squared deviations of variance-contribution shares
    from 1/N. A non-positive-definite covariance or an inaccurate solver result
    is reported as a failure rather than replaced with equal weights.
    """
    try:
        matrix = _covariance_array(covariance, require_positive_definite=True)
    except ValueError as exc:
        return OptimizationResult(False, None, str(exc))
    count = matrix.shape[0]
    if count == 1:
        return OptimizationResult(True, np.ones(1), "Single-asset portfolio.")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("Tolerance and max_iterations must be positive.")
    diagonal = np.diag(matrix)
    if (diagonal <= 0).any():
        return OptimizationResult(False, None, "Every asset must have positive variance.")
    target = 1.0 / count

    def objective(weights: np.ndarray) -> float:
        marginal_variance = matrix @ weights
        contributions = weights * marginal_variance
        total_variance = float(weights @ marginal_variance)
        if total_variance <= 1e-20:
            return 1e6
        shares = contributions / total_variance
        return float(np.sum((shares - target) ** 2))

    starts = [
        np.full(count, 1.0 / count),
        (1.0 / np.sqrt(diagonal)) / np.sum(1.0 / np.sqrt(diagonal)),
    ]
    best: tuple[float, object] | None = None
    for initial in starts:
        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(1e-10, 1.0)] * count,
            constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
            options={"ftol": 1e-12, "maxiter": max_iterations, "disp": False},
        )
        score = objective(result.x) if np.isfinite(result.x).all() else float("inf")
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    score, result = best
    weights = np.asarray(result.x, dtype=float)
    if not result.success:
        return OptimizationResult(False, None, f"ERC optimizer failed: {result.message}")
    if not np.isfinite(weights).all() or (weights < -1e-8).any():
        return OptimizationResult(False, None, "ERC optimizer returned invalid weights.")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return OptimizationResult(False, None, "ERC optimizer returned zero total weight.")
    weights /= total
    deviation = float(np.sqrt(max(score, 0.0)))
    if deviation > tolerance:
        return OptimizationResult(
            False,
            None,
            f"ERC solution did not converge within tolerance "
            f"(risk-share deviation {deviation:.6g}).",
            deviation,
        )
    return OptimizationResult(True, weights, "ERC optimization converged.", deviation)

"""Return calculations with an explicit no-imputation convention."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def prices_to_returns(
    prices: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
) -> pd.DataFrame:
    """Calculate simple or log close-to-close returns without filling gaps.

    Simple return is P[t] / P[t-1] - 1. Log return is log(P[t] / P[t-1]).
    Any return spanning a missing price remains NaN; callers can explicitly use
    complete observations for synchronized portfolio calculations.
    """
    if not isinstance(prices, pd.DataFrame) or prices.shape[0] < 2:
        raise ValueError("At least two price rows are required.")
    if prices.empty or prices.columns.has_duplicates:
        raise ValueError("Price data must have unique instrument columns.")
    values = prices.astype(float)
    if np.isinf(values.to_numpy()).any() or (values < 0).any().any():
        raise ValueError("Prices must be finite or missing and cannot be negative.")
    if method == "simple":
        returns = values.pct_change(fill_method=None)
    elif method == "log":
        if (values <= 0).any().any():
            raise ValueError("Log returns require strictly positive prices.")
        returns = np.log(values).diff()
    else:
        raise ValueError("method must be 'simple' or 'log'.")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    return returns.iloc[1:]

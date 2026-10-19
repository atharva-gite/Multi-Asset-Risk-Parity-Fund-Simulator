"""Small chart-data helpers, independent of Streamlit rendering."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def cumulative_returns(return_series: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Build compounded growth curves from simple return series."""
    curves: dict[str, pd.Series] = {}
    for name, values in return_series.items():
        series = values.astype(float)
        if np.isinf(series.to_numpy()).any() or (series.dropna() <= -1.0).any():
            raise ValueError("Simple returns must be finite and greater than -100%.")
        curves[name] = (1.0 + series.fillna(0.0)).cumprod() - 1.0
    return pd.DataFrame(curves)


def drawdown_series(return_series: pd.Series) -> pd.Series:
    """Calculate drawdown from compounded net simple returns."""
    values = return_series.astype(float)
    if np.isinf(values.to_numpy()).any() or (values.dropna() <= -1.0).any():
        raise ValueError("Simple returns must be finite and greater than -100%.")
    wealth = (1.0 + values.fillna(0.0)).cumprod()
    drawdowns = wealth / wealth.cummax() - 1.0
    drawdowns.name = "drawdown"
    return drawdowns

"""Market price sources and explicit synthetic demonstration data.

Yahoo Finance data is retrieved on demand and cached in-process. Price
observations are never forward-filled. The synthetic generator is only for
software demonstration and must not be interpreted as historical performance.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from risk_simulator.config import ASSET_UNIVERSE

_SYNTHETIC_PARAMETERS: dict[str, tuple[float, float, tuple[float, float, float]]] = {
    "SPY": (0.045, 0.18, (0.82, 0.00, 0.05)),
    "VEA": (0.040, 0.19, (0.75, 0.12, 0.04)),
    "IEF": (0.025, 0.07, (-0.08, 0.68, 0.05)),
    "TLT": (0.030, 0.14, (-0.17, 0.83, 0.04)),
    "DBC": (0.035, 0.20, (0.24, 0.10, 0.68)),
    "TIP": (0.027, 0.06, (0.12, 0.42, 0.18)),
}


def validate_market_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a date-indexed price table without filling gaps.

    Missing cells are preserved for downstream complete-observation handling.
    Each instrument must have at least two finite, strictly positive prices.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("Price data must be a non-empty pandas DataFrame.")
    if prices.columns.has_duplicates:
        raise ValueError("Price data contains duplicate instrument columns.")
    result = prices.copy()
    try:
        result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    except (TypeError, ValueError) as exc:
        raise ValueError("Price data must have a valid date index.") from exc
    if result.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()
    result.columns = [str(column) for column in result.columns]
    try:
        result = result.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Price data must contain numeric values.") from exc
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Price data contains infinite values.")
    if (result < 0).any().any():
        raise ValueError("Prices cannot be negative.")
    result = result.dropna(how="all")
    if result.empty:
        raise ValueError("Price data contains no observed prices.")
    for column in result.columns:
        observed = result[column].dropna()
        if len(observed) < 2:
            raise ValueError(f"{column} has fewer than two observed prices.")
        if (observed <= 0).any():
            raise ValueError(f"{column} prices must be strictly positive.")
    return result


def generate_synthetic_demo_prices(
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Create reproducible, explicitly synthetic business-day price paths."""
    selected = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers))
    if not selected:
        raise ValueError("Select at least one instrument.")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise ValueError("The start date must be earlier than the end date.")
    dates = pd.bdate_range(start_ts, end_ts)
    if len(dates) < 2:
        raise ValueError("The selected date range must contain at least two business days.")

    rng = np.random.default_rng(20241019)
    factor_draws = rng.standard_normal((len(dates) - 1, 3))
    idiosyncratic_draws = rng.standard_normal((len(dates) - 1, len(selected)))
    daily_returns = np.empty((len(dates) - 1, len(selected)), dtype=float)
    for column, ticker in enumerate(selected):
        annual_mean, annual_vol, loadings_tuple = _SYNTHETIC_PARAMETERS.get(
            ticker, (0.03, 0.15, (0.50, 0.25, 0.25))
        )
        loadings = np.asarray(loadings_tuple, dtype=float)
        common_fraction = float(loadings @ loadings)
        if common_fraction >= 1.0:
            raise ValueError(f"Invalid synthetic factor parameters for {ticker}.")
        common = factor_draws @ (annual_vol * loadings) / np.sqrt(252.0)
        idiosyncratic_vol = annual_vol * np.sqrt(1.0 - common_fraction)
        specific = idiosyncratic_draws[:, column] * idiosyncratic_vol / np.sqrt(252.0)
        daily_returns[:, column] = annual_mean / 252.0 + common + specific

    # Avoid an impossible negative price after an extreme synthetic draw.
    daily_returns = np.maximum(daily_returns, -0.95)
    growth = np.vstack([np.ones((1, len(selected))), 1.0 + daily_returns])
    values = 100.0 * np.cumprod(growth, axis=0)
    return validate_market_prices(pd.DataFrame(values, index=dates, columns=selected))


def fetch_yahoo_prices(
    tickers: Iterable[str],
    start: date | str | pd.Timestamp,
    end: date | str | pd.Timestamp,
) -> pd.DataFrame:
    """Fetch auto-adjusted daily Yahoo Finance closes, inclusive of end date.

    Repeated calls with the same tickers and dates reuse a bounded in-process
    cache. The function raises a descriptive ValueError when data is unavailable.
    """
    selected = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers))
    if not selected:
        raise ValueError("Select at least one instrument.")
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise ValueError("The start date must be earlier than the end date.")
    return _cached_yahoo_download(
        selected,
        start_ts.strftime("%Y-%m-%d"),
        (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    ).copy()


@st.cache_data(ttl=3600, max_entries=16, show_spinner=False)
def _cached_yahoo_download(
    tickers: tuple[str, ...], start: str, end_exclusive: str
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Yahoo Finance mode requires yfinance. Install the requirements.txt dependencies."
        ) from exc
    try:
        raw = yf.download(
            tickers=list(tickers),
            start=start,
            end=end_exclusive,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance retrieval failed: {exc}") from exc
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance returned no rows for the selected tickers and dates.")
    if isinstance(raw.columns, pd.MultiIndex):
        close_frame: pd.DataFrame | pd.Series | None = None
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close_frame = raw.xs("Close", axis=1, level=level, drop_level=True)
                break
        if close_frame is None:
            raise ValueError("Yahoo Finance response did not contain adjusted close prices.")
    elif "Close" in raw.columns:
        close_frame = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise ValueError("Yahoo Finance response did not contain adjusted close prices.")
    if isinstance(close_frame, pd.Series):
        close_frame = close_frame.to_frame(name=tickers[0])
    close_frame.columns = [str(column).upper() for column in close_frame.columns]
    missing = sorted(set(tickers) - set(close_frame.columns))
    if missing:
        raise ValueError(f"Yahoo Finance returned no price series for: {', '.join(missing)}.")
    return validate_market_prices(close_frame.loc[:, list(tickers)])


def known_instruments() -> dict[str, str]:
    """Return a copy of the configurable default instrument universe."""
    return dict(ASSET_UNIVERSE)

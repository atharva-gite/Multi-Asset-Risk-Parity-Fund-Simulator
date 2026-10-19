"""Streamlit entry point for the multi-asset risk-parity simulator."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from risk_simulator.backtesting.engine import run_erc_backtest, run_fixed_weight_backtest
from risk_simulator.config import (
    ASSET_UNIVERSE,
    BENCHMARK_WEIGHTS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_LEVERAGE,
    DEFAULT_REBALANCE_FREQUENCY,
    DEFAULT_TARGET_VOLATILITY,
    DEFAULT_TRANSACTION_COST_BPS,
)
from risk_simulator.data.market_data import fetch_yahoo_prices, generate_synthetic_demo_prices
from risk_simulator.portfolio.optimization import solve_erc_weights
from risk_simulator.portfolio.returns import prices_to_returns
from risk_simulator.portfolio.risk import (
    estimate_covariance,
    portfolio_volatility,
    risk_contributions,
)
from risk_simulator.visualization.charts import cumulative_returns, drawdown_series

st.set_page_config(
    page_title="Multi-Asset Risk Parity Simulator",
    page_icon="⚖️",
    layout="wide",
)

st.title("Multi-Asset Risk Parity Fund Simulator")
st.caption(
    "An open research scaffold for long-only equal-risk-contribution portfolios, "
    "volatility targeting, and a transparent 60/40 comparison."
)
st.info(
    "Research demonstration only. It does not reproduce Bridgewater's proprietary "
    "strategy, and no result implies profitability. Synthetic mode is not historical "
    "market performance. Backtests omit financing, taxes, market impact, and fund fees."
)

today = date.today()
with st.sidebar:
    st.header("Simulation settings")
    mode = st.radio(
        "Data mode",
        ("Synthetic demonstration", "Yahoo Finance"),
        help="Synthetic data is deterministic and clearly labeled. Yahoo Finance requires internet access.",
    )
    selected_dates = st.date_input(
        "Evaluation date range",
        value=(today - timedelta(days=365 * 5), today),
        min_value=date(1990, 1, 1),
        max_value=today,
    )
    tickers = st.multiselect(
        "Asset universe",
        options=list(ASSET_UNIVERSE),
        default=list(ASSET_UNIVERSE),
        format_func=lambda ticker: f"{ticker} — {ASSET_UNIVERSE[ticker]}",
    )
    lookback = st.slider("Covariance lookback (trading days)", 20, 252, DEFAULT_LOOKBACK_DAYS, 1)
    rebalance_frequency = st.selectbox(
        "Rebalancing frequency",
        ("Monthly", "Quarterly"),
        index=0 if DEFAULT_REBALANCE_FREQUENCY == "Monthly" else 1,
    )
    target_volatility = (
        st.slider("Target annualized volatility", 2.0, 25.0, DEFAULT_TARGET_VOLATILITY * 100, 0.5)
        / 100.0
    )
    max_leverage = st.slider("Maximum gross exposure", 0.5, 3.0, DEFAULT_MAX_LEVERAGE, 0.1)
    transaction_cost_bps = st.slider(
        "Transaction cost (basis points per gross traded exposure)",
        0.0,
        20.0,
        DEFAULT_TRANSACTION_COST_BPS,
        1.0,
    )

if mode == "Synthetic demonstration":
    st.warning(
        "SYNTHETIC DEMONSTRATION DATA — all prices and performance curves below are "
        "generated for interface testing and are not historical observations."
    )

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    st.error("Choose both a start date and an end date.")
    st.stop()
if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()
if not tickers:
    st.warning("Select at least one asset to run the simulator.")
    st.stop()

# Fetch a warm-up window before the requested evaluation period. The strategy
# still reports performance only from the user's selected start date onward.
warmup_days = max(lookback * 3, 180)
fetch_start = pd.Timestamp(start_date - timedelta(days=int(warmup_days)))
with st.spinner("Loading prices and calculating the portfolio..."):
    try:
        if mode == "Synthetic demonstration":
            prices = generate_synthetic_demo_prices(fetch_start, end_date, tickers)
            mode_label = "Synthetic demonstration (not historical)"
        else:
            prices = fetch_yahoo_prices(tickers, fetch_start, end_date)
            mode_label = "Yahoo Finance auto-adjusted daily closes"
    except (RuntimeError, ValueError, ImportError) as exc:
        st.error(f"Price retrieval failed: {exc}")
        st.info("Select Synthetic demonstration in the sidebar to continue without network access.")
        st.stop()

returns = prices_to_returns(prices, method="simple")
selected_end = pd.Timestamp(end_date)
evaluation_start = pd.Timestamp(start_date)
returns = returns.loc[returns.index <= selected_end]
complete_returns = returns.dropna(how="any")
evaluation_rows = complete_returns.loc[complete_returns.index >= evaluation_start]

st.subheader("Data status")
status_columns = st.columns(4)
status_columns[0].metric("Selected mode", mode_label)
status_columns[1].metric("Price coverage", f"{prices.index.min().date()} to {prices.index.max().date()}")
status_columns[2].metric("Complete daily returns", f"{len(complete_returns):,}")
status_columns[3].metric("Evaluation observations", f"{len(evaluation_rows):,}")
missing_cells = int(prices.isna().sum().sum())
if missing_cells:
    st.caption(
        f"{missing_cells} missing price cells were retained; synchronized portfolio calculations "
        "exclude return rows that are incomplete. No prices were forward-filled."
    )
else:
    st.caption("No missing price cells were found in the loaded table.")

window = complete_returns.loc[complete_returns.index <= selected_end].tail(lookback)
if len(window) < lookback:
    st.error(
        f"Need {lookback} complete daily return observations for the selected lookback; "
        f"found {len(window)}. Expand the date range or reduce the lookback."
    )
    st.stop()
try:
    covariance = estimate_covariance(window, min_observations=lookback)
    optimization = solve_erc_weights(covariance)
except ValueError as exc:
    st.error(f"Portfolio risk could not be estimated: {exc}")
    st.stop()
if not optimization.success or optimization.weights is None:
    st.error(f"ERC optimization failed: {optimization.message}")
    st.info("No risk-parity weights are shown for a failed optimization.")
    st.stop()

base_weights = optimization.weights
base_volatility = portfolio_volatility(base_weights, covariance)
current_leverage = min(target_volatility / base_volatility, max_leverage)
target_exposures = base_weights * current_leverage
targeted_volatility = portfolio_volatility(target_exposures, covariance)
contributions = risk_contributions(target_exposures, covariance)

st.subheader("Current allocation and risk")
metric_columns = st.columns(4)
metric_columns[0].metric("ERC sleeve volatility", f"{base_volatility:.1%}")
metric_columns[1].metric("Targeted ex-ante volatility", f"{targeted_volatility:.1%}")
metric_columns[2].metric("Gross exposure / leverage", f"{current_leverage:.2f}×")
metric_columns[3].metric("Annualization", "252 trading days")
left, right = st.columns(2)
with left:
    st.markdown("**Target exposures**")
    exposure_table = pd.DataFrame(
        {"weight": base_weights, "target exposure": target_exposures},
        index=covariance.columns,
    )
    st.dataframe(exposure_table.style.format("{:.1%}"))
    st.bar_chart(exposure_table[["target exposure"]])
with right:
    st.markdown("**Risk contributions**")
    contribution_table = contributions[
        ["risk_contribution", "risk_contribution_pct"]
    ].copy()
    st.dataframe(
        contribution_table.style.format(
            {"risk_contribution": "{:.2%}", "risk_contribution_pct": "{:.1%}"}
        )
    )
    st.bar_chart(contribution_table[["risk_contribution_pct"]])
    st.caption(
        f"Contribution amounts sum to targeted ex-ante portfolio volatility "
        f"({contributions['risk_contribution'].sum():.2%})."
    )

st.subheader("Backtest")
st.caption(
    f"ERC rebalances {rebalance_frequency.lower()} using the preceding {lookback} "
    "complete daily return observations. The rebalance-day return is excluded from "
    "the estimate. Volatility targeting scales exposures up or down and caps gross "
    "exposure at the selected maximum. Costs use gross traded exposure."
)
try:
    strategy = run_erc_backtest(
        returns,
        lookback=lookback,
        rebalance_frequency=rebalance_frequency,
        target_annual_volatility=target_volatility,
        max_leverage=max_leverage,
        transaction_cost_bps=transaction_cost_bps,
        evaluation_start=evaluation_start,
    )
except (RuntimeError, ValueError) as exc:
    st.error(f"Backtest unavailable: {exc}")
    strategy = None

if strategy is not None:
    if all(ticker in returns.columns for ticker in BENCHMARK_WEIGHTS):
        try:
            benchmark = run_fixed_weight_backtest(
                returns.loc[:, list(BENCHMARK_WEIGHTS)],
                BENCHMARK_WEIGHTS,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=transaction_cost_bps,
                evaluation_start=evaluation_start,
            )
            benchmark_returns = benchmark.results["net_return"].reindex(strategy.results.index).dropna()
            if benchmark_returns.empty:
                raise ValueError("No overlapping 60/40 benchmark observations are available.")
            strategy_returns = strategy.results["net_return"].reindex(benchmark_returns.index)
            curves = cumulative_returns(
                {"ERC strategy": strategy_returns, "60/40 benchmark": benchmark_returns}
            )
            drawdowns = pd.DataFrame(
                {
                    "ERC strategy": drawdown_series(strategy_returns),
                    "60/40 benchmark": drawdown_series(benchmark_returns),
                }
            )
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**Cumulative net returns**")
                st.line_chart(curves, y_label="Cumulative return")
            with chart_right:
                st.markdown("**Drawdowns**")
                st.line_chart(drawdowns, y_label="Drawdown")
            realized_vol = (
                float(strategy_returns.std(ddof=1) * (252.0**0.5))
                if len(strategy_returns) > 1
                else float("nan")
            )
            final_metrics = st.columns(4)
            final_metrics[0].metric(
                "ERC cumulative net return",
                f"{curves['ERC strategy'].iloc[-1]:.1%}",
            )
            final_metrics[1].metric(
                "60/40 cumulative net return",
                f"{curves['60/40 benchmark'].iloc[-1]:.1%}",
            )
            final_metrics[2].metric(
                "ERC maximum drawdown",
                f"{drawdowns['ERC strategy'].min():.1%}",
            )
            final_metrics[3].metric("ERC realized annualized volatility", f"{realized_vol:.1%}")
            st.caption(
                "The benchmark is a monthly (or quarterly, as selected) rebalanced 60% SPY / "
                "40% IEF portfolio. Both strategies use the same stated gross-traded-exposure "
                "transaction-cost rate. Financing costs are excluded."
            )
        except (RuntimeError, ValueError) as exc:
            st.warning(f"60/40 comparison unavailable: {exc}")
    else:
        st.warning("60/40 comparison unavailable: select both SPY and IEF in the asset universe.")

st.caption(
    "Yahoo Finance availability and adjusted-price history can change. This prototype "
    "does not include FRED data, taxes, financing costs, liquidity constraints, or "
    "production-grade execution assumptions."
)

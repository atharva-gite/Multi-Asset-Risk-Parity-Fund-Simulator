"""Streamlit entry point for the multi-asset risk-parity simulator."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent


def _local_yahoo_mode_enabled() -> bool:
    """Read the local-only Yahoo opt-in without consulting hosted environment settings."""
    try:
        lines = (ROOT / ".env").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        setting, separator, value = line.partition("=")
        if separator and setting.strip() == "ENABLE_LOCAL_YAHOO_MODE":
            return value.strip().strip("\"'").lower() in {"1", "true", "yes"}
    return False


LOCAL_YAHOO_MODE_ENABLED = _local_yahoo_mode_enabled()
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
from risk_simulator.data.market_data import (
    fetch_yahoo_prices,
    generate_synthetic_demo_prices,
    load_uploaded_prices_csv,
    validate_market_prices,
)
from risk_simulator.portfolio.optimization import solve_erc_weights
from risk_simulator.portfolio.returns import prices_to_returns
from risk_simulator.portfolio.risk import (
    estimate_covariance,
    portfolio_volatility,
    risk_contributions,
)
from risk_simulator.reporting import build_run_manifest, manifest_json
from risk_simulator.visualization.charts import cumulative_returns, drawdown_series

st.set_page_config(
    page_title="Multi-Asset Risk Parity Simulator",
    page_icon="⚖️",
    layout="wide",
)

st.title("Multi-Asset Risk Parity Fund Simulator")
st.caption(
    "An open research scaffold for long-only equal-risk-contribution portfolios, "
    "volatility targeting, and a transparent 60/40 comparison using daily data."
)
st.info(
    "Research demonstration only. It does not reproduce Bridgewater's proprietary "
    "strategy, and no result implies profitability. Synthetic mode is not historical "
    "market performance. Backtests omit financing, taxes, market impact, and fund fees."
)

today = date.today()
uploaded_prices = None
with st.sidebar:
    st.header("Simulation settings")
    data_modes = ["Synthetic demonstration", "User-uploaded licensed CSV"]
    if LOCAL_YAHOO_MODE_ENABLED:
        data_modes.append("Yahoo Finance (local research only)")
    mode = st.radio(
        "Data mode",
        data_modes,
        help=(
            "Synthetic data works offline. Upload a licensed daily-price CSV, or enable "
            "Yahoo mode only in a private local research environment."
        ),
    )
    if mode == "User-uploaded licensed CSV":
        csv_template = pd.DataFrame(columns=["Date", *ASSET_UNIVERSE]).to_csv(index=False)
        st.download_button(
            "Download CSV template",
            data=csv_template,
            file_name="daily-prices-template.csv",
            mime="text/csv",
            key="download_prices_template",
        )
        uploaded_file = st.file_uploader(
            "Daily prices CSV",
            type=["csv"],
            help=(
                "Use a Date column followed by one adjusted-close price column per ticker. "
                "Include enough pre-evaluation history for the covariance lookback. "
                "The app does not adjust uploaded prices or write the file to disk."
            ),
        )
        if uploaded_file is not None:
            try:
                uploaded_prices = load_uploaded_prices_csv(uploaded_file)
            except ValueError as exc:
                st.error(f"Could not load uploaded prices: {exc}")
        else:
            st.caption("Upload a CSV with daily prices to select its instruments.")
    selected_dates = st.date_input(
        "Evaluation date range",
        value=(today - timedelta(days=365 * 5), today),
        min_value=date(1990, 1, 1),
        max_value=today,
    )
    ticker_options = (
        list(uploaded_prices.columns)
        if mode == "User-uploaded licensed CSV" and uploaded_prices is not None
        else list(ASSET_UNIVERSE)
        if mode != "User-uploaded licensed CSV"
        else []
    )
    tickers = st.multiselect(
        "Asset universe",
        options=ticker_options,
        default=ticker_options,
        format_func=lambda ticker: (
            f"{ticker} — {ASSET_UNIVERSE.get(ticker, 'Uploaded instrument')}"
        ),
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
elif mode == "Yahoo Finance (local research only)":
    st.warning(
        "LOCAL RESEARCH MODE — Yahoo Finance data is enabled by a local .env setting. "
        "This mode is hidden by default and must not be enabled on the public deployment."
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
    if mode == "User-uploaded licensed CSV" and uploaded_prices is None:
        st.info("Upload a daily prices CSV in the sidebar to continue.")
        st.stop()
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
        elif mode == "Yahoo Finance (local research only)":
            prices = fetch_yahoo_prices(tickers, fetch_start, end_date)
            mode_label = "Yahoo Finance auto-adjusted daily closes (local research)"
        else:
            assert uploaded_prices is not None
            selected_prices = uploaded_prices.loc[
                uploaded_prices.index <= pd.Timestamp(end_date), list(tickers)
            ]
            prices = validate_market_prices(selected_prices)
            mode_label = "User-uploaded licensed daily prices"
    except (RuntimeError, ValueError, ImportError) as exc:
        st.error(f"Price retrieval failed: {exc}")
        st.info(
            "Select Synthetic demonstration or upload licensed CSV prices to continue "
            "without a network data source."
        )
        st.stop()

returns = prices_to_returns(prices, method="simple")
selected_end = pd.Timestamp(end_date)
evaluation_start = pd.Timestamp(start_date)
returns = returns.loc[returns.index <= selected_end]
complete_returns = returns.dropna(how="any")
evaluation_rows = complete_returns.loc[complete_returns.index >= evaluation_start]

st.subheader("Data status")
status_columns = st.columns(5)
status_columns[0].metric("Selected mode", mode_label)
status_columns[1].metric("Latest price observation", prices.index.max().date().isoformat())
status_columns[2].metric(
    "Price history", f"{prices.index.min().date()} to {prices.index.max().date()}"
)
status_columns[3].metric("Complete daily returns", f"{len(complete_returns):,}")
status_columns[4].metric("Evaluation observations", f"{len(evaluation_rows):,}")
st.caption(
    f"Requested evaluation period: {start_date} to {end_date}. Available prices run through "
    f"{prices.index.max().date()}; results use observed data only."
)
missing_cells = int(prices.isna().sum().sum())
if missing_cells:
    st.caption(
        f"{missing_cells} missing price cells were retained; synchronized portfolio calculations "
        "exclude return rows that are incomplete. No prices were forward-filled."
    )
else:
    st.caption("No missing price cells were found in the loaded table.")
if mode == "User-uploaded licensed CSV":
    st.caption(
        "Uploaded price files are processed for this app session and are not written to disk. "
        "The user who uploads the data is responsible for having display and use rights."
    )

window = complete_returns.loc[complete_returns.index <= selected_end].tail(lookback)
if len(window) < lookback:
    st.error(
        f"Need {lookback} complete daily return observations for the selected lookback; "
        f"found {len(window)}. Expand the date range or reduce the lookback."
    )
    st.stop()
st.caption(
    f"Latest-data allocation snapshot: covariance uses the most recent {len(window)} "
    f"complete return observations through {window.index.max().date()}."
)
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

st.subheader("Latest-data allocation and risk")
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
    contribution_table = contributions[["risk_contribution", "risk_contribution_pct"]].copy()
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
    benchmark_export_returns = None
    if all(ticker in returns.columns for ticker in BENCHMARK_WEIGHTS):
        try:
            benchmark = run_fixed_weight_backtest(
                returns.loc[:, list(BENCHMARK_WEIGHTS)],
                BENCHMARK_WEIGHTS,
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=transaction_cost_bps,
                evaluation_start=evaluation_start,
            )
            benchmark_returns = (
                benchmark.results["net_return"].reindex(strategy.results.index).dropna()
            )
            if benchmark_returns.empty:
                raise ValueError("No overlapping 60/40 benchmark observations are available.")
            benchmark_export_returns = benchmark.results["net_return"].rename(
                "benchmark_60_40_net_return"
            )
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

    export_frame = strategy.results.add_prefix("erc_").join(
        strategy.weights.add_prefix("erc_exposure_")
    )
    if benchmark_export_returns is not None:
        export_frame = export_frame.join(
            benchmark_export_returns.reindex(export_frame.index), how="left"
        )
    export_frame.index.name = "date"
    csv_data = export_frame.to_csv(
        date_format="%Y-%m-%d",
        float_format="%.10g",
        lineterminator="\n",
    )
    manifest = build_run_manifest(
        prices,
        data_source=mode_label,
        evaluation_start=evaluation_start,
        evaluation_end=selected_end,
        tickers=tickers,
        lookback=lookback,
        rebalance_frequency=rebalance_frequency,
        target_annual_volatility=target_volatility,
        max_leverage=max_leverage,
        transaction_cost_bps=transaction_cost_bps,
        benchmark_weights=BENCHMARK_WEIGHTS if benchmark_export_returns is not None else None,
    )
    with st.expander("Export run artifacts"):
        st.caption(
            "The manifest records the price-table fingerprint and selected assumptions. "
            "The CSV contains daily ERC returns, exposures, turnover, costs, and the "
            "60/40 net return when that comparison is available."
        )
        download_columns = st.columns(2)
        download_columns[0].download_button(
            "Download backtest CSV",
            data=csv_data,
            file_name="risk-parity-backtest.csv",
            mime="text/csv",
            key="download_backtest_csv",
        )
        download_columns[1].download_button(
            "Download run manifest",
            data=manifest_json(manifest),
            file_name="risk-parity-run-manifest.json",
            mime="application/json",
            key="download_run_manifest",
        )

st.caption(
    "This is a daily-frequency research application with no live quotes or order routing. "
    "Uploaded prices are used as provided. The simulator does not include financing costs, "
    "taxes, liquidity constraints, or production-grade execution assumptions."
)

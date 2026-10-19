"""Central defaults for the simulator."""

ASSET_UNIVERSE: dict[str, str] = {
    "SPY": "US equities",
    "VEA": "Developed ex-US equities",
    "IEF": "Intermediate US Treasuries",
    "TLT": "Long US Treasuries",
    "DBC": "Broad commodities",
    "TIP": "US inflation-protected Treasuries",
}

BENCHMARK_WEIGHTS: dict[str, float] = {"SPY": 0.60, "IEF": 0.40}
TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 63
DEFAULT_REBALANCE_FREQUENCY = "Monthly"
DEFAULT_TARGET_VOLATILITY = 0.10
DEFAULT_TRANSACTION_COST_BPS = 5.0
DEFAULT_MAX_LEVERAGE = 2.0

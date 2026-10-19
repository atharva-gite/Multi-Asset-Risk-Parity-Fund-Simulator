# Multi-Asset Risk Parity Fund Simulator

A small Streamlit research application for exploring a long-only equal-risk-contribution (ERC) portfolio, volatility targeting, monthly or quarterly rebalancing, and a fixed 60/40 comparison. The code separates financial calculations from the user interface so the calculation modules can be tested directly.

This is a learning and software scaffold. It does not reproduce Bridgewater's proprietary strategy and does not make claims about future returns or profitability. Synthetic mode generates deterministic data for interface and calculation demonstrations; it is not historical market data.

## Requirements

- Python 3.10 or newer
- Internet access only when using Yahoo Finance mode or installing packages
- GitHub account and a Streamlit Community Cloud account for hosted deployment

Yahoo Finance access is public and does not require an API key. The prototype does not call FRED and requires no credentials or secrets.

## Local setup

Run these commands from the repository root.

Create a virtual environment:

~~~bash
python3 -m venv .venv
~~~

Activate it on macOS or Linux:

~~~bash
source .venv/bin/activate
~~~

On Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

Install runtime dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Launch the application:

~~~bash
streamlit run app.py
~~~

Choose **Synthetic demonstration** to run without external market-data access. Switch to **Yahoo Finance** to retrieve auto-adjusted daily prices. Yahoo Finance symbols and available history are controlled by that service and can change.

## Dashboard behavior

- Choose the evaluation period, assets, lookback, rebalance frequency, target annualized volatility, maximum gross exposure, and transaction-cost assumption in the sidebar.
- The app fetches a warm-up period before the selected evaluation start. Backtest results are shown only from the requested start date.
- Incomplete synchronized return rows are excluded; missing prices are never forward-filled.
- A Yahoo Finance error is displayed with a path back to synthetic mode.
- The 60/40 comparison uses 60% SPY and 40% IEF, rebalanced at the same selected calendar frequency. Both assets must be selected.
- The default instrument universe is centralized in **src/risk_simulator/config.py**.

## Calculation conventions

- Returns are simple close-to-close returns, P[t] / P[t-1] - 1. The library also exposes log returns.
- Covariance uses complete daily return rows and the sample covariance estimator, annualized by 252 trading days.
- ERC weights are non-negative and sum to one. A covariance matrix must be positive definite for ERC optimization. Failed or insufficient optimization is surfaced; no equal-weight fallback is called risk parity.
- Risk contributions use Euler volatility contributions, w[i] * (covariance @ w)[i] / portfolio_volatility, and sum to portfolio volatility within numerical tolerance.
- A rolling rebalance uses only the preceding lookback observations. The return on the rebalance date is excluded from that estimate.
- The strategy's target volatility scales its ERC sleeve up or down, subject to the gross-exposure cap.
- Rebalancing happens on the first available observation of each calendar month or quarter. Transaction cost is the configured basis-point rate times gross traded exposure (the L1 change in exposures), charged at rebalances.
- The preset 60/40 benchmark uses the same schedule and cost rate. Financing, cash yield, borrow fees, taxes, liquidity, market impact, and fund fees are not modeled. Results are illustrative historical calculations, not investment advice.

## Tests and checks

Run the unit tests with the Python standard library:

~~~bash
python -m unittest discover -s tests -v
~~~

GitHub Actions runs the same tests and compilation check on pushes and pull requests. The app smoke test uses Streamlit's test harness in synthetic mode and does not fetch external data.

Compile the application and package:

~~~bash
python -m compileall -q app.py src tests
~~~

## Streamlit Community Cloud deployment

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select that repository and branch.
3. Set the main file path to app.py.
4. Deploy. Community Cloud installs the dependencies from requirements.txt.
5. Select Synthetic demonstration to confirm the app starts without credentials; choose Yahoo Finance when internet market data is desired.

No secrets configuration is required. Do not commit .streamlit/secrets.toml or environment files. FRED integration is not implemented; if it is added later, it should remain optional and read any API key from Streamlit secrets or environment configuration.

## Known limitations and troubleshooting

- Yahoo Finance is a third-party public source without a service-level guarantee. If retrieval fails or a ticker has incomplete history, the app reports the issue; use synthetic mode to explore the interface.
- The initial universe consists of representative ETFs, not a complete global asset universe. Ticker availability and history depend on Yahoo Finance.
- ERC uses a historical sample covariance estimate and can be unstable in short or highly collinear samples. The app reports failed optimization rather than presenting fallback weights.
- Volatility targeting is a simple ex-ante scaling rule. Backtests ignore financing costs and execution effects and may differ from live implementation.
- If the selected period has too few complete observations, expand the dates or reduce the lookback.
- If imports fail, activate the virtual environment and install requirements.txt again. The app requires Python 3.10 or newer.

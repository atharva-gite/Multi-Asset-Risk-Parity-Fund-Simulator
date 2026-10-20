# Multi-Asset Risk Parity Fund Simulator

A small Streamlit research application for exploring a long-only equal-risk-contribution (ERC) portfolio, volatility targeting, monthly or quarterly rebalancing, and a fixed 60/40 comparison. The code separates financial calculations from the user interface so the calculation modules can be tested directly.

This is a learning and software scaffold. It does not reproduce Bridgewater's proprietary strategy and does not make claims about future returns or profitability. Synthetic mode generates deterministic data for interface and calculation demonstrations; it is not historical market data.

## Requirements

- Python 3.10 or newer
- Internet access when installing packages or using local Yahoo Finance research mode
- GitHub account and a Streamlit Community Cloud account for hosted deployment

The public app defaults to deterministic synthetic data and also accepts a visitor's own licensed daily-price CSV. Yahoo Finance mode is opt-in for private local research and requires no API key. No credentials are required for the public app.

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

Copy the environment example for local development:

~~~bash
cp .env.example .env
~~~

Set `ENABLE_LOCAL_YAHOO_MODE=true` in `.env` only when running a private local research session. Yahoo mode is disabled by default and should stay disabled on the public deployment. The optional adapter can be installed with `python -m pip install -e ".[research]"`.

For development with local Yahoo research support, install the package and the free lint/format tool:

~~~bash
python -m pip install -e ".[dev,research]"
~~~

Launch the application:

~~~bash
streamlit run app.py
~~~

The app opens in **Synthetic demonstration** mode and works offline. For a real daily price history, choose **User-uploaded licensed CSV** and provide a CSV with a date column followed by one adjusted-close column per instrument, for example:

~~~csv
Date,SPY,IEF
YYYY-MM-DD,<adjusted close>,<adjusted close>
~~~

Include enough history before the evaluation start to cover the covariance lookback. Uploaded prices are used as provided; the app does not adjust prices or write the upload to disk. Users must have the rights to use and display their data. Uploads are limited to 10 MB.
A blank CSV template is available from the upload section in the app sidebar.

When `ENABLE_LOCAL_YAHOO_MODE=true` is set in a local `.env`, the sidebar also offers **Yahoo Finance (local research only)**. That mode retrieves auto-adjusted daily prices through the unofficial `yfinance` package. Keep it disabled in public deployments and review Yahoo's terms for permitted use.

## Dashboard behavior

- Choose the evaluation period, assets, lookback, rebalance frequency, target annualized volatility, maximum gross exposure, and transaction-cost assumption in the sidebar.
- The app fetches a warm-up period before the selected evaluation start. Backtest results are shown only from the requested start date.
- Incomplete synchronized return rows are excluded; missing prices are never forward-filled.
- The latest available price-observation date is shown for every source so users can assess data staleness.
- Uploaded CSV instruments are read from the file header. A simple CSV with one date column and one daily price series per instrument is supported; multi-row headers are not.
- Yahoo Finance is hidden unless explicitly enabled in a local `.env` file.
- The 60/40 comparison uses 60% SPY and 40% IEF, rebalanced at the same selected calendar frequency. Both assets must be selected.
- Each completed run can export a daily CSV with strategy returns, exposures, leverage, turnover, costs, and available benchmark returns, plus a JSON manifest with the selected assumptions, dependency versions, and SHA-256 fingerprint of the loaded price table.
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
The workflow also checks lint and formatting with Ruff.

Compile the application and package:

~~~bash
python -m compileall -q app.py src tests
~~~

## Streamlit Community Cloud deployment

1. Push this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app and select that repository and branch.
3. Set the main file path to app.py.
4. Deploy. Community Cloud installs the dependencies from requirements.txt.
5. The app starts in Synthetic demonstration mode without credentials. Visitors can upload data they are licensed to use. Yahoo Finance mode remains hidden unless explicitly enabled in a local environment.

No secrets configuration is required. Do not commit `.env` or `.streamlit/secrets.toml`. FRED integration is not implemented.

## Known limitations and troubleshooting

- Yahoo Finance is an unofficial local research source with no service-level guarantee. Its availability and history can change.
- The default universe consists of representative ETFs, not a complete global asset universe. Uploaded files may define a different set of instruments.
- Uploaded data is used without adjustment or persistence. It is the uploader's responsibility to supply consistent, licensed adjusted-close prices.
- ERC uses a historical sample covariance estimate and can be unstable in short or highly collinear samples. The app reports failed optimization rather than presenting fallback weights.
- Volatility targeting is a simple ex-ante scaling rule. Backtests ignore financing costs and execution effects and may differ from live implementation.
- If the selected period has too few complete observations, expand the dates or reduce the lookback.
- If imports fail, activate the virtual environment and install requirements.txt again. The app requires Python 3.10 or newer.

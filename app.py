"""Indian equity portfolio optimisation dashboard.

This is an educational analytics application, not investment advice.
"""

from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
import shutil
import tempfile

import certifi
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

# curl_cffi (used by yfinance) can fail on Windows when its certificate path
# contains non-ASCII characters. Copy it to the ASCII-only Windows temp path.
CA_BUNDLE = Path(tempfile.gettempdir()) / "yfinance-ca-bundle.pem"
if not CA_BUNDLE.exists():
    shutil.copyfile(certifi.where(), CA_BUNDLE)
os.environ["CURL_CA_BUNDLE"] = str(CA_BUNDLE)
os.environ["REQUESTS_CA_BUNDLE"] = str(CA_BUNDLE)

import yfinance as yf


st.set_page_config(page_title="India Portfolio Optimizer", page_icon="🇮🇳", layout="wide")

# Ten sector groups, each with ten liquid NSE symbols. Yahoo Finance uses .NS.
SECTORS: dict[str, list[str]] = {
    "Banking": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB"],
    "Information Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "PERSISTENT", "COFORGE", "OFSS"],
    "Energy & Utilities": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "COALINDIA", "GAIL", "BPCL", "IOC", "HINDPETRO", "TATAPOWER"],
    "Pharmaceuticals": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM", "GLENMARK"],
    "Automobiles": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR", "HEROMOTOCO", "ASHOKLEY", "MOTHERSON", "BOSCHLTD"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "GODREJCP", "MARICO", "COLPAL", "EMAMILTD"],
    "Financial Services": ["BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIGI", "ICICIPRULI", "HDFCAMC", "MUTHOOTFIN", "CHOLAFIN", "SHRIRAMFIN"],
    "Metals & Mining": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "NMDC", "SAIL", "NALCO", "HINDZINC", "APLAPOLLO"],
    "Telecom & Media": ["BHARTIARTL", "IDEA", "INDUSTOWER", "ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "DISHTV", "NAZARA", "TV18BRDCST"],
    "Consumer & Retail": ["TITAN", "TRENT", "DMART", "PAGEIND", "ABFRL", "SHOPERSTOP", "KALYANKJIL", "METROBRAND", "VBL", "WESTLIFE"],
}

TICKER_TO_SECTOR = {ticker: sector for sector, tickers in SECTORS.items() for ticker in tickers}
DEFAULT_TICKERS = [tickers[0] for tickers in SECTORS.values()]


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""
    # Bump this whenever the download environment changes, so stale failed
    # responses are never reused by Streamlit's persistent cache.
    cache_version = "nse-download-ca-bundle-v2"
    symbols = [f"{ticker}.NS" for ticker in tickers]
    raw = yf.download(symbols, start=start, end=end + timedelta(days=1), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices.columns = [str(column).replace(".NS", "") for column in prices.columns]
    return prices.dropna(axis=1, how="all").ffill().dropna()


def portfolio_statistics(weights: np.ndarray, mean_returns: pd.Series, covariance: pd.DataFrame, risk_free_rate: float) -> tuple[float, float, float]:
    annual_return = float(np.dot(weights, mean_returns) * 252)
    annual_volatility = float(np.sqrt(weights @ (covariance * 252) @ weights))
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else 0.0
    return annual_return, annual_volatility, sharpe


def optimise(mean_returns: pd.Series, covariance: pd.DataFrame, risk_free_rate: float, max_weight: float, objective: str) -> np.ndarray:
    count = len(mean_returns)
    initial = np.repeat(1 / count, count)

    def negative_sharpe(weights: np.ndarray) -> float:
        return -portfolio_statistics(weights, mean_returns, covariance, risk_free_rate)[2]

    def volatility(weights: np.ndarray) -> float:
        return portfolio_statistics(weights, mean_returns, covariance, risk_free_rate)[1]

    outcome = minimize(
        negative_sharpe if objective == "Maximum Sharpe ratio" else volatility,
        initial,
        method="SLSQP",
        bounds=[(0, max_weight)] * count,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        options={"maxiter": 1000},
    )
    if not outcome.success:
        raise ValueError(outcome.message)
    return outcome.x


def efficient_frontier(mean_returns: pd.Series, covariance: pd.DataFrame, max_weight: float, points: int = 30) -> pd.DataFrame:
    count = len(mean_returns)
    annual_means = mean_returns * 252
    targets = np.linspace(float(annual_means.min()), float(annual_means.max()), points)
    records: list[dict[str, float]] = []
    for target in targets:
        result = minimize(
            lambda weights: np.sqrt(weights @ (covariance * 252) @ weights),
            np.repeat(1 / count, count),
            method="SLSQP",
            bounds=[(0, max_weight)] * count,
            constraints=(
                {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
                {"type": "eq", "fun": lambda weights, t=target: np.dot(weights, annual_means) - t},
            ),
        )
        if result.success:
            records.append({"Return": target, "Volatility": result.fun})
    return pd.DataFrame(records)


st.title("🇮🇳 Indian Market Portfolio Optimizer")
st.caption("NSE equity universe • 10 sector groups • 10 companies per sector • educational use only")

with st.sidebar:
    st.header("Portfolio inputs")
    selected_sectors = st.multiselect("Sectors", list(SECTORS), default=list(SECTORS))
    available = [ticker for sector in selected_sectors for ticker in SECTORS[sector]]
    selected_tickers = st.multiselect("NSE stocks", available, default=[ticker for ticker in DEFAULT_TICKERS if ticker in available])
    period_label = st.selectbox("Price history", ["1 year", "3 years", "5 years"], index=1)
    period_days = {"1 year": 365, "3 years": 365 * 3, "5 years": 365 * 5}[period_label]
    objective = st.radio("Optimisation target", ["Maximum Sharpe ratio", "Minimum volatility"])
    risk_free_rate = st.number_input("Annual risk-free rate", min_value=0.0, max_value=0.20, value=0.07, step=0.005, format="%.3f")
    max_weight = st.slider("Maximum stock weight", 0.05, 1.0, 0.25, 0.05)
    run_analysis = st.button("Optimise portfolio", type="primary", use_container_width=True)

st.info("Start with the 10 default large-cap names, one per sector. You may choose up to 30 stocks for a responsive analysis.")

if run_analysis:
    if len(selected_tickers) < 2:
        st.error("Select at least two stocks.")
        st.stop()
    if len(selected_tickers) > 30:
        st.error("Please select no more than 30 stocks at a time.")
        st.stop()
    if max_weight * len(selected_tickers) < 1:
        st.error("The maximum stock weight is too low for this number of stocks.")
        st.stop()

    end_date = date.today()
    start_date = end_date - timedelta(days=period_days)
    with st.spinner("Downloading NSE price history and solving the portfolio…"):
        prices = download_prices(tuple(selected_tickers), start_date, end_date)
    missing = sorted(set(selected_tickers) - set(prices.columns))
    if missing:
        st.warning("No usable price data for: " + ", ".join(missing))
    if prices.shape[1] < 2 or len(prices) < 60:
        st.error("Not enough overlapping price data to optimise this selection. Try different stocks or a longer history.")
        st.stop()

    returns = prices.pct_change().dropna()
    mean_returns, covariance = returns.mean(), returns.cov()
    try:
        weights = optimise(mean_returns, covariance, risk_free_rate, max_weight, objective)
    except ValueError as error:
        st.error(f"Optimisation could not converge: {error}")
        st.stop()

    allocation = pd.DataFrame({"Ticker": mean_returns.index, "Weight": weights})
    allocation["Sector"] = allocation["Ticker"].map(TICKER_TO_SECTOR)
    allocation["Weight"] = allocation["Weight"].clip(lower=0)
    allocation = allocation.sort_values("Weight", ascending=False)
    annual_return, annual_volatility, sharpe = portfolio_statistics(weights, mean_returns, covariance, risk_free_rate)

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Expected annual return", f"{annual_return:.1%}")
    metric_2.metric("Expected annual volatility", f"{annual_volatility:.1%}")
    metric_3.metric("Sharpe ratio", f"{sharpe:.2f}")
    metric_4.metric("Stocks with allocation", str((allocation["Weight"] > 0.001).sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Recommended allocation")
        chart = px.bar(allocation.query("Weight > 0.001"), x="Ticker", y="Weight", color="Sector", text_auto=".1%")
        chart.update_layout(yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Sector allocation")
        sector_weights = allocation.groupby("Sector", as_index=False)["Weight"].sum()
        st.plotly_chart(px.pie(sector_weights, names="Sector", values="Weight", hole=0.45), use_container_width=True)

    st.subheader("Efficient frontier")
    frontier = efficient_frontier(mean_returns, covariance, max_weight)
    frontier_chart = go.Figure()
    frontier_chart.add_trace(go.Scatter(x=frontier["Volatility"], y=frontier["Return"], mode="lines", name="Efficient frontier"))
    frontier_chart.add_trace(go.Scatter(x=[annual_volatility], y=[annual_return], mode="markers", marker={"size": 13, "color": "#f97316"}, name=objective))
    frontier_chart.update_layout(xaxis_title="Annual volatility", yaxis_title="Expected annual return", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(frontier_chart, use_container_width=True)

    st.subheader("Historical portfolio comparison")
    optimised_daily = returns @ weights
    equal_daily = returns.mean(axis=1)
    performance = pd.DataFrame({"Optimised": (1 + optimised_daily).cumprod(), "Equal-weighted": (1 + equal_daily).cumprod()})
    performance.index.name = "Date"
    st.plotly_chart(px.line(performance, labels={"value": "Growth of ₹1", "Date": "Date", "variable": "Strategy"}), use_container_width=True)

    st.subheader("Allocation table")
    display = allocation.assign(Weight=allocation["Weight"].map("{:.2%}".format))
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download allocation CSV", allocation.to_csv(index=False).encode("utf-8"), "indian_portfolio_allocation.csv", "text/csv")

st.divider()
st.caption("Data are supplied by Yahoo Finance. Historical performance does not predict future results; this dashboard is not investment advice.")

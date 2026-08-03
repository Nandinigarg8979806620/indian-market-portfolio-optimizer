# Indian Market Portfolio Optimizer

An interactive Streamlit dashboard for educational Indian equity portfolio analysis. It provides an NSE stock universe of ten sector groups with ten companies in each group, then uses historical market data to calculate a long-only portfolio allocation.

## Features

- 100-stock Indian NSE universe across Banking, IT, Energy & Utilities, Pharmaceuticals, Automobiles, FMCG, Financial Services, Metals & Mining, Telecom & Media, and Consumer & Retail
- Maximum-Sharpe and minimum-volatility optimisation
- Individual-stock allocation, sector allocation, efficient frontier, and equal-weight comparison
- Configurable history window, risk-free rate, and maximum stock-weight constraint
- CSV export of the recommended allocation

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository named `indian-portfolio-optimizer`.
2. Upload this folder's contents to that repository.
3. In Streamlit Community Cloud, select the repository and choose `app.py` as the entry point.

## Disclaimer

This project is for education and portfolio-analysis demonstration only. It is not investment advice. Historical data and optimisation outputs do not guarantee future performance.

# US Rates Curve & Spread Analytics

US Treasury yield curve modeling, spread analytics, and interest rate shock scenarios.

This project is a full-stack fixed income analytics tool designed to analyze US Treasury yields, term structure dynamics, and curve risk metrics using real market data from the Federal Reserve (FRED).

It combines:

A Python (Flask) backend for data retrieval, curve fitting, and risk calculations

A pure HTML / JavaScript frontend for interactive visualization and analytics

# Key Features
-Yield & Spread Analysis

Fetches US Treasury yields directly from FRED (3M, 2Y, 5Y, 10Y, 30Y)

Computes yield spreads (e.g. 2Y–10Y, 5Y–30Y)

Historical statistics:

Current spread (bps)

Mean & standard deviation

Z-score

Percentile ranking

-Yield Curve Modeling

Fits a Nelson–Siegel–Svensson (NSS) curve to the latest market snapshot

Produces a smooth zero-coupon yield curve from 0.25Y to 30Y

Extracts fitted yields (e.g. model-implied 10Y rate)

-Rate Shock Scenarios

Apply deterministic interest rate shocks:

Parallel shift

Steepener

Flattener

Compare:

Spot curve vs shocked curve

Visual impact across maturities

-Risk Metrics (Proxy)

Modified duration approximation

DV01 proxy (per 100 notional)

Designed for intuitive curve-risk interpretation

# Methodology

Data source: Federal Reserve Economic Data (FRED)

Curve model: Nelson–Siegel–Svensson (OLS calibration with fixed taus)

Spreads: Calculated in basis points (bps)

Duration: Analytical approximation for par bullet bonds

DV01: Duration × Notional × 1bp

This tool is intended for educational, analytical, and prototyping purposes, not for production trading systems.

# Frontend (index.html)

Pure HTML / CSS / JavaScript

Charting with Chart.js

Interactive controls:

Maturity selection (short / long)

Date range

Shock size & shock type

KPI dashboard:

Spread

Z-score

Percentile

Model-implied 10Y yield

DV01 proxy

Responsive dark UI inspired by professional rates dashboards

# Backend (main.py)
Tech Stack

Python

Flask + Flask-CORS

pandas / numpy

pandas-datareader (FRED)

nelson-siegel-svensson

scipy

API Endpoint
POST /api/analyze


Payload

{
  "short": 2,
  "long": 10,
  "start": "2023-01-01",
  "end": "2024-12-01",
  "shock_bp": 10,
  "shock_type": "parallel"
}


Response

Time series (yields & spreads)

Curve data (spot & shocked)

Statistical indicators

Risk metrics

 Getting Started

1- Clone the repository
git clone https://github.com/g79ky6c79c-bot/US-Rates-Curve-Spread-Analytics

2- Install dependencies
pip install flask flask-cors pandas numpy pandas-datareader scipy nelson-siegel-svensson

3- Run the backend
python main.py


The API will run on:

http://127.0.0.1:5000

Open the frontend

Simply open index.html in your browser.

Use Cases

Yield curve interpretation

Spread regime analysis (inversion, steepening)

Macro & monetary policy analysis

Fixed income education & demonstrations

Rates risk intuition (DV01, duration effects)

# Disclaimer

This project is not a trading system and does not provide investment advice.
It is intended for research, learning, and demonstration purposes only.

# Author

Toussaint Yonga
Finance · Fixed Income · Risk · Quantitative Analysis

If you’re a professional in rates, risk, or macro and want to discuss improvements or extensions, feel free to connect.

# GoldForecast

GoldForecast is a system for forecasting the daily direction and % return of gold prices,
using a Hybrid ML approach (Classification + Regression) across multiple horizons
(t+1, t+5, t+10). It generates Buy/Sell/Hold signals and evaluates them through
backtesting, with an automated pipeline powered by Airflow, MLflow experiment tracking,
a web dashboard (FastAPI + React), and full automated testing/CI.

A CPE232 course project — extending the original proposal into a portfolio-grade system
with a full data pipeline, dashboard, and testing, rather than just an analysis notebook.

---

## Overview

- **Data**: Gold prices (OHLCV) + macroeconomic indicators (DXY, Crude Oil, VIX,
  US 10Y Yield, S&P 500), 10 years of history from Yahoo Finance
- **Models**: XGBoost/LightGBM (regression + classification) with a Logistic Regression
  baseline, plus LSTM as an extra baseline
- **Multi-horizon**: forecasts t+1, t+5, and t+10 trading days ahead, to see how far the
  model's edge actually holds
- **Signal**: a rule-based threshold converts model output → BUY / SELL / HOLD with a
  confidence score
- **Backtest**: simulates the strategy on test data including transaction costs,
  compared against buy-and-hold
- **Pipeline**: Apache Airflow fetches data/predicts daily, retrains models weekly with
  a quality gate before promotion
- **Dashboard**: FastAPI + React showing the latest signal, price chart, and backtest
  results
- **Testing/CI**: unit tests, data validation tests, model quality gate, GitHub Actions

## Architecture

```
Yahoo Finance ──┐
                 ▼
        ┌─────────────────┐        ┌──────────┐
        │  Airflow DAGs    │───────▶│  MLflow  │  (experiment tracking)
        │  (daily/retrain) │        └──────────┘
        └────────┬─────────┘
                  ▼
           ┌─────────────┐
           │  PostgreSQL  │  (market data, predictions, signals, backtest)
           └──────┬──────┘
                  ▼
           ┌─────────────┐        ┌──────────────┐
           │   FastAPI    │───────▶│ React Dashboard │
           └─────────────┘        └──────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Python, pandas, XGBoost/LightGBM, scikit-learn, MLflow |
| Orchestration | Apache Airflow, Docker Compose |
| Database | PostgreSQL |
| Backend | FastAPI |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| CI/CD | GitHub Actions |
| Deployment | Vercel/Netlify (frontend), Render/Fly.io (backend), Managed Postgres |

## Project Structure

```
GoldForcast/
├── ml/               # data pipeline, feature engineering, models, signals, backtest
├── airflow/          # Airflow DAGs
├── backend/          # FastAPI service
├── frontend/         # React dashboard
├── notebooks/        # EDA
├── docs/             # project documentation (PROJECT.md, ADR)
├── plan.md           # roadmap broken down by Issue/Branch
└── docker-compose.yml
```

## Getting Started

> The project is under active development; the commands below describe the intended
> workflow (see [`plan.md`](plan.md) for actual progress)

```bash
# install the ml package's dependencies
cd ml && pip install -e .

# run all services locally (Postgres, Airflow, MLflow, FastAPI)
docker compose up -d

# run the frontend dev server
cd frontend && npm install && npm run dev

# run tests
pytest ml/tests backend/tests
```

## Roadmap

Work is split into 8 Issues/Branches by milestone (setup → data pipeline → modeling →
signal/backtest → db & orchestration → backend → frontend → CI/CD & deploy).
See full details in [`plan.md`](plan.md)

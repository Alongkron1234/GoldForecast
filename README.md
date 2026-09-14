# GoldForcast

GoldForcast คือระบบพยากรณ์ทิศทางและ % ผลตอบแทนราคาทองคำรายวัน ด้วย Hybrid ML
(Classification + Regression) ครอบคลุมหลาย horizon (t+1, t+5, t+10) สร้างสัญญาณ
Buy/Sell/Hold ประเมินผลด้วย Backtesting มี pipeline อัตโนมัติผ่าน Airflow, MLflow
tracking, เว็บ dashboard (FastAPI + React) พร้อม automated testing/CI

โครงงานวิชา CPE232 — ต่อยอดจาก proposal เดิมให้เป็นระบบระดับ portfolio ที่มี data
pipeline, dashboard และ testing ครบวงจร ไม่ใช่แค่ notebook วิเคราะห์ข้อมูล

---

## ภาพรวม (Overview)

- **ข้อมูล**: ราคาทองคำ (OHLCV) + ดัชนีเศรษฐกิจมหภาค (DXY, Crude Oil, VIX, US 10Y Yield,
  S&P 500) ย้อนหลัง 10 ปี จาก Yahoo Finance
- **โมเดล**: XGBoost/LightGBM (regression + classification) พร้อม Logistic Regression
  baseline, LSTM เป็น baseline เสริม
- **Multi-horizon**: พยากรณ์ล่วงหน้า t+1, t+5, t+10 วันทำการ เพื่อดูว่าโมเดลมี edge
  จริงถึงระยะไหน
- **Signal**: กฎ rule-based threshold แปลงผลโมเดล → BUY / SELL / HOLD พร้อม
  confidence score
- **Backtest**: จำลองกลยุทธ์บนข้อมูล test รวม transaction cost เทียบกับ buy-and-hold
- **Pipeline**: Apache Airflow ดึงข้อมูล/predict รายวัน, retrain โมเดลรายสัปดาห์พร้อม
  quality gate ก่อน promote
- **Dashboard**: FastAPI + React แสดง signal ล่าสุด, กราฟราคา, ผล backtest
- **Testing/CI**: unit tests, data validation tests, model quality gate, GitHub Actions

## สถาปัตยกรรม (Architecture)

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

| ส่วน | เทคโนโลยี |
|---|---|
| Data & ML | Python, pandas, XGBoost/LightGBM, scikit-learn, MLflow |
| Orchestration | Apache Airflow, Docker Compose |
| Database | PostgreSQL |
| Backend | FastAPI |
| Frontend | React, Vite, TypeScript, Tailwind CSS, Recharts |
| CI/CD | GitHub Actions |
| Deployment | Vercel/Netlify (frontend), Render/Fly.io (backend), Managed Postgres |

## โครงสร้างโปรเจกต์

```
GoldForcast/
├── ml/               # data pipeline, feature engineering, models, signals, backtest
├── airflow/          # Airflow DAGs
├── backend/          # FastAPI service
├── frontend/         # React dashboard
├── notebooks/        # EDA
├── docs/             # เอกสารโครงงาน (PROJECT.md, ADR)
├── plan.md           # แผนงานแบ่งตาม Issue/Branch
└── docker-compose.yml
```

## เริ่มต้นใช้งาน (Getting Started)

> โปรเจกต์อยู่ระหว่างพัฒนา คำสั่งด้านล่างเป็น workflow ที่ตั้งใจไว้ (ดูความคืบหน้าจริงใน
> [`plan.md`](plan.md))

```bash
# ติดตั้ง dependencies ของ ml package
cd ml && pip install -e .

# รัน service ทั้งหมด (Postgres, Airflow, MLflow, FastAPI) แบบ local
docker compose up -d

# รัน frontend dev server
cd frontend && npm install && npm run dev

# รัน tests
pytest ml/tests backend/tests
```

## แผนพัฒนา (Roadmap)

งานแบ่งเป็น 8 Issue/Branch ตาม milestone (setup → data pipeline → modeling →
signal/backtest → db & orchestration → backend → frontend → CI/CD & deploy)
ดูรายละเอียดทั้งหมดที่ [`plan.md`](plan.md)

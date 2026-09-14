# GoldForcast — Issue & Branch Plan

เอกสารนี้แบ่งงานทั้งโครงงานออกเป็น 8 Issue ใหญ่ตาม Milestone — **1 Issue = 1 Branch** งานย่อย
ภายในแต่ละ Issue ทำเป็น commit ทยอยลงใน branch เดียวกันได้เลย ไม่ต้องแตก branch แยก
(เพราะงานฝั่ง `ml/` ส่วนใหญ่ต่อเนื่องกันเป็น pipeline เดียว แยก branch ละเอียดเกินไปจะ merge ยาก)
อ้างอิงรายละเอียดเชิงเทคนิคเต็มจาก [`docs/PROJECT.md`](docs/PROJECT.md)

**หลักการตั้งชื่อ branch**: `feat/<เลข issue>-<slug>`

---

## Issue #1 — Project Setup & Scaffolding
branch: `feat/01-project-setup`

- [ ] โครงสร้างโฟลเดอร์ monorepo (`ml/`, `airflow/`, `backend/`, `frontend/`, `notebooks/`, `.github/`)
- [ ] `.gitignore`, `.env.example`, `README.md`, `Makefile`
- [ ] `ml/pyproject.toml` + virtualenv, lint config (ruff/eslint)
- [ ] `docker-compose.yml` โครงเปล่า (จะเติม service ทีละ Issue ถัดไป)
- [ ] `CONTEXT.md` (ย้าย Domain Vocabulary จาก `docs/PROJECT.md`)

depends on: —

---

## Issue #2 — Data Pipeline: Ingestion, Features & EDA
branch: `feat/02-data-pipeline`

- [ ] `data/fetch.py` — ดึงข้อมูลจาก Yahoo Finance ทีละ ticker (GC=F, DX-Y.NYB, CL=F, ^VIX, ^TNX, ^GSPC)
- [ ] `data/validate.py` — schema/missing-value/range check, raise แบบ loud fail
- [ ] `data/merge.py` — รวมทุก source เป็น DataFrame เดียวด้วย Date
- [ ] `features/indicators.py` — MA, EMA, RSI, MACD
- [ ] `features/lags.py` + `labels.py` — lag features และ `make_targets(df, horizon)` รองรับ t+1/t+5/t+10
- [ ] `split.py` — chronological train/val/test split 60/20/20
- [ ] `notebooks/eda.ipynb` — trend, correlation, return distribution, volatility, scatter, outlier
- [ ] unit tests ครบทุกโมดูลข้างบน (`ml/tests/`)

depends on: #1

---

## Issue #3 — Model Training & Experiment Tracking
branch: `feat/03-model-training`

- [ ] เพิ่ม MLflow เข้า `docker-compose.yml`
- [ ] `models/train.py` — `train_horizon(horizon)`: XGBoost/LightGBM regressor+classifier +
      Logistic Regression baseline, log ผลเข้า MLflow
- [ ] เทรนครบทุก horizon (t+1, t+5, t+10) ผ่าน `config.HORIZONS`
- [ ] `models/registry.py` — pointer โมเดลที่ "promoted" ต่อ (horizon, task)
- [ ] LSTM baseline เทียบผล (ทำถ้ามีเวลาเหลือ)

depends on: #2

---

## Issue #4 — Signal Generation & Backtesting
branch: `feat/04-signal-backtest`

- [ ] `signals/rules.py` — `generate_signal()` ตาม threshold X/Y (BUY/SELL/HOLD + confidence)
- [ ] threshold optimizer หา X/Y จาก validation set เพื่อ maximize Sharpe ratio
- [ ] `backtest/engine.py` — Sharpe, cumulative return, win rate, vs buy-and-hold, transaction cost
- [ ] unit tests: threshold edge case, synthetic series ที่รู้คำตอบล่วงหน้า

depends on: #3

---

## Issue #5 — Database & Orchestration (Postgres + Airflow)
branch: `feat/05-db-orchestration`

- [ ] `db/models.py` — schema เต็ม (market_data, features_wide, model_registry, predictions,
      signals, outcomes, backtest_results) + เพิ่ม Postgres เข้า `docker-compose.yml`
- [ ] `db/io.py` — upsert/read helpers ใช้ร่วมกันทั้ง Airflow และ FastAPI
- [ ] สคริปต์ seed ทดลองรัน pipeline เต็ม (fetch→...→backtest) เข้า Postgres local
- [ ] เพิ่ม Airflow เข้า `docker-compose.yml` (`airflow/Dockerfile`)
- [ ] `daily_pipeline_dag.py` — fetch→validate→merge→features→inference (mapped ต่อ horizon)→write
- [ ] `retrain_dag.py` — retrain→evaluate quality gate→promote→log MLflow
- [ ] ทดสอบ backfill DAG เทียบผลกับสคริปต์ seed, ทดสอบเคส data source พังแล้ว DAG fail ดังๆ

depends on: #4

---

## Issue #6 — Backend API (FastAPI)
branch: `feat/06-backend-api`

- [ ] `backend/app/main.py`, `db.py`, `schemas.py`, CORS setup
- [ ] `routers/signals.py` — `/signals/latest`, `/signals/history`
- [ ] `routers/prices.py` — ราคาย้อนหลังสำหรับกราฟ
- [ ] `routers/backtest.py` — `/backtest/summary`
- [ ] backend tests: seed test DB, assert JSON response ทุก endpoint (ไม่ต้องพึ่ง Airflow จริง)

depends on: #5 (ใช้ schema เดียวกัน แต่เริ่มเขียนคู่ขนานได้ก่อน Airflow เสร็จจริงก็ได้)

---

## Issue #7 — Frontend Dashboard (React)
branch: `feat/07-frontend-dashboard`

- [ ] Vite + React + TypeScript + Tailwind scaffolding, `api/client.ts`
- [ ] `PriceChart.tsx` — กราฟราคาทอง + marker ของ signal ย้อนหลัง
- [ ] `SignalCard.tsx` — การ์ด signal ล่าสุดต่อ horizon พร้อม confidence
- [ ] `BacktestPanel.tsx` — Sharpe, cumulative return, win rate เทียบ buy-and-hold
- [ ] `Dashboard.tsx`, `Backtest.tsx` — ประกอบหน้าเต็ม ทดสอบกับ backend ที่มี fixture data จริง

depends on: #6

---

## Issue #8 — CI/CD & Deployment
branch: `feat/08-cicd-deploy`

- [ ] `.github/workflows/ci.yml` — lint + pytest (`ml/`, `backend/`) ทุก push/PR
- [ ] `.github/workflows/quality-gate.yml` — รัน backtest engine เทียบ threshold, ทดสอบทั้งเคสผ่าน/ไม่ผ่าน
- [ ] Deploy backend → Render/Fly.io + managed Postgres (Render Postgres/Supabase/Neon)
- [ ] Deploy frontend → Vercel/Netlify ชี้ไปที่ backend ที่ deploy แล้ว
- [ ] ตัดสินใจ Airflow: deploy บน VM เล็ก หรือทำวิดีโอ/screenshot ประกอบรายงานแทน

depends on: #5, #6, #7

---

## ลำดับความสัมพันธ์

```
#1 → #2 → #3 → #4 → #5 → #6 → #7 → #8
```

เส้นทางหลักค่อนข้างเรียงเป็นเส้นตรงเพราะ `ml/` เป็น pipeline ต่อเนื่อง แต่ทีม 5 คนแบ่งงานคู่ขนาน
ได้จริงในทางปฏิบัติ เช่น เริ่ม #7 (frontend scaffolding เปล่าๆ) หรือ #8 (เขียน CI workflow เปล่า)
รอไว้ล่วงหน้าระหว่างรอ #5/#6 เสร็จ แล้วค่อย merge logic เข้าไปทีหลัง

## สถานะ

ยังไม่เริ่มลงมือทำ issue ใดๆ — ไฟล์นี้เป็นแผนงานสำหรับสร้าง GitHub Issues และ branch จริง
เมื่อพร้อมเริ่มโครงงาน

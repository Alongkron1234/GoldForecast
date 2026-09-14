# GoldForcast — เอกสารสรุปโครงงาน

## บริบท (Context)

GoldForcast เป็นโครงงานวิชา CPE232 (ทีม 5 คน) ต่อยอดจาก proposal เดิมเรื่อง
**"Gold Price Direction & Return Forecasting"** โดยทีมต้องการขยาย scope จากงานวิเคราะห์ข้อมูล +
ML เชิงวิชาการ ให้กลายเป็นระบบระดับ portfolio/resume: มี data pipeline ที่ orchestrate จริง,
เว็บ dashboard ที่ deploy ใช้งานได้จริง, และมีชุด automated testing/CI ครบ — เพื่อให้โครงงานที่
จบออกมาแสดงทักษะทั้ง data engineering และ MLOps ไม่ใช่แค่ notebook วิเคราะห์ข้อมูล

ตอนเริ่มต้น repository (`/Users/alongkorn/Projects/GoldForcast`) ยังว่างเปล่า เป็นการเริ่มสร้างใหม่ทั้งหมด

---

## Scope เดิมจาก Proposal (ไม่เปลี่ยนแปลง)

- **ข้อมูล**: ราคาทองคำรายวันย้อนหลัง (OHLCV) + ดัชนีเศรษฐกิจมหภาค — ค่าเงินดอลลาร์ (DXY),
  ราคาน้ำมันดิบ (CL=F), ดัชนีความผันผวน (VIX), อัตราผลตอบแทนพันธบัตรรัฐบาล 10 ปี (TNX),
  ดัชนีตลาดหุ้น (S&P 500 / GSPC) — แหล่งข้อมูล: Yahoo Finance
- **Preprocessing**: รวมข้อมูลทุกแหล่งด้วย Date, จัดการ missing values, ปรับให้อยู่ในรูปแบบ time series
- **EDA**: วิเคราะห์แนวโน้มราคา, correlation, distribution ของ return, ความผันผวน, scatter plot, outlier
- **Feature Engineering**: Technical indicators (MA, EMA, RSI, MACD) + Lag features
- **แบ่งข้อมูล**: แบบ time series — train 60% / validation 20% / test 20%
- **Hybrid Model**:
  - Regression → ทำนาย % การเปลี่ยนแปลงราคาปิดของทองในวันถัดไป (t+1)
  - Classification → ทำนายทิศทาง (ขึ้น/ลง) ของวันถัดไป
- **Signal Generation**: รวมผลจากทั้งสองโมเดล สร้างสัญญาณ BUY/SELL/HOLD พร้อม Confidence Score
- **Backtesting**: ประเมินผลกลยุทธ์ signal บนชุดข้อมูล test ที่โมเดลไม่เคยเห็น

---

## Scope ที่เพิ่มเข้ามา (จากการ grill ความต้องการ)

### 1. Data Pipeline — Apache Airflow
- ใช้ Apache Airflow จริง (ไม่ใช่ตัวช่วยแบบเบาๆ) orchestrate: ดึงข้อมูล → clean/merge →
  feature engineering → predict → เขียนผลลง database
- รันผ่าน Docker Compose (local)
- **ความถี่**: ดึงข้อมูลใหม่ทุกวัน (ทำให้ dashboard ไม่ static) ส่วนการ retrain โมเดลทำแบบ
  รายสัปดาห์/on-demand ไม่ใช่ทุกวัน
- **ช่วงข้อมูลย้อนหลัง**: 10 ปี (~2016–ปัจจุบัน) ครอบคลุมทั้งช่วงดอกเบี้ยต่ำ, COVID shock,
  และช่วงดอกเบี้ยขาขึ้นล่าสุด เหลือข้อมูลประมาณ 6 ปีสำหรับ training

### 2. การเลือกโมเดล
- **แนวทางหลัก**: Classical ML — XGBoost/LightGBM สำหรับทั้ง regression และ classification
  พร้อม Logistic Regression เป็น baseline สำหรับ classification
  เหตุผล: ข้อมูลการเงินรายวันมี noise สูงและ sample size จำกัด tree ensemble ทนทานกว่า
  deep learning ในสถานการณ์นี้ และตีความ feature importance ได้ ช่วยเรื่อง EDA/รายงาน
- **Baseline เสริม**: LSTM/GRU ใส่ไว้เทียบผลเฉยๆ ถ้ามีเวลา
- **Multi-horizon forecasting**: นอกจากทำนาย t+1 (ตาม proposal) ยังเพิ่มทำนาย t+5 และ t+10
  (วันทำการ) เหตุผล: ราคาทองมีพฤติกรรมใกล้เคียง random walk ความแม่นยำของโมเดลจะลดลงตาม
  horizon ที่ยาวขึ้น — ส่วนขยายนี้ตอบคำถามสำคัญได้โดยตรงว่า "โมเดลมี edge จริงถึงระยะไหน"
  และกลายเป็นผลการค้นพบที่มีคุณค่าในรายงาน (กราฟ accuracy/Sharpe เทียบตาม horizon)
- **Experiment Tracking**: ใช้ MLflow (containerized) log params/metrics/artifacts ทุกรอบเทรน
  ช่วยเรื่อง tune threshold ของ signal (ด้านล่าง) และเป็นจุดขายด้าน resume ด้วย

### 3. กฎการสร้าง Signal
- ใช้ Rule-based threshold (ตรงกับที่ proposal ระบุว่า "สร้างกฎการตัดสินใจ"):
  BUY ถ้า predicted_direction = ขึ้น และ predicted_return_pct > +X% และ confidence > Y;
  SELL ในเงื่อนไขตรงข้าม; นอกนั้น HOLD
- X และ Y เป็น hyperparameter ที่ tune จาก validation set เพื่อ maximize Sharpe ratio จาก backtest
  — ทำให้มีการทดลอง tuning จริงให้รายงานได้
- ทางเลือกที่พิจารณาแล้วไม่เลือก: การเทรนโมเดลที่ 3 (meta-model) ที่รับ output จาก 2 โมเดล
  แล้วทำนาย BUY/SELL/HOLD โดยตรง — แม่นยำกว่าในทางทฤษฎี แต่ซับซ้อนกว่าและเสี่ยง overfit
  เพราะข้อมูลรายวันมีจำกัด

### 4. ความสมจริงของ Backtest
- คิดรวม transaction cost/slippage (% คงที่ต่อการเทรด) ให้ backtest สะท้อนต้นทุนการเทรดจริง
- Position sizing: ใช้ fixed size ต่อการเทรดเป็น baseline ส่วนแบบ scale ตาม confidence
  เป็นการทดลองเสริมเทียบกัน ไม่ใช่ default

### 5. ที่เก็บข้อมูล
- PostgreSQL รันเป็น Docker container ตอน local (ใน docker-compose.yml เดียวกับ Airflow)
  ตอน deploy จริงจะสลับไปใช้ managed Postgres (เช่น Render Postgres / Supabase / Neon
  free tier) เพราะ hosting ฟรีส่วนใหญ่ไม่มี persistent disk ที่เชื่อถือได้สำหรับรัน Postgres เอง

### 6. Backend / Frontend — แยกกัน
- **Backend**: FastAPI — อ่านผลพยากรณ์/signal ที่ Airflow เขียนไว้ใน Postgres มา serve เท่านั้น
  ไม่รัน inference สด สอดคล้องกับจังหวะพยากรณ์รายวัน ทำให้ API layer เบา
- **Frontend**: React + Vite + TypeScript, Recharts (หรือ library แบบ lightweight-charts
  สำหรับกราฟแท่งเทียน), Tailwind CSS — เลือกแทน Next.js เพราะไม่ต้องการ SEO/SSR

### 7. การ Deploy
- Deploy จริง ไม่ใช่แค่ demo local เพราะกลุ่มเป้าหมายรวมถึงคนดู resume ที่จะไม่รันโปรเจกต์เอง:
  - Frontend → Vercel หรือ Netlify
  - Backend → Render หรือ Fly.io
  - Database → managed Postgres
  - Airflow → รันบน VM เล็กๆ หรือแสดงผ่าน screenshot/วิดีโอในรายงานแทนการเปิดสาธารณะ
    (หลีกเลี่ยงค่าใช้จ่าย/ความเสี่ยงด้าน security จากการเปิด Airflow webserver สู่สาธารณะ)

### 8. Testing & CI
- Unit tests (pytest) สำหรับ feature engineering และ preprocessing — โค้ดที่เสี่ยงพังง่ายและถูก
  ใช้ซ้ำมากที่สุด
- Data validation tests ที่รันทุกครั้งหลัง pipeline ดึงข้อมูล (schema, missing values, ช่วงค่าที่
  สมเหตุสมผล) — ป้องกันการพังเงียบๆ ถ้า source เปลี่ยนรูปแบบ
- **Model Quality Gate**: คนละอย่างกับการวิเคราะห์ backtest ใน proposal (ซึ่งเป็นการประเมินผล
  ครั้งเดียว/ตามรอบสำหรับรายงาน) — อันนี้คือการเช็คอัตโนมัติใน CI: โมเดลที่เทรนใหม่ต้องผ่าน
  threshold ของ backtest metric ก่อนถูก promote ไปใช้งานจริง ไม่งั้น pipeline จะ fail แทนที่จะ
  deploy โมเดลที่แย่ลงแบบเงียบๆ
- GitHub Actions รัน lint + unit tests + quality gate ทุกครั้งที่ push/PR

---

## คำศัพท์หลักของโครงงาน (Domain Vocabulary)

- **Prediction (ผลพยากรณ์)**: ผลลัพธ์ของโมเดลสำหรับ 1 horizon บน 1 target date ประกอบด้วย
  `predicted_return_pct` (จาก regression) และ `predicted_direction` (จาก classification
  พร้อม probability)
- **Horizon**: ระยะเวลาที่ Prediction มองไปข้างหน้า — t+1, t+5, หรือ t+10 วันทำการ
- **Signal**: BUY / SELL / HOLD ที่แปลงมาจาก Prediction ผ่านกฎ threshold — ไม่ใช่ผลลัพธ์ดิบ
  จากโมเดลโดยตรง
- **Confidence Score**: ค่า probability ของ classifier สำหรับทิศทางที่ทำนาย ติดมากับ Signal
- **Backtest**: การจำลองผลของกลยุทธ์ Signal บนช่วงข้อมูล test (รวม transaction cost)
  ใช้ประเมินและ tune กลยุทธ์ (คนละอย่างกับ Quality Gate)
- **Quality Gate**: เกณฑ์ขั้นต่ำของ backtest metric ที่โมเดลที่เทรนใหม่ต้องผ่านใน CI ก่อนถูก
  promote ให้ API นำไปใช้งาน

---

## แผนการ Implementation (สรุปย่อ)

โครงสร้างแบบ monorepo แยกเป็น `ml/` (pipeline + โมเดล, package เดียวที่ทั้ง Airflow และ
FastAPI import ใช้ร่วมกัน), `airflow/` (DAGs), `backend/` (FastAPI), `frontend/` (React),
`notebooks/` (EDA), และ `.github/workflows/` (CI)

ลำดับการสร้าง (10 phase):
1. Repo scaffolding
2. Data ingestion + validation
3. Feature engineering + EDA
4. Model training + MLflow + multi-horizon
5. Signal generation + backtesting
6. Postgres schema + DB layer
7. Airflow DAGs
8. FastAPI backend
9. React frontend
10. CI/CD + deployment

รายละเอียดเชิงเทคนิคแบบเต็ม (โครงสร้างไฟล์, schema, ไฟล์วิกฤตที่ต้องแก้) อยู่ในแผนงานที่
วางไว้แล้ว พร้อมเริ่มดำเนินการเมื่อได้รับสัญญาณให้เริ่มลงมือทำ

---

## สถานะปัจจุบัน

ยืนยันความเข้าใจร่วมกันเรียบร้อยแล้ว มีแผน implementation ที่ชัดเจนพร้อมใช้งาน — **ยังไม่เริ่ม
ลงมือเขียนโค้ด** ตามที่ผู้ใช้แจ้งให้รอก่อน

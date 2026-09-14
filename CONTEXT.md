# GoldForcast

ระบบพยากรณ์ทิศทางและ % ผลตอบแทนราคาทองคำรายวัน ด้วย hybrid ML (classification + regression)
พร้อมแปลงผลเป็นสัญญาณซื้อขายและประเมินผลผ่าน backtesting

## Language

**Prediction**:
ผลลัพธ์ของโมเดลสำหรับหนึ่ง Horizon บนหนึ่ง target date ประกอบด้วย `predicted_return_pct`
(จาก regression model) และ `predicted_direction` (จาก classification model พร้อม probability)
_Avoid_: Forecast, output

**Horizon**:
ระยะเวลาที่ Prediction มองไปข้างหน้า — t+1, t+5, หรือ t+10 วันทำการ
_Avoid_: Window, lookahead period

**Signal**:
BUY / SELL / HOLD ที่แปลงมาจาก Prediction ผ่านกฎ rule-based threshold ไม่ใช่ผลลัพธ์ดิบจาก
โมเดลโดยตรง
_Avoid_: Recommendation, action

**Confidence Score**:
ค่า probability ของ classifier สำหรับทิศทางที่ทำนาย ติดมากับ Signal เสมอ
_Avoid_: Probability (ใช้คำนี้เฉพาะตอนอธิบาย raw classifier output ก่อนแปลงเป็น Signal)

**Backtest**:
การจำลองผลของกลยุทธ์ Signal บนช่วงข้อมูล test (รวม transaction cost) ใช้ประเมินและ tune
กลยุทธ์ — คนละอย่างกับ Quality Gate
_Avoid_: Simulation, evaluation

**Quality Gate**:
เกณฑ์ขั้นต่ำของ backtest metric ที่โมเดลที่เทรนใหม่ต้องผ่านใน CI ก่อนถูก promote ให้ API
นำไปใช้งานจริง — เป็นการเช็คอัตโนมัติแบบผ่าน/ไม่ผ่าน ไม่ใช่ตัว Backtest analysis เอง
_Avoid_: Threshold (คำนี้ใช้เฉพาะตอนพูดถึง X/Y ในกฎของ Signal)

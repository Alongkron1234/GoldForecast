"""One-off script: fetch ~10 years of real market data, run it through the
full feature pipeline, and cache the result locally under ml/data/raw/ so
the EDA notebook (and quick local experiments) have real data to work with
before Postgres exists (see plan.md, Issue #5)."""

from datetime import date, timedelta
from pathlib import Path

from goldforecast import config
from goldforecast.data.fetch import fetch_source
from goldforecast.data.merge import merge_sources
from goldforecast.data.validate import missing_value_check, range_check, schema_check
from goldforecast.features.build import build_feature_matrix
from goldforecast.labels import make_targets
from goldforecast.split import chronological_split

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    # กำหนดช่วงเวลาในการดึงข้อมูลย้อนหลัง 10 ปีจากวันนี้
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * 10)).isoformat()

    print(f"fetching {start} .. {end} for {len(config.TICKERS)} tickers")

    # วน fetch ข้อมูลสำหรับแต่ละ ticker และตรวจสอบความถูกต้องของข้อมูล
    sources = {}
    for name, ticker in config.TICKERS.items():
        df = fetch_source(ticker, start=start, end=end)
        schema_check(df)
        missing_value_check(df)
        # Only gold gets a strict non-negative check — crude oil futures (CL=F)
        # legitimately traded negative on 2020-04-20 during the COVID storage
        # crunch, so a lower-bound check there would be a false positive.
        if name == "gold":
            range_check(df, "Close", min_value=0)
        sources[name] = df
        date_range = f"{df['Date'].min().date()} .. {df['Date'].max().date()}"
        print(f"  {name} ({ticker}): {len(df)} rows, {date_range}")

    # รวม sources เข้าด้วยกันเป็นตารางเดียว
    merged = merge_sources(sources)
    print(f"merged: {merged.shape}")

    # สร้าง feature matrix ทั้งหมด
    featured = build_feature_matrix(merged)

    # สร้าง target columns สำหรับแต่ละ horizon
    for horizon in config.HORIZONS:
        featured = make_targets(featured, horizon)
    print(f"featured (with targets): {featured.shape}")

    # แบ่ง train test split
    train, val, test = chronological_split(featured)
    print(f"split: train={len(train)}, val={len(val)}, test={len(test)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # บันทึกไฟล์ CSV ของ dataset ที่สร้างขึ้น
    featured.to_csv(OUT_DIR / "featured_dataset.csv", index=False)
    for name, df in sources.items():
        df.to_csv(OUT_DIR / f"{name}_raw.csv", index=False)

    print(f"cached to {OUT_DIR}")


if __name__ == "__main__":
    main()

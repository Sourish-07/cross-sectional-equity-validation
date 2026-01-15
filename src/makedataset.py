"""
S.T.E.L.L.A.R. Master Dataset Creator — Research-Grade (2026)

Design goals:
- Leakage-safe
- Rate-limit safe
- Adjustment-aware
- Defensible under academic review
"""

import os
import time
import logging
import warnings
from datetime import timedelta

import pandas as pd
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
ROOT_FOLDER = "Data"                     # Kaggle legacy data
OUTPUT_FILE = "master_stock_data_2000_2026.csv"
METADATA_FILE = "ticker_metadata.csv"
LOG_FILE = "dataset_creation_log.txt"
TEMP_FOLDER = "TempChunks"

MIN_DATE = pd.Timestamp("2000-01-01")
MIN_YAHOO_DATE = pd.Timestamp("2017-01-01")

YF_BATCH_SIZE = 25            # Conservative for rate limits
YF_RETRIES = 5
YF_SLEEP = 5.0                # Seconds between batches

CHUNK_SIZE = 750_000
RET_THRESHOLD = 3.0           # ±300% daily return cap

BAD_TICKER_SUFFIXES = (
    "_A","_B","_C","_D","_E","_F","_G","_H",
    "_P","_WS","_W","_U","_Z",".OLD",".TEST"
)

os.makedirs(TEMP_FOLDER, exist_ok=True)

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING MASTER DATASET CREATION (FINAL) ===")

# ===================== HELPERS =====================
def safe_date(series):
    s = pd.to_datetime(series, errors="coerce")
    if s.isna().sum():
        logger.warning(f"Invalid dates coerced to NaT: {s.isna().sum():,}")
    return s

def valid_ticker(t):
    if not (1 <= len(t) <= 10):
        return False
    if any(t.endswith(s) for s in BAD_TICKER_SUFFIXES):
        return False
    return True

# ===================== PHASE 1: LEGACY =====================
def load_legacy():
    logger.info("Phase 1: Loading legacy Kaggle data")
    dfs = []
    total, skipped = 0, 0

    for folder in ["Stocks", "ETFs"]:
        path = os.path.join(ROOT_FOLDER, folder)
        if not os.path.exists(path):
            continue

        files = [f for f in os.listdir(path) if f.lower().endswith(".txt")]
        total += len(files)

        for fname in tqdm(files, desc=f"Legacy {folder}"):
            ticker = fname.replace(".us.txt","").replace(".txt","").upper()
            if not valid_ticker(ticker):
                skipped += 1
                continue

            full_path = os.path.join(path, fname)
            df = None

            for sep in [",", "|"]:
                try:
                    tmp = pd.read_csv(full_path, sep=sep)
                    if "Date" in tmp.columns or "date" in tmp.columns:
                        df = tmp
                        break
                except Exception:
                    continue

            if df is None or len(df) < 10:
                skipped += 1
                continue

            df.columns = [c.lower().strip() for c in df.columns]
            if not {"date","close"}.issubset(df.columns):
                skipped += 1
                continue

            df["date"] = safe_date(df["date"])
            df = df.dropna(subset=["date","close"])
            df = df[df["date"] >= MIN_DATE]

            keep = ["date","open","high","low","close","volume"]
            df = df[[c for c in keep if c in df.columns]]
            df["ticker"] = ticker
            df["source"] = "legacy"

            dfs.append(df)

    legacy = pd.concat(dfs, ignore_index=True)
    legacy = legacy.sort_values(["ticker","date"]).drop_duplicates(["ticker","date"])

    logger.info(
        f"Legacy loaded: {len(legacy):,} rows | "
        f"{total - skipped} files used | {skipped} skipped"
    )

    return legacy

# ===================== PHASE 2: YAHOO =====================
def update_yahoo(legacy):
    logger.info("Phase 2: Yahoo Finance updates")

    last_dates = (
        legacy[legacy["date"] >= MIN_YAHOO_DATE]
        .groupby("ticker")["date"].max()
    )

    tickers = [t for t in last_dates.index if valid_ticker(t)]
    failed = []

    chunk_id = 0
    for i in tqdm(range(0, len(tickers), YF_BATCH_SIZE), desc="Yahoo batches"):
        batch = tickers[i:i+YF_BATCH_SIZE]
        start = (last_dates[batch].min() + timedelta(days=1)).strftime("%Y-%m-%d")

        for attempt in range(YF_RETRIES):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    interval="1d",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )
                if not data.empty:
                    break
            except Exception as e:
                logger.warning(f"Retry {attempt+1}/{YF_RETRIES}: {e}")
                time.sleep((attempt+1)*5)
        else:
            failed.extend(batch)
            continue

        updates = []
        for t in batch:
            try:
                df_t = data.xs(t, level=1, axis=1)
                df_t = df_t.reset_index()
                df_t.columns = [c.lower() for c in df_t.columns]

                keep = ["date","open","high","low","close","volume"]
                df_t = df_t[[c for c in keep if c in df_t.columns]]
                df_t["ticker"] = t
                df_t["source"] = "yahoo"

                updates.append(df_t)
            except Exception:
                failed.append(t)

        if updates:
            out = pd.concat(updates, ignore_index=True)
            out.to_csv(
                os.path.join(TEMP_FOLDER, f"yahoo_{chunk_id:03d}.csv"),
                index=False
            )
            chunk_id += 1

        time.sleep(YF_SLEEP)

    if failed:
        logger.warning(f"Yahoo failures: {len(failed):,} tickers")

# ===================== PHASE 3: MERGE + FILTER =====================
def finalize():
    logger.info("Phase 3: Final merge & validation")

    chunks = [
        pd.read_csv(os.path.join(TEMP_FOLDER, f), parse_dates=["date"])
        for f in os.listdir(TEMP_FOLDER)
        if f.endswith(".csv")
    ]

    full = pd.concat(chunks, ignore_index=True)
    full = full.dropna(subset=["date","close","ticker"])
    full = full.sort_values(["ticker","date"])
    full = full.drop_duplicates(["ticker","date"])

    # ---- Return-based sanity filter ----
    full["ret"] = full.groupby(["ticker","source"])["close"].pct_change()
    bad = full["ret"].abs() > RET_THRESHOLD

    logger.info(f"Removing {bad.sum():,} rows with |ret| > {RET_THRESHOLD:.0f}x")
    full = full[~bad].drop(columns="ret")

    # ---- Metadata ----
    meta = full.groupby("ticker").agg(
        first_date=("date","min"),
        last_date=("date","max"),
        n_days=("date","count"),
        min_close=("close","min"),
        max_close=("close","max")
    ).reset_index()

    meta["lifespan_years"] = (
        (meta["last_date"] - meta["first_date"]).dt.days / 365.25
    )
    meta["long_lived_5y"] = meta["lifespan_years"] >= 5
    meta["long_lived_10y"] = meta["lifespan_years"] >= 10

    meta.to_csv(METADATA_FILE, index=False)
    full.to_csv(OUTPUT_FILE, index=False)

    logger.info("=== DATASET COMPLETE ===")
    logger.info(f"Rows: {len(full):,}")
    logger.info(f"Tickers: {full['ticker'].nunique():,}")
    logger.info(f"Date range: {full['date'].min().date()} → {full['date'].max().date()}")

# ===================== MAIN =====================
if __name__ == "__main__":
    legacy = load_legacy()
    legacy.to_csv(os.path.join(TEMP_FOLDER, "legacy.csv"), index=False)

    update_yahoo(legacy)
    finalize()
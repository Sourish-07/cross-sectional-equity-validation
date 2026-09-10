# diagnostics/debug_systemic_dates.py
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/features.parquet")[["date", "ticker", "ret_fwd"]]

# For each date, what fraction of tickers show an extreme forward return?
def frac_extreme(x, thresh=0.30):
    return (x.abs() > thresh).mean()

daily_extreme_frac = df.groupby("date")["ret_fwd"].apply(frac_extreme).sort_values(ascending=False)
print("Top 20 dates by fraction of tickers with |ret_fwd| > 30%:")
print(daily_extreme_frac.head(20))

# Check specific known split ratios: -0.50 (2:1), -0.667 (3:1), -0.75 (4:1), -0.857 (7:1)
worst_date = daily_extreme_frac.index[0]
sub = df[df["date"] == worst_date]
print(f"\nOn {worst_date}, ret_fwd distribution across tickers:")
print(sub["ret_fwd"].describe())
print("\nSample of extreme values on that date:")
print(sub[sub["ret_fwd"].abs() > 0.30][["ticker", "ret_fwd"]].sort_values("ret_fwd").head(20))
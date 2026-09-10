# diagnostics/debug_1110_check.py
import pandas as pd
raw = pd.read_csv("data/master_stock_data_paper2_snapshot.csv", parse_dates=["date"],
                   usecols=["date","ticker","close","volume"])
sub = raw[(raw["ticker"]=="GOOGL") & (raw["date"] >= "2017-11-05") & (raw["date"] <= "2017-11-15")]
print(sub)
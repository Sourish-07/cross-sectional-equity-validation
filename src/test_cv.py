# test_cv.py
import pandas as pd
from validation import CombinatorialPurgedCV

df = pd.read_parquet("data/processed/features.parquet")

# Optional subsampling for ultra-fast testing
# df = df[df['date'] >= '2015-01-01']
# df = df[df['ticker'].isin(df['ticker'].unique()[:1000])]

cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2)

print(f"Number of combinatorial splits: {cv.get_n_splits()}")

splits = cv.split(df)  # generator - don't list() all if memory tight
for i, (train_idx, test_list) in enumerate(list(splits)[:3]):
    test_idx = test_list[0]
    print(f"Split {i}: Train rows = {len(train_idx):,}, Test rows = {len(test_idx):,}")
print("CPCV test successful - no memory crash!")
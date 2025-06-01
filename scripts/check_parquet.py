import pandas as pd
from pathlib import Path
parquet_file = Path(__file__).resolve().parents[1] / "PQ" / "scrobble.parquet"
df = pd.read_parquet(parquet_file)
print("Current columns in scrobble.parquet:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

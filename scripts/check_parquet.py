import pandas as pd
from pathlib import Path

# Path to the parquet file
parquet_file = Path(__file__).resolve().parents[1] / "PQ" / "scrobble.parquet"

# Read the parquet file
df = pd.read_parquet(parquet_file)

# Print the columns
print("Current columns in scrobble.parquet:")
print(df.columns.tolist())

# Print the first few rows
print("\nFirst 5 rows:")
print(df.head())
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

# Path to the parquet file
parquet_path = Path("PQ/artist_info.parquet")

# Read the parquet file
table = pq.read_table(parquet_path)
df = table.to_pandas()

# Print basic information
print(f"File size: {parquet_path.stat().st_size} bytes")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")

# Print the first few rows
print("\nFirst 5 rows:")
print(df.head())

# Check for any issues
print("\nNull values per column:")
print(df.isnull().sum())

# Check for duplicates
duplicates = df.duplicated(subset=["artist_name"], keep=False)
print(f"\nNumber of duplicate artist names: {duplicates.sum()}")
if duplicates.sum() > 0:
    print("\nSample of duplicates:")
    print(df[duplicates].head())
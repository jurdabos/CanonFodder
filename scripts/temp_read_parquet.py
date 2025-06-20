import pyarrow.parquet as pq
from pathlib import Path
parquet_path = Path("PQ/artist_info.parquet")
table = pq.read_table(parquet_path)
df = table.to_pandas()
print(f"File size: {parquet_path.stat().st_size} bytes")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nNull values per column:")
print(df.isnull().sum())
duplicates = df.duplicated(subset=["artist_name"], keep=False)
print(f"\nNumber of duplicate artist names: {duplicates.sum()}")
if duplicates.sum() > 0:
    print("\nSample of duplicates:")
    print(df[duplicates].head())

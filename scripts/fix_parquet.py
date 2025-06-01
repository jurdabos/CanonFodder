import pandas as pd
from pathlib import Path

parquet_file = Path(__file__).resolve().parents[1] / "PQ" / "scrobble.parquet"
df = pd.read_parquet(parquet_file)
print("Original columns in scrobble.parquet:")
print(df.columns.tolist())
required_columns = ['artist_name', 'album_title', 'track_title', 'play_time', 'artist_mbid']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit(1)
df_fixed = df[required_columns]
print("\nNew columns in scrobble.parquet:")
print(df_fixed.columns.tolist())
df_fixed.to_parquet(parquet_file, compression='snappy', index=False)
print("\nParquet file has been fixed successfully!")

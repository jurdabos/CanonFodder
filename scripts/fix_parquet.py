import pandas as pd
from pathlib import Path

# Path to the parquet file
parquet_file = Path(__file__).resolve().parents[1] / "PQ" / "scrobble.parquet"

# Read the parquet file
df = pd.read_parquet(parquet_file)

# Print the original columns
print("Original columns in scrobble.parquet:")
print(df.columns.tolist())

# Keep only the required columns according to the documentation
required_columns = ['artist_name', 'album_title', 'track_title', 'play_time', 'artist_mbid']

# Check if all required columns exist
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Error: Missing required columns: {missing_columns}")
    exit(1)

# Create a new DataFrame with only the required columns
df_fixed = df[required_columns]

# Print the new columns
print("\nNew columns in scrobble.parquet:")
print(df_fixed.columns.tolist())

# Save the fixed DataFrame back to the parquet file
df_fixed.to_parquet(parquet_file, compression='snappy', index=False)

print("\nParquet file has been fixed successfully!")

"""
Script to fix the scrobble.parquet file by removing extra columns.
"""
import pandas as pd
from pathlib import Path
# Define paths
PQ_DIR = Path(__file__).resolve().parents[1] / "PQ"
SCROBBLE_PARQUET = PQ_DIR / "scrobble.parquet"
# Define expected columns
EXPECTED_COLUMNS = ["artist_name", "album_title", "play_time", "track_title", "artist_mbid"]


def fix_scrobble_parquet():
    """
    Fix the scrobble.parquet file by removing extra columns.
    """
    print(f"Reading {SCROBBLE_PARQUET}...")
    df = pd.read_parquet(SCROBBLE_PARQUET)
    
    # Print current columns
    print(f"Current columns: {df.columns.tolist()}")
    
    # Keep only expected columns that exist in the dataframe
    available_columns = [col for col in EXPECTED_COLUMNS if col in df.columns]
    
    # Check if we have all expected columns
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing expected columns: {missing_columns}")
    
    # Create a new dataframe with only the expected columns
    df_fixed = df[available_columns]
    
    # Create a backup of the original file
    backup_path = SCROBBLE_PARQUET.with_suffix('.parquet.bak')
    print(f"Creating backup at {backup_path}...")
    df.to_parquet(backup_path, index=False)
    
    # Save the fixed dataframe
    print(f"Saving fixed dataframe to {SCROBBLE_PARQUET}...")
    df_fixed.to_parquet(SCROBBLE_PARQUET, index=False)
    
    print(f"Fixed columns: {df_fixed.columns.tolist()}")
    print(f"Removed columns: {[col for col in df.columns if col not in df_fixed.columns]}")
    print(f"Done! Scrobble parquet file has been fixed.")

if __name__ == "__main__":
    fix_scrobble_parquet()
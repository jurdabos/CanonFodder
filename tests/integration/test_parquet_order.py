"""
Test script to verify that the parquet file columns are in the expected order.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from helpers import io

def test_parquet_column_order():
    """Test that the parquet file columns are in the expected order."""
    # Create a test DataFrame with columns in a random order
    test_data = {
        "track_title": ["Song1", "Song2", "Song3"],
        "artist_name": ["Artist1", "Artist2", "Artist3"],
        "album_title": ["Album1", "Album2", "Album3"],
        "artist_mbid": ["mbid1", "mbid2", "mbid3"],
        "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    }
    df = pd.DataFrame(test_data)
    
    # Save the DataFrame to a test parquet file
    test_file = Path("PQ") / "test_column_order.parquet"
    io.append_or_create_parquet(df, test_file)
    
    # Load the parquet file and check the column order
    loaded_df = pd.read_parquet(test_file)
    print(f"Loaded columns: {loaded_df.columns.tolist()}")
    
    # Expected column order
    expected_order = ["artist_name", "album_title", "play_time", "track_title", "artist_mbid"]
    
    # Check if the columns are in the expected order
    actual_order = loaded_df.columns.tolist()
    if actual_order == expected_order:
        print("SUCCESS: Columns are in the expected order!")
    else:
        print(f"FAILURE: Columns are not in the expected order.")
        print(f"Expected: {expected_order}")
        print(f"Actual: {actual_order}")
    
    # Clean up the test file
    test_file.unlink(missing_ok=True)

if __name__ == "__main__":
    test_parquet_column_order()
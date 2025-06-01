"""
Test script to verify that io.dump_parquet() maintains the correct column order.
This simulates the specific operations in profile.py lines 158-161.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from helpers import io
from DB import engine
from DB.ops import load_scrobble_table_from_db_to_df

def test_dump_parquet():
    """Test that io.dump_parquet() maintains the correct column order."""
    # Create a test DataFrame with columns in a random order
    test_data = {
        "track_title": ["Song1", "Song2", "Song3"],
        "artist_name": ["Artist1", "Artist2", "Artist3"],
        "album_title": ["Album1", "Album2", "Album3"],
        "artist_mbid": ["mbid1", "mbid2", "mbid3"],
        "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    }
    df = pd.DataFrame(test_data)

    # Save the DataFrame to a test parquet file using io.dump_parquet()
    test_file = Path("PQ") / "test_dump_parquet.parquet"

    # Simulate the io.dump_parquet() call in profile.py
    io.dump_parquet(df, constant=True)

    # Now check if the constant file (scrobble.parquet) has the correct column order
    constant_file = Path("PQ") / "scrobble.parquet"
    if constant_file.exists():
        loaded_df = pd.read_parquet(constant_file)
        print(f"Loaded columns from scrobble.parquet: {loaded_df.columns.tolist()}")

        # Expected column order
        expected_order = ["artist_name", "album_title", "play_time", "track_title", "artist_mbid"]

        # Check if the columns are in the expected order
        actual_order = loaded_df.columns.tolist()
        if actual_order == expected_order:
            print("SUCCESS: Columns in scrobble.parquet are in the expected order!")
        else:
            print(f"FAILURE: Columns in scrobble.parquet are not in the expected order.")
            print(f"Expected: {expected_order}")
            print(f"Actual: {actual_order}")
    else:
        print("ERROR: scrobble.parquet was not created.")

    # Clean up the test file
    test_file.unlink(missing_ok=True)

if __name__ == "__main__":
    test_dump_parquet()
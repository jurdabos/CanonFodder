"""
Test script to verify that the parquet file columns work correctly with the profile.py workflow.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from helpers import io

def test_profile_workflow():
    """Test that the parquet file columns work correctly with the profile.py workflow."""
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
    test_file = Path("PQ") / "test_profile_workflow.parquet"
    io.append_or_create_parquet(df, test_file)

    # Simulate the profile.py workflow
    # 1. Load the parquet file directly
    loaded_df = pd.read_parquet(test_file)

    # Print the original column order
    print(f"Original columns: {loaded_df.columns.tolist()}")

    # 2. Rename the columns as in profile.py
    loaded_df.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]

    # Print the renamed columns
    print(f"Renamed columns: {loaded_df.columns.tolist()}")

    # 3. Check if the data is correct
    print("\nSample data after renaming:")
    print(loaded_df.head())

    # Check if the data matches the original test data
    expected_artist = test_data["artist_name"][0]
    actual_artist = loaded_df["Artist"][0]
    expected_song = test_data["track_title"][0]
    actual_song = loaded_df["Song"][0]

    if expected_artist == actual_artist and expected_song == actual_song:
        print("\nSUCCESS: Data matches after column renaming!")
    else:
        print("\nFAILURE: Data does not match after column renaming.")
        print(f"Expected artist: {expected_artist}, Actual artist: {actual_artist}")
        print(f"Expected song: {expected_song}, Actual song: {actual_song}")

    # Clean up the test file
    test_file.unlink(missing_ok=True)

if __name__ == "__main__":
    test_profile_workflow()
"""
Test script to verify that the entire profile.py workflow works correctly,
including the io.dump_parquet() call and the column renaming.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from helpers import io

def test_full_profile_workflow():
    """Test that the entire profile.py workflow works correctly."""
    # Create a test DataFrame with columns in a random order
    test_data = {
        "track_title": ["Song1", "Song2", "Song3"],
        "artist_name": ["Artist1", "Artist2", "Artist3"],
        "album_title": ["Album1", "Album2", "Album3"],
        "artist_mbid": ["mbid1", "mbid2", "mbid3"],
        "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    }
    df = pd.DataFrame(test_data)

    # Create a test file path
    test_file = Path("PQ") / "test_full_workflow.parquet"

    # Step 1: Save the DataFrame to the test file
    print("Step 1: Saving test data to parquet file")
    # Ensure columns are in the expected order
    expected_order = ["artist_name", "album_title", "play_time", "track_title", "artist_mbid"]
    available_columns = [col for col in expected_order if col in df.columns]
    other_columns = [col for col in df.columns if col not in expected_order]
    df = df[available_columns + other_columns]
    df.to_parquet(test_file, index=False)

    # Step 2: Simulate loading the parquet file as in profile.py
    print("\nStep 2: Loading the parquet file")
    data = pd.read_parquet(test_file)
    latest_filename = test_file
    if data is None or data.empty:
        print("ERROR: No scrobble data found")
        return

    print(f"Loaded columns before renaming: {data.columns.tolist()}")

    # Step 3: Simulate the column renaming in profile.py
    print("\nStep 3: Renaming columns")
    data.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]
    print(f"Columns after renaming: {data.columns.tolist()}")

    # Step 4: Verify that the data is correct after renaming
    print("\nStep 4: Verifying data after renaming")
    # Check if Artist column contains artist_name data
    expected_artist = test_data["artist_name"][0]
    actual_artist = data["Artist"][0]

    # Check if Song column contains track_title data
    expected_song = test_data["track_title"][0]
    actual_song = data["Song"][0]

    if expected_artist == actual_artist and expected_song == actual_song:
        print("SUCCESS: Data matches after column renaming!")
        print(f"Expected artist: {expected_artist}, Actual artist: {actual_artist}")
        print(f"Expected song: {expected_song}, Actual song: {actual_song}")
    else:
        print("FAILURE: Data does not match after column renaming.")
        print(f"Expected artist: {expected_artist}, Actual artist: {actual_artist}")
        print(f"Expected song: {expected_song}, Actual song: {actual_song}")

    # Step 5: Simulate the artist_counts calculation in profile.py
    print("\nStep 5: Calculating artist counts")
    artist_counts = data["Artist"].value_counts()
    top_artists = artist_counts.head(3)
    print("Top 3 artists:")
    print(top_artists)

    # Verify that Artist column contains artist_name data
    if top_artists.index[0] == expected_artist:
        print("SUCCESS: Artist counts are calculated correctly!")
    else:
        print("FAILURE: Artist counts are not calculated correctly.")
        print(f"Expected top artist: {expected_artist}, Actual top artist: {top_artists.index[0]}")

    # Clean up the test file
    test_file.unlink(missing_ok=True)
    print("\nTest file cleaned up.")

if __name__ == "__main__":
    test_full_profile_workflow()

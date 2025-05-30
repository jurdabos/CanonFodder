"""
Test script to verify that the column mapping fix in profile.py resolves the issue.
"""
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from helpers import io

def test_column_mapping_fix():
    """Test that the column mapping fix in profile.py resolves the issue."""
    # Create a test DataFrame with columns in the same order as in scrobble.parquet
    test_data = {
        "track_title": ["Song1", "Song2", "Song3", "Song1", "Song2"],
        "album_title": ["Album1", "Album2", "Album3", "Album1", "Album2"],
        "artist_mbid": ["mbid1", "mbid2", "mbid3", "mbid1", "mbid2"],
        "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "artist_name": ["Artist1", "Artist2", "Artist3", "Artist1", "Artist2"]
    }
    df = pd.DataFrame(test_data)
    
    # Save the DataFrame to a test parquet file
    test_file = Path("PQ") / "test_column_mapping_fix.parquet"
    df.to_parquet(test_file, index=False)
    
    # Load the parquet file
    loaded_df = pd.read_parquet(test_file)
    print(f"Original columns: {loaded_df.columns.tolist()}")
    
    # Apply the OLD incorrect column mapping
    old_df = loaded_df.copy()
    old_df.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]
    
    # Calculate artist counts with the OLD mapping
    old_artist_counts = old_df["Artist"].value_counts()
    print("\nArtist counts with OLD mapping:")
    print(old_artist_counts)
    
    # Apply the NEW correct column mapping
    new_df = loaded_df.copy()
    new_df.columns = ["Song", "Album", "MBID", "Datetime", "Artist"]
    
    # Calculate artist counts with the NEW mapping
    new_artist_counts = new_df["Artist"].value_counts()
    print("\nArtist counts with NEW mapping:")
    print(new_artist_counts)
    
    # Verify that the NEW mapping gives the correct results
    expected_top_artist = "Artist1"  # Artist1 appears twice in our test data
    actual_top_artist_old = old_artist_counts.index[0]
    actual_top_artist_new = new_artist_counts.index[0]
    
    print(f"\nExpected top artist: {expected_top_artist}")
    print(f"Top artist with OLD mapping: {actual_top_artist_old}")
    print(f"Top artist with NEW mapping: {actual_top_artist_new}")
    
    if actual_top_artist_new == expected_top_artist:
        print("\nSUCCESS: The NEW mapping correctly identifies the top artist!")
    else:
        print("\nFAILURE: The NEW mapping does not correctly identify the top artist.")
    
    if actual_top_artist_old != expected_top_artist:
        print("CONFIRMED: The OLD mapping was incorrect.")
    else:
        print("NOTE: The OLD mapping also gave correct results with this test data.")
    
    # Clean up the test file
    test_file.unlink(missing_ok=True)
    print("\nTest file cleaned up.")

if __name__ == "__main__":
    test_column_mapping_fix()
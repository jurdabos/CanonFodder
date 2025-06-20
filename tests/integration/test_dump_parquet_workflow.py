"""
Test script to verify that io.dump_parquet() maintains the correct column order
when called without arguments (as in profile.py).
"""
import sys
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
from helpers import io


def test_dump_parquet_workflow():
    """Test that io.dump_parquet() maintains the correct column order."""
    # Use a test-specific parquet file to avoid interference with existing data
    test_parquet = Path("PQ") / "test_workflow_scrobble.parquet"

    # Remove the test file if it exists
    if test_parquet.exists():
        test_parquet.unlink()
        print(f"Removed existing test file: {test_parquet}")

    try:
        # Create a test DataFrame with columns in the database schema order (minus id)
        # This simulates what load_scrobble_table_from_db_to_df would return
        test_data = {
            "artist_name": ["Artist1", "Artist2", "Artist3"],
            "artist_mbid": ["mbid1", "mbid2", "mbid3"],
            "album_title": ["Album1", "Album2", "Album3"],
            "track_title": ["Song1", "Song2", "Song3"],
            "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
        }
        df = pd.DataFrame(test_data)

        # Print the original column order
        print(f"Original columns: {df.columns.tolist()}")

        # Call io.dump_parquet() with this DataFrame and a custom output path
        # This simulates the call in profile.py but uses our test file
        io.dump_parquet(df, constant=False, stamp=datetime.now())

        # Rename the generated file to our test file
        latest_files = sorted(Path("PQ").glob("scrobbles_*.parquet"), reverse=True)
        if latest_files:
            latest_file = latest_files[0]
            if test_parquet.exists():
                test_parquet.unlink()
            latest_file.rename(test_parquet)
            print(f"Renamed {latest_file} to {test_parquet}")

        # Load the resulting parquet file and check the column order
        if test_parquet.exists():
            loaded_df = pd.read_parquet(test_parquet)
            print(f"Loaded columns from {test_parquet}: {loaded_df.columns.tolist()}")

            # Expected column order
            expected_order = ["artist_name", "album_title", "play_time", "track_title", "artist_mbid"]

            # Check if the columns are in the expected order
            actual_order = loaded_df.columns.tolist()
            if actual_order == expected_order:
                print(f"SUCCESS: Columns in {test_parquet.name} are in the expected order!")
            else:
                print(f"FAILURE: Columns in {test_parquet.name} are not in the expected order.")
                print(f"Expected: {expected_order}")
                print(f"Actual: {actual_order}")

            # Now simulate the profile.py workflow by renaming columns
            loaded_df.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]
            print(f"Renamed columns: {loaded_df.columns.tolist()}")

            # Calculate artist counts
            artist_counts = loaded_df["Artist"].value_counts()
            print("\nArtist counts:")
            print(artist_counts)

            # Verify that Artist column contains artist_name data
            expected_artist = test_data["artist_name"][0]
            actual_artist = loaded_df["Artist"][0]

            if expected_artist == actual_artist:
                print("\nSUCCESS: Artist column contains artist_name data!")
            else:
                print("\nFAILURE: Artist column does not contain artist_name data.")
                print(f"Expected: {expected_artist}, Actual: {actual_artist}")
        else:
            print(f"ERROR: {test_parquet} was not created.")

    finally:
        # Clean up the test file
        if test_parquet.exists():
            test_parquet.unlink()
            print(f"Cleaned up test file: {test_parquet}")

if __name__ == "__main__":
    test_dump_parquet_workflow()
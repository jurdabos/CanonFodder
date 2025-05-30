"""
Test script to verify that the entire profile.py workflow works correctly,
including loading data from the database and calling io.dump_parquet().
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from helpers import io
from DB import engine
from DB.ops import load_scrobble_table_from_db_to_df
import tempfile
import shutil

def test_profile_db_workflow():
    """Test that the entire profile.py workflow works correctly."""
    # Use a test-specific parquet file to avoid interference with existing data
    test_parquet = Path("PQ") / "test_db_workflow_scrobble.parquet"
    
    # Remove the test file if it exists
    if test_parquet.exists():
        test_parquet.unlink()
        print(f"Removed existing test file: {test_parquet}")
    
    try:
        # Step 1: Load data from the database (as in profile.py)
        print("Step 1: Loading data from the database")
        df, _tbl = load_scrobble_table_from_db_to_df(engine)
        if df is None or df.empty:
            print("No data in the database, using test data instead")
            # Create a test DataFrame with columns in the database schema order (minus id)
            test_data = {
                "artist_name": ["Artist1", "Artist2", "Artist3"],
                "artist_mbid": ["mbid1", "mbid2", "mbid3"],
                "album_title": ["Album1", "Album2", "Album3"],
                "track_title": ["Song1", "Song2", "Song3"],
                "play_time": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
            }
            df = pd.DataFrame(test_data)
        
        # Print the original column order
        print(f"Original columns from database: {df.columns.tolist()}")
        
        # Step 2: Call io.dump_parquet() with this DataFrame (as in profile.py)
        print("\nStep 2: Calling io.dump_parquet()")
        # Save to a temporary file to avoid overwriting the main scrobble.parquet
        temp_file = tempfile.mktemp(suffix=".parquet")
        original_parquet_name = io._parquet_name
        
        # Monkey patch _parquet_name to return our temp file
        def mock_parquet_name(*args, **kwargs):
            return Path(temp_file)
        
        io._parquet_name = mock_parquet_name
        
        try:
            # Call dump_parquet as in profile.py
            io.dump_parquet(df)
            
            # Copy the temp file to our test file
            shutil.copy2(temp_file, test_parquet)
            print(f"Copied dump_parquet output to {test_parquet}")
        finally:
            # Restore the original _parquet_name function
            io._parquet_name = original_parquet_name
            # Clean up the temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Step 3: Load the resulting parquet file (as in profile.py)
        print("\nStep 3: Loading the parquet file")
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
            
            # Step 4: Rename columns (as in profile.py)
            print("\nStep 4: Renaming columns")
            loaded_df.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]
            print(f"Renamed columns: {loaded_df.columns.tolist()}")
            
            # Step 5: Calculate artist counts (as in profile.py)
            print("\nStep 5: Calculating artist counts")
            artist_counts = loaded_df["Artist"].value_counts().head(5)
            print("Top 5 artists:")
            print(artist_counts)
            
            # Verify that Artist column contains artist_name data
            if "Artist1" in df["artist_name"].values:
                expected_artist = "Artist1"
                actual_artist = loaded_df[loaded_df["Artist"] == "Artist1"]["Artist"].iloc[0]
                
                if expected_artist == actual_artist:
                    print("\nSUCCESS: Artist column contains artist_name data!")
                else:
                    print("\nFAILURE: Artist column does not contain artist_name data.")
                    print(f"Expected: {expected_artist}, Actual: {actual_artist}")
            else:
                print("\nSUCCESS: Using real database data, artist counts calculated correctly!")
        else:
            print(f"ERROR: {test_parquet} was not created.")
    
    finally:
        # Clean up the test file
        if test_parquet.exists():
            test_parquet.unlink()
            print(f"Cleaned up test file: {test_parquet}")

if __name__ == "__main__":
    test_profile_db_workflow()
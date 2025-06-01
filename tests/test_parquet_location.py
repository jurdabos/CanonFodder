"""
Test script to verify that parquet files are saved in the correct location.
"""
from pathlib import Path
import pandas as pd
import sys

# Adding the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import the dump_parquet function from helpers.io
from helpers.io import dump_parquet, PQ_DIR


def     """Test that parquet files are saved in the correct location."""
    print(f"Project root: {project_root}")
    print(f"PQ_DIR: {PQ_DIR}")):
    """Test that parquet files are saved in the correct location."""
    print(f"Project root: {project_root}")
    print(f"PQ_DIR: {PQ_DIR}")


# Creating a simple DataFrame
    df = pd.DataFrame({
        "artist_name": ["Test Artist 1", "Test Artist 2"],
        "album_title": ["Test Album 1", "Test Album 2"],
        "track_title": ["Test Track 1", "Test Track 2"],
        "play_time": [pd.Timestamp.now(), pd.Timestamp.now()],
    })
    
    # Save the DataFrame to a parquet file
    test_file = "test_parquet_location.parquet"
    parquet_path = dump_parquet(df, constant=False)
    print(f"Parquet file saved to: {parquet_path}")
    
# Checking if the file was created in the correct location
    expected_dir = project_root / "PQ"
    actual_dir = parquet_path.parent
    
    print(f"Expected directory: {expected_dir}")
    print(f"Actual directory: {actual_dir}")
    
    if expected_dir.samefile(actual_dir):
        print("SUCCESS: Parquet file was saved in the correct location.")
    else:
        print("ERROR: Parquet file was saved in the wrong location.")
        print(f"Expected: {expected_dir}")
        print(f"Actual: {actual_dir}")
    
# Checking if a file was also created in dev/PQ
    dev_pq_dir = project_root / "dev" / "PQ"
    if dev_pq_dir.exists():
        dev_files = list(dev_pq_dir.glob("*.parquet"))
        if dev_files:
            print(f"WARNING: Found {len(dev_files)} parquet files in dev/PQ:")
            for file in dev_files:
                print(f"  - {file}")
        else:
            print("No parquet files found in dev/PQ.")
    else:
        print("dev/PQ directory does not exist.")

if __name__ == "__main__":
    test_parquet_location()
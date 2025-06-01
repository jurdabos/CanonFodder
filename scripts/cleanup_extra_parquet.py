"""
Cleanup script to remove the extra scrobble.parquet file in dev/PQ.
"""
import os
from pathlib import Path


def     """Remove the extra scrobble.parquet file in dev/PQ."""
# Geting the project root
    project_root = Path(__file__).resolve().parents[1]
    # Path to the extra parquet file
    extra_parquet = project_root / "dev" / "PQ" / "scrobble.parquet"
# Removing the file if it exists
    if extra_parquet.exists():
        os.remove(extra_parquet)
        print(f"Removed extra parquet file: {extra_parquet}")
    else:
        print(f"No extra parquet file found at: {extra_parquet}")):
    """Remove the extra scrobble.parquet file in dev/PQ."""
# Geting the project root
    project_root = Path(__file__).resolve().parents[1]
    # Path to the extra parquet file
    extra_parquet = project_root / "dev" / "PQ" / "scrobble.parquet"
# Removing the file if it exists
    if extra_parquet.exists():
        os.remove(extra_parquet)
        print(f"Removed extra parquet file: {extra_parquet}")
    else:
        print(f"No extra parquet file found at: {extra_parquet}")


if __name__ == "__main__":
    cleanup_extra_parquet()
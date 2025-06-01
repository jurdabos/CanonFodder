# Parquet Location Fix

## Issue
An extra `scrobble.parquet` file was being created in the `dev/PQ` directory, while the documentation requires that the Eval DB (parquet files) should be in the regular `project_root/PQ` directory.

## Root Cause
The issue was in `helpers/io.py`, where `PQ_DIR` was defined as:

```python
PQ_DIR = Path.cwd() / "PQ"
```

This meant that `PQ_DIR` was set to the current working directory plus "PQ", not the project root directory plus "PQ". When running scripts from the `dev` directory, the current working directory was `dev/`, so `PQ_DIR` became `dev/PQ`.

In contrast, in `dev/profile.py`, `PQ_DIR` was defined as:

```python
PQ_DIR = PROJECT_ROOT / "PQ"
```

which correctly points to the project root directory plus "PQ".

This inconsistency caused duplicate parquet files - one in `dev/PQ` and one in the main `PQ` directory.

## Solution
The solution was to modify `helpers/io.py` to use the project root directory instead of the current working directory for `PQ_DIR`. A `find_project_root()` function was added, similar to the one in `dev/profile.py`, which looks for JSON and PQ directories to determine the project root:

```python
def find_project_root():
    """Find the project root by looking for JSON and PQ directories."""
    if "__file__" in globals():
        # Try the standard approach first
        candidate = Path(__file__).resolve().parents[1]
        if (candidate / "JSON").exists() and (candidate / "PQ").exists():
            return candidate
    # If that fails, try the current directory and its parent
    current = Path.cwd()
    if (current / "JSON").exists() and (current / "PQ").exists():
        return current
    if (current.parent / "JSON").exists() and (current.parent / "PQ").exists():
        return current.parent
    # If all else fails, use an absolute path
    return Path(r"C:\Users\jurda\PycharmProjects\CanonFodder")

PROJECT_ROOT = find_project_root()
PQ_DIR = PROJECT_ROOT / "PQ"
```

This ensures that parquet files are always saved in the main `project_root/PQ` directory, regardless of the current working directory.

## Testing
A test script was created to verify that parquet files are saved in the correct location:

```python
python tests\test_parquet_location.py
```

The test confirmed that parquet files are now saved in the correct location.

## Cleanup
The existing `scrobble.parquet` file in `dev/PQ` can be safely removed:

```python
import os
from pathlib import Path

# Get the project root
project_root = Path(__file__).resolve().parents[1]
# Path to the extra parquet file
extra_parquet = project_root / "dev" / "PQ" / "scrobble.parquet"
# Remove the file if it exists
if extra_parquet.exists():
    os.remove(extra_parquet)
    print(f"Removed extra parquet file: {extra_parquet}")
else:
    print(f"No extra parquet file found at: {extra_parquet}")
```

This code can be run as a standalone script or added to an existing script to clean up the extra file.
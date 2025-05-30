# Parquet File Structure Fix

This directory contains scripts to check and fix the structure of parquet files in the project.

## Issue Description

The `scrobble.parquet` file had an overflow of columns. According to the project guidelines in `docs/CanonFodder.md`, the file should only contain the following columns:
- artist_name
- album_title
- track_title
- play_time
- artist_mbid

However, the file contained additional columns: 'Album', 'uts', 'Song', 'Artist'.

## Solution

Two scripts were created to address this issue:

1. `check_parquet.py`: This script checks the current structure of the `scrobble.parquet` file by printing its columns and the first few rows of data.

2. `fix_parquet.py`: This script fixes the structure of the `scrobble.parquet` file by keeping only the required columns and saving it back to the same file.

## Usage

To check the structure of the parquet file:
```
python scripts\check_parquet.py
```

To fix the structure of the parquet file:
```
python scripts\fix_parquet.py
```

## Results

After running the fix script, the `scrobble.parquet` file now has the correct structure with only the 5 required columns:
- artist_name
- album_title
- track_title
- play_time
- artist_mbid

This ensures that the file adheres to the project guidelines and maintains consistency in the data structure.

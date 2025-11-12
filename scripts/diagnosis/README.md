# Diagnosis Scripts

This folder contains diagnostic and repair scripts for data quality issues in the CanonFodder database.

## Scripts

### investigate_empty_artists.py
Investigates records with empty or whitespace artist names in the database. Provides detailed analysis including:
- Count of affected records
- Date range of problematic data
- Sample tracks with empty artists
- Hexadecimal representation of empty values

### fix_empty_artists.py
Analyzes empty artist scrobbles by album patterns to help identify potential artist information. Groups empty artist records by album and looks for patterns that might help recover the missing artist data.

### fix_unknown_artists.py
Applies the fix for empty artist names by replacing them with `[Unknown Artist]` placeholder. This prevents empty strings from appearing as the top artist in statistics while preserving the data for potential future recovery.

## Usage

All scripts should be run from the project root using uv:

```bash
# Investigate the problem
uv run python scripts/diagnosis/investigate_empty_artists.py

# Analyze patterns
uv run python scripts/diagnosis/fix_empty_artists.py

# Apply the fix
uv run python scripts/diagnosis/fix_unknown_artists.py
```

## Related Documentation

See `docs/UNKNOWN_ARTIST_FIX.md` for complete documentation of the unknown artist issue and its resolution.
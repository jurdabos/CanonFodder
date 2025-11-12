# Unknown Artist Fix Documentation

## Problem Identified
On 2025-10-08, discovered that 7,579 scrobbles in the database had completely empty artist names (`''`), causing them to incorrectly appear as the #1 top artist in statistics queries.

## Root Cause
- These entries appeared between May 29 and September 6, 2025
- Likely from a batch import where artist information was lost or not properly parsed
- Affected legitimate albums including classical music, soundtracks, and various compilations

## Solution Implemented

### 1. Database Update
- Updated all empty artist names to `[Unknown Artist]` placeholder
- This preserves the data while clearly marking it as incomplete
- SQL executed: `UPDATE scrobble SET artist_name = '[Unknown Artist]' WHERE artist_name = '' OR artist_name IS NULL`

### 2. Query Updates
Created helper queries and functions to properly exclude unknown artists from statistics:
- Added `WHERE artist_name NOT LIKE '[Unknown%'` to top artist queries
- Created `helpers/sql_queries_unknown_artists.sql` with template queries
- Added `helpers/stats_unknown_handler.py` module with functions:
  - `get_top_artists()` - Get top artists with option to exclude unknowns
  - `get_data_quality_stats()` - Monitor percentage of unknown artists
  - `get_unknown_albums()` - Identify albums that could be fixed
  - `fix_known_album()` - Update specific albums when artist is identified

## Current Status

### Data Quality Metrics
- **Total scrobbles:** 307,961
- **Unknown artist scrobbles:** 7,585 (2.46%)
- **Unique known artists:** 19,020

### Top Artists (After Fix)
1. Autechre - 5,055 plays
2. Radiohead - 4,398 plays
3. Secret Chiefs 3 - 4,235 plays
4. Bohren & der Club of Gore - 4,027 plays
5. John Zorn - 2,873 plays

### Albums Needing Artist Identification
Some notable albums with unknown artists that could be identified:
- J. S. Bach: The Well-Tempered Clavier, Book II (96 plays)
- Murmur of the Bath Spirits (90 plays)
- Horizons: French Mélodies (78 plays)
- Pachinko: Season 2 Soundtrack (70 plays)
- Villa-Lobos: Complete Works for Solo Guitar (50 plays)

## Future Improvements

### To Fix Individual Albums
When you identify the correct artist for an album, use:
```python
from helpers.stats_unknown_handler import fix_known_album
rows_updated = fix_known_album("Album Title", "Correct Artist Name")
```

### To Monitor Data Quality
```python
from helpers.stats_unknown_handler import print_data_quality_report
print_data_quality_report()
```

### Prevention
- Add validation in data import pipeline to reject or flag empty artist names
- Consider implementing a data quality check before bulk imports
- Log import sources to track down origin of bad data

## Files Modified/Created
- `fix_unknown_artists.py` - Script that performed the initial fix
- `helpers/sql_queries_unknown_artists.sql` - SQL query templates
- `helpers/stats_unknown_handler.py` - Python module for handling unknown artists
- `investigate_empty_artists.py` - Investigation script
- `fix_empty_artists.py` - Analysis script

## Related Issues
- The original issue was likely caused by a problematic import between May-September 2025
- Consider investigating the import logs from that period if available
- Some legitimate bands have "Unknown" in their name (e.g., Unknown Mortal Orchestra) - these are NOT affected by this fix
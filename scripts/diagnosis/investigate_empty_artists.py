"""
Investigate empty or whitespace artist names in the scrobble table.
"""

from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

# Load environment variables
load_dotenv()

# Get database URL
db_url = os.getenv("DB_URL")
if not db_url:
    print("Error: DB_URL not found in environment variables")
    exit(1)

# Create engine
engine = create_engine(db_url)

print("Investigating empty/whitespace artist names in scrobbles...")
print("=" * 60)

with engine.connect() as conn:
    # Check for empty/whitespace artist names
    query1 = text("""
        SELECT 
            LENGTH(artist_name) as name_length,
            HEX(artist_name) as hex_value,
            COUNT(*) as count
        FROM scrobble
        WHERE artist_name = '' 
           OR artist_name IS NULL 
           OR artist_name REGEXP '^[[:space:]]+$'
           OR LENGTH(TRIM(artist_name)) = 0
        GROUP BY artist_name
        ORDER BY count DESC
        LIMIT 5
    """)
    
    result = conn.execute(query1)
    rows = result.fetchall()
    
    if rows:
        print("Found empty/whitespace artist names:")
        for row in rows:
            print(f"  Length: {row[0]}, Hex: {row[1]}, Count: {row[2]}")
    else:
        print("No empty/whitespace artist names found with basic query.")
    
    print("\n" + "-" * 60)
    
    # Check the top artist that appears empty
    query2 = text("""
        SELECT 
            artist_name,
            LENGTH(artist_name) as name_length,
            HEX(artist_name) as hex_value,
            COUNT(*) as play_count
        FROM scrobble
        GROUP BY artist_name
        ORDER BY play_count DESC
        LIMIT 1
    """)
    
    result = conn.execute(query2)
    row = result.fetchone()
    
    if row:
        print(f"Top artist by play count:")
        print(f"  Artist: '{row[0]}'")
        print(f"  Length: {row[1]} characters")
        print(f"  Hex representation: {row[2]}")
        print(f"  Play count: {row[3]}")
    
    print("\n" + "-" * 60)
    
    # Sample some tracks with this empty artist
    query3 = text("""
        SELECT 
            track_title,
            album_title,
            UNIX_TIMESTAMP(play_time) as timestamp_value,
            artist_mbid
        FROM scrobble
        WHERE artist_name = :artist
        ORDER BY play_time DESC
        LIMIT 10
    """)
    
    if row and row[3] > 1000:  # If we have a problematic artist with many plays
        result = conn.execute(query3, {"artist": row[0]})
        tracks = result.fetchall()
        
        print(f"Sample tracks from the problematic artist:")
        for track in tracks:
            timestamp = datetime.fromtimestamp(track[2], timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Track: {track[0]}")
            print(f"  Album: {track[1]}")
            print(f"  Time: {timestamp}")
            print(f"  MBID: {track[3]}")
            print("  ---")
    
    print("\n" + "-" * 60)
    
    # Check when these empty artists started appearing
    query4 = text("""
        SELECT 
            DATE(MIN(play_time)) as first_date,
            DATE(MAX(play_time)) as last_date,
            COUNT(*) as total_count
        FROM scrobble
        WHERE artist_name = :artist
    """)
    
    if row:
        result = conn.execute(query4, {"artist": row[0]})
        date_info = result.fetchone()
        
        if date_info:
            print(f"Date range for problematic artist:")
            print(f"  First appearance: {date_info[0]}")
            print(f"  Last appearance: {date_info[1]}")
            print(f"  Total scrobbles: {date_info[2]}")

print("\n" + "=" * 60)
print("Investigation complete.")
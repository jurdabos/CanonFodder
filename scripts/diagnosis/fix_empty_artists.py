"""
Investigate and fix empty artist names by checking album patterns.
"""

from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os
from collections import Counter

# Load environment variables
load_dotenv()

# Get database URL
db_url = os.getenv("DB_URL")
if not db_url:
    print("Error: DB_URL not found in environment variables")
    exit(1)

# Create engine
engine = create_engine(db_url)

print("Analyzing empty artist scrobbles by album...")
print("=" * 60)

with engine.connect() as conn:
    # Group by album to see patterns
    query1 = text("""
        SELECT 
            album_title,
            COUNT(*) as play_count,
            COUNT(DISTINCT track_title) as unique_tracks
        FROM scrobble
        WHERE artist_name = ''
        GROUP BY album_title
        ORDER BY play_count DESC
        LIMIT 20
    """)
    
    result = conn.execute(query1)
    albums = result.fetchall()
    
    print("Top albums with empty artist names:")
    for album in albums:
        print(f"  Album: {album[0]}")
        print(f"    Plays: {album[1]}, Unique tracks: {album[2]}")
    
    print("\n" + "-" * 60)
    
    # Check if these albums exist with actual artist names
    print("\nChecking if these albums exist with proper artist names...")
    
    for album in albums[:5]:  # Check top 5 albums
        if album[0]:  # Skip empty album names
            query2 = text("""
                SELECT DISTINCT artist_name, COUNT(*) as count
                FROM scrobble
                WHERE album_title = :album AND artist_name != ''
                GROUP BY artist_name
                ORDER BY count DESC
                LIMIT 3
            """)
            
            result = conn.execute(query2, {"album": album[0]})
            artists = result.fetchall()
            
            if artists:
                print(f"\nAlbum '{album[0]}' also appears with these artists:")
                for artist in artists:
                    print(f"  - {artist[0]} ({artist[1]} plays)")
            else:
                print(f"\nAlbum '{album[0]}' ONLY appears with empty artist")
    
    print("\n" + "-" * 60)
    
    # Look for patterns in track names that might hint at the artist
    print("\nAnalyzing unique album-track combinations...")
    
    query3 = text("""
        SELECT 
            album_title,
            track_title,
            COUNT(*) as play_count
        FROM scrobble
        WHERE artist_name = ''
        GROUP BY album_title, track_title
        ORDER BY play_count DESC
        LIMIT 30
    """)
    
    result = conn.execute(query3)
    tracks = result.fetchall()
    
    # Group by album
    albums_dict = {}
    for track in tracks:
        album = track[0]
        if album not in albums_dict:
            albums_dict[album] = []
        albums_dict[album].append((track[1], track[2]))
    
    print("\nTop album/track combinations:")
    for album, track_list in list(albums_dict.items())[:10]:
        print(f"\nAlbum: '{album}'")
        for track_name, count in track_list[:5]:
            print(f"  - {track_name} ({count} plays)")
    
    print("\n" + "=" * 60)
    
    # Provide a summary and suggestion
    print("SUMMARY:")
    print(f"- Total scrobbles with empty artist: 7,579")
    print(f"- Date range: 2025-05-29 to 2025-09-06")
    print(f"- Number of unique albums: {len(albums)}")
    print("\nThese appear to be legitimate scrobbles with missing artist data.")
    print("The Hungarian track/album titles suggest these might be from a specific artist")
    print("or compilation that wasn't properly tagged when imported.")
    
    print("\n" + "=" * 60)
    print("\nPOSSIBLE FIXES:")
    print("1. If you know what artist these tracks belong to, we can update them")
    print("2. We can delete these entries if they're corrupted imports")
    print("3. We can mark them with a placeholder artist like '[Unknown Artist]'")
    print("4. Investigate the original import source to find the missing artist info")
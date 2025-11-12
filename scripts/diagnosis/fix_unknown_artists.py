"""
Fix empty artist names by replacing them with '[Unknown Artist]' placeholder.
This prevents them from appearing as the top artist while preserving the data.
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

print("Fixing empty artist names in the database...")
print("=" * 60)

with engine.begin() as conn:  # Use begin() for automatic commit/rollback
    # First, check how many records we're about to update
    check_query = text("""
        SELECT COUNT(*) as count
        FROM scrobble
        WHERE artist_name = '' OR artist_name IS NULL
    """)
    
    result = conn.execute(check_query)
    count = result.fetchone()[0]
    
    print(f"Found {count} scrobbles with empty/null artist names")
    
    if count > 0:
        print("\nUpdating empty artist names to '[Unknown Artist]'...")
        
        # Update empty and NULL artist names
        update_query = text("""
            UPDATE scrobble 
            SET artist_name = '[Unknown Artist]'
            WHERE artist_name = '' OR artist_name IS NULL
        """)
        
        result = conn.execute(update_query)
        rows_updated = result.rowcount
        
        print(f"Successfully updated {rows_updated} records")
        
        # Verify the update
        verify_query = text("""
            SELECT artist_name, COUNT(*) as play_count
            FROM scrobble
            WHERE artist_name LIKE '%Unknown%'
            GROUP BY artist_name
        """)
        
        result = conn.execute(verify_query)
        unknown_artists = result.fetchall()
        
        if unknown_artists:
            print("\nUnknown artist entries after update:")
            for artist in unknown_artists:
                print(f"  - {artist[0]}: {artist[1]} plays")
    else:
        print("No empty artist names found - database is already clean!")
    
    print("\n" + "-" * 60)
    print("\nVerifying top artists (excluding unknowns)...")
    
    # Check top artists excluding unknown
    top_artists_query = text("""
        SELECT 
            ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
            artist_name,
            COUNT(*) AS play_count
        FROM scrobble
        WHERE artist_name NOT LIKE '[Unknown%'
        GROUP BY artist_name
        ORDER BY play_count DESC
        LIMIT 10
    """)
    
    result = conn.execute(top_artists_query)
    top_artists = result.fetchall()
    
    print("\nTop 10 artists (excluding unknown):")
    print("-" * 50)
    print(f"{'Rank':<6} {'Artist':<30} {'Plays':<10}")
    print("-" * 50)
    for artist in top_artists:
        rank = artist[0]
        name = artist[1][:28] + ".." if len(artist[1]) > 30 else artist[1]
        plays = artist[2]
        print(f"{rank:<6} {name:<30} {plays:<10}")
    
    print("\n" + "=" * 60)
    print("Fix completed successfully!")
    print("\nNOTE: You may want to update your queries and views to:")
    print("1. Exclude '[Unknown Artist]' from top artist charts")
    print("2. Handle unknown artists separately in analytics")
    print("3. Consider investigating the source of these unknown entries")
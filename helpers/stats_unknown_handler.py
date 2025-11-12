"""
Helper functions for handling unknown artists in statistics.
This module provides functions to properly exclude unknown artists
from statistics and analytics queries.
"""

import pandas as pd
from sqlalchemy import text, create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create session using DB_URL from environment
def get_session():
    db_url = os.getenv("DB_URL")
    if not db_url:
        # Fallback to SessionLocal if DB_URL not set
        from DB import SessionLocal
        return SessionLocal()
    engine = create_engine(db_url)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    return Session()


def get_top_artists(limit=20, exclude_unknown=True):
    """
    Get top artists by play count, optionally excluding unknown artists.
    
    Parameters:
    -----------
    limit : int
        Number of top artists to return
    exclude_unknown : bool
        If True, excludes artists marked as [Unknown Artist]
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: rank, artist_name, play_count
    """
    with get_session() as session:
        if exclude_unknown:
            query = text("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
                    artist_name,
                    COUNT(*) AS play_count
                FROM scrobble
                WHERE artist_name NOT LIKE '[Unknown%'
                GROUP BY artist_name
                ORDER BY play_count DESC
                LIMIT :limit
            """)
        else:
            query = text("""
                SELECT 
                    ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
                    artist_name,
                    COUNT(*) AS play_count
                FROM scrobble
                GROUP BY artist_name
                ORDER BY play_count DESC
                LIMIT :limit
            """)
        
        result = session.execute(query, {"limit": limit})
        rows = result.fetchall()
        
    return pd.DataFrame(rows, columns=['rank', 'artist_name', 'play_count'])


def get_data_quality_stats():
    """
    Get statistics about data quality including unknown artist percentage.
    
    Returns:
    --------
    dict
        Dictionary with data quality statistics
    """
    with get_session() as session:
        query = text("""
            SELECT 
                COUNT(*) AS total_scrobbles,
                COUNT(CASE WHEN artist_name LIKE '[Unknown%' THEN 1 END) AS unknown_scrobbles,
                COUNT(DISTINCT artist_name) AS unique_artists,
                COUNT(DISTINCT CASE WHEN artist_name NOT LIKE '[Unknown%' THEN artist_name END) AS unique_known_artists,
                COUNT(DISTINCT album_title) AS unique_albums,
                COUNT(DISTINCT track_title) AS unique_tracks
            FROM scrobble
        """)
        
        result = session.execute(query)
        row = result.fetchone()
        
        if row:
            total = row[0]
            unknown = row[1]
            unknown_pct = (unknown / total * 100) if total > 0 else 0
            
            return {
                'total_scrobbles': total,
                'unknown_scrobbles': unknown,
                'unknown_percentage': round(unknown_pct, 2),
                'unique_artists': row[2],
                'unique_known_artists': row[3],
                'unique_albums': row[4],
                'unique_tracks': row[5]
            }
    
    return {}


def get_unknown_albums(min_plays=10):
    """
    Get albums with unknown artists that could potentially be identified.
    
    Parameters:
    -----------
    min_plays : int
        Minimum play count to include an album
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with albums that have unknown artists
    """
    with get_session() as session:
        query = text("""
            SELECT 
                album_title,
                COUNT(DISTINCT track_title) AS track_count,
                COUNT(*) AS play_count,
                MIN(play_time) AS first_play,
                MAX(play_time) AS last_play
            FROM scrobble
            WHERE artist_name = '[Unknown Artist]'
              AND album_title IS NOT NULL 
              AND album_title != ''
            GROUP BY album_title
            HAVING COUNT(*) >= :min_plays
            ORDER BY play_count DESC
        """)
        
        result = session.execute(query, {"min_plays": min_plays})
        rows = result.fetchall()
        
    return pd.DataFrame(
        rows, 
        columns=['album_title', 'track_count', 'play_count', 'first_play', 'last_play']
    )


def fix_known_album(album_title, correct_artist_name):
    """
    Fix an album with unknown artist by updating it to the correct artist name.
    
    Parameters:
    -----------
    album_title : str
        The album title to fix
    correct_artist_name : str
        The correct artist name for this album
    
    Returns:
    --------
    int
        Number of rows updated
    """
    with get_session() as session:
        query = text("""
            UPDATE scrobble 
            SET artist_name = :artist
            WHERE artist_name = '[Unknown Artist]'
              AND album_title = :album
        """)
        
        result = session.execute(
            query, 
            {"artist": correct_artist_name, "album": album_title}
        )
        session.commit()
        
        return result.rowcount


def print_data_quality_report():
    """
    Print a comprehensive data quality report including unknown artist handling.
    """
    stats = get_data_quality_stats()
    
    print("=" * 60)
    print("DATA QUALITY REPORT - Unknown Artist Handling")
    print("=" * 60)
    print(f"Total scrobbles: {stats.get('total_scrobbles', 0):,}")
    print(f"Unknown artist scrobbles: {stats.get('unknown_scrobbles', 0):,}")
    print(f"Unknown percentage: {stats.get('unknown_percentage', 0):.2f}%")
    print("-" * 60)
    print(f"Unique artists (total): {stats.get('unique_artists', 0):,}")
    print(f"Unique known artists: {stats.get('unique_known_artists', 0):,}")
    print(f"Unique albums: {stats.get('unique_albums', 0):,}")
    print(f"Unique tracks: {stats.get('unique_tracks', 0):,}")
    print("=" * 60)
    
    # Show top artists excluding unknowns
    print("\nTOP 10 ARTISTS (excluding unknowns):")
    print("-" * 60)
    top_artists = get_top_artists(limit=10, exclude_unknown=True)
    for _, row in top_artists.iterrows():
        print(f"{int(row['rank']):3}. {row['artist_name'][:40]:<40} {int(row['play_count']):>6} plays")
    
    # Show some unknown albums that might be identifiable
    print("\n" + "=" * 60)
    print("IDENTIFIABLE ALBUMS WITH UNKNOWN ARTISTS:")
    print("-" * 60)
    unknown_albums = get_unknown_albums(min_plays=20)
    if not unknown_albums.empty:
        for _, row in unknown_albums.head(10).iterrows():
            print(f"Album: {row['album_title'][:50]}")
            print(f"  Plays: {int(row['play_count'])}, Tracks: {int(row['track_count'])}")
    else:
        print("No albums with unknown artists found (min 20 plays)")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run the data quality report when script is executed directly
    print_data_quality_report()
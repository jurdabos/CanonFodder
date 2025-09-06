#!/usr/bin/env python
"""Quick status check for CanonFodder pipeline."""

from dotenv import load_dotenv
load_dotenv()

from DB import SessionLocal
from DB.models import ArtistInfo, Scrobble
from datetime import datetime, UTC

def check_status():
    """Check the current status of the pipeline."""
    session = SessionLocal()
    
    # Check artist info
    artist_count = session.query(ArtistInfo).count()
    print(f"Total artist_info records: {artist_count}")
    
    # Check scrobbles
    scrobble_count = session.query(Scrobble).count()
    print(f"Total scrobbles: {scrobble_count}")
    
    # Get recent artists
    recent_artists = session.query(ArtistInfo).order_by(ArtistInfo.id.desc()).limit(5).all()
    if recent_artists:
        print("\nLast 5 artists added:")
        for artist in recent_artists:
            mbid_short = artist.mbid[:8] + "..." if artist.mbid else "None"
            print(f"  - {artist.artist_name} (MBID: {mbid_short})")
    
    # Get recent scrobbles  
    recent_scrobbles = session.query(Scrobble).order_by(Scrobble.play_time.desc()).limit(5).all()
    if recent_scrobbles:
        print("\nLast 5 scrobbles:")
        for scrobble in recent_scrobbles:
            print(f"  - {scrobble.artist_name} - {scrobble.track_title} ({scrobble.play_time})")
    
    session.close()
    print("\n✓ Database connection successful")

if __name__ == "__main__":
    check_status()

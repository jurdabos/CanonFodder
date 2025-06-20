"""
Data cleaning functionality for CanonFodder.

This module provides functions for cleaning and maintaining the database,
including removing duplicates and orphaned records.
"""

from __future__ import annotations
from collections import Counter
from typing import Tuple

from sqlalchemy import select, func

from DB import SessionLocal
from DB.models import ArtistInfo, Scrobble


def clean_artist_info_table() -> Tuple[int, int]:
    """
    Clean up the ArtistInfo table by removing duplicates and unnecessary entries.
    This function:
    1. Identifies duplicate artist names
    2. Keeps only the most complete record for each artist
    3. Removes any orphaned artists (not referenced in the Scrobble table)

    Returns:
        tuple: (cleaned_count, total_count) - number of records cleaned and total remaining
    """
    print("Starting ArtistInfo table cleanup...")

    with SessionLocal() as session:
        # Get all artists from the ArtistInfo table
        artists_query = select(ArtistInfo)
        artists = session.execute(artists_query).scalars().all()

        if not artists:
            print("No artists found in the ArtistInfo table.")
            return 0, 0

        total_before = len(artists)
        print(f"Found {total_before} artists in the ArtistInfo table.")

        # Find artists with duplicate names
        artist_names = [a.artist_name for a in artists]
        name_counts = Counter(artist_names)
        duplicates = {name: count for name, count in name_counts.items() if count > 1}

        if duplicates:
            print(f"Found {len(duplicates)} artist names with duplicates.")

            # For each duplicate, keep only the most complete record
            for name, count in duplicates.items():
                dupes = session.execute(
                    select(ArtistInfo).where(ArtistInfo.artist_name == name)
                ).scalars().all()

                # Sort by completeness (non-null fields)
                def completeness_score(artist_record):
                    score = 0
                    if artist_record.mbid:
                        score += 2  # MBID is most important
                    if artist_record.country:
                        score += 1
                    if artist_record.disambiguation_comment:
                        score += 1
                    if artist_record.aliases:
                        score += 1
                    return score

                sorted_dupes = sorted(dupes, key=completeness_score, reverse=True)

                # Keep the most complete record, delete the rest
                for dupe in sorted_dupes[1:]:
                    session.delete(dupe)

            session.commit()

        # Find orphaned artists (not referenced in Scrobble table)
        # This is optional and might be slow on large databases

        # Get all artist names from the Scrobble table
        scrobble_artists = session.execute(
            select(Scrobble.artist_name).distinct()
        ).scalars().all()

        scrobble_artist_set = set(scrobble_artists)

        # Find artists in ArtistInfo that aren't in Scrobble
        orphaned = []
        for artist in artists:
            if artist.artist_name not in scrobble_artist_set:
                orphaned.append(artist)

        if orphaned:
            print(f"Found {len(orphaned)} orphaned artists (not referenced in Scrobble table).")
            # Uncomment to actually delete orphaned artists
            # for artist in orphaned:
            #     session.delete(artist)
            # session.commit()

        # Get final count
        final_count = session.execute(select(func.count()).select_from(ArtistInfo)).scalar_one()
        cleaned_count = total_before - final_count

        print(f"Cleanup complete. Removed {cleaned_count} records, {final_count} artists remain.")
        return cleaned_count, final_count

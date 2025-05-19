# helpers/aliases.py
"""
Sync canonised variants → MusicBrainz aliases.
Usage (CLI):
    python -m helpers.aliases  [--dry-run]
To be called programmatically from main.py’s menu later.
"""
from __future__ import annotations

import logging
from typing import Sequence

from sqlalchemy import select

import mbAPI
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized, ArtistInfo

log = logging.getLogger(__name__)


def _variants(record: ArtistVariantsCanonized) -> Sequence[str]:
    """Split the stored text into the individual variant strings."""
    sep = "{"
    return [v.strip() for v in record.artist_variants_text.split(sep) if v.strip()]


def push_aliases(dry_run: bool = False) -> None:
    """Post every missing variant as an alias on MusicBrainz."""
    with SessionLocal() as s:
        rows = s.scalars(
            select(ArtistVariantsCanonized)
            .where(ArtistVariantsCanonized.to_link.is_(True))
        ).all()
    for row in rows:
        variants = [v for v in _variants(row) if v != row.canonical_name]
        if not variants:
            continue
        # try MBID from ArtistInfo first – fallback to search
        mbid = (
            SessionLocal()
            .scalar(select(ArtistInfo.mbid)
                    .where(ArtistInfo.artist_name == row.canonical_name))
        )
        if not mbid:
            res = mbAPI.search_artist(row.canonical_name, limit=1)
            mbid = res[0]["id"] if res else None
        if not mbid:
            log.warning("❌  No MBID for “%s” – skipped", row.canonical_name)
            continue
        current = set(mbAPI.get_aliases(mbid))
        missing = [v for v in variants if v not in current]
        if not missing:
            continue
        for alias in missing:
            if dry_run:
                log.info("[dry-run] would add alias “%s” → %s", alias, row.canonical_name)
                continue
            try:
                mbAPI.add_alias(mbid, alias)
                log.info("✓ added alias “%s” for %s", alias, row.canonical_name)
            except Exception as exc:                          # noqa: BLE001
                log.warning("⚠ failed to add “%s”: %s", alias, exc)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Push canonised variants as MB aliases")
    parser.add_argument("--dry-run", action="store_true", help="don’t POST, just list")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    push_aliases(dry_run=args.dry_run)

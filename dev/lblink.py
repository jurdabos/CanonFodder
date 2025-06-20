"""
Very small ListenBrainz helper for CanonFodder development.
Usage (CLI):
    python dev\\lblink.py --user <MB_USER_ID> [--count 10]
    python dev\\lblink.py --user <MB_USER_ID> --export <OUTPUT_PARQUET> [--count 1000]
Typical use (library):
    from dev.lblink import LBClient
    lb = LBClient()                          # token is read from env/.env
    listens = lb.get_listens("iliekcomputers", count=5)
    for listen in listens:
        print(listen["track_metadata"]["track_name"])
"""
from __future__ import annotations
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
try:
    import pylistenbrainz  # type: ignore
except ModuleNotFoundError:  # ↳ keep running without it
    pylistenbrainz = None  # noqa: N816

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
API_ROOT = "https://api.listenbrainz.org/1"


# Token resolution order:
#   1. LB_TOKEN environment variable
#   2. .env file in project root (dotenv format, line: TOKEN=****)
#   3. fall back to unauthenticated (discouraged, but still works)
def _load_token() -> Optional[str]:
    token = os.getenv("LB_TOKEN")
    if token:
        return token
    dotenv = Path.cwd().parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("LB_TOKEN="):
                return line.partition("=")[2].strip()
    return None


LB_TOKEN: Optional[str] = _load_token()
AUTH_HEADER = {"Authorization": f"Token {LB_TOKEN}"} if LB_TOKEN else {}


# --------------------------------------------------------------------------- #
# Core implementation – plain HTTP (requests)
# --------------------------------------------------------------------------- #
class _RequestsBackend:
    """Internal backend – only minimal functionality that we need."""

    def __init__(self) -> None:
        self.session = requests.Session()
        if AUTH_HEADER:
            self.session.headers.update(AUTH_HEADER)

    # --------------- public-ish helpers ---------------- #
    def get_listens(
            self,
            username: str,
            *,
            min_ts: int | None = None,
            max_ts: int | None = None,
            count: int | None = None,
    ) -> List[Dict[str, Any]]:
        params = {
            "min_ts": min_ts,
            "max_ts": max_ts,
            "count": count,
        }
        # remove None values so that the URL is clean
        params = {k: v for k, v in params.items() if v is not None}
        url = f"{API_ROOT}/user/{username}/listens"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()["payload"]["listens"]

    def lookup_metadata(
            self,
            track_name: str,
            artist_name: str,
            *,
            incs: str | None = None,
    ) -> Dict[str, Any]:
        params = {
            "recording_name": track_name,
            "artist_name": artist_name,
        }
        if incs:
            params["metadata"] = "true"
            params["incs"] = incs
        url = f"{API_ROOT}/metadata/lookup/"
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def submit_listens(self, listen_doc: Dict[str, Any]) -> None:
        """Submit *one* listen document.  Build the JSON yourself."""
        if not LB_TOKEN:
            raise RuntimeError("Submit requires an auth TOKEN – none configured.")
        url = f"{API_ROOT}/submit-listens"
        resp = self.session.post(url, json=listen_doc, timeout=15)
        resp.raise_for_status()
        logger.info("✅ Listen submitted (status %s)", resp.status_code)


# --------------------------------------------------------------------------- #
# Public convenience wrapper                                                #
# This is what external code should instantiate.
# --------------------------------------------------------------------------- #
class LBClient:
    """Tiny façade"""

    def __init__(self) -> None:
        if pylistenbrainz is not None:
            try:
                # Recent versions don’t accept any args in the ctor.
                self._client = pylistenbrainz.ListenBrainz()
                self._backend = "pylb"
                logger.debug("Using `pylistenbrainz` backend.")
            except TypeError as exc:  # Signature mismatch? → fall back.
                logger.warning(
                    "pylistenbrainz instantiation failed (%s) – "
                    "falling back to plain HTTP backend.",
                    exc,
                )
                self._client = _RequestsBackend()
                self._backend = "http"
        else:
            logger.debug("Using fallback `requests` backend.")
            self._client = _RequestsBackend()
            self._backend = "http"

    # ---------- public API for CanonFodder code ---------------- #
    def get_listens(
            self,
            username: str,
            *,
            min_ts: int | None = None,
            max_ts: int | None = None,
            count: int | None = None,
    ) -> List[Dict[str, Any]]:
        if self._backend == "pylb":
            listens_raw = self._client.get_listens(
                username=username,
                min_ts=min_ts,
                max_ts=max_ts,
                count=count,
            )

            # Convert library objects → plain dicts so callers see a uniform type.
            def _to_dict(obj: Any) -> Dict[str, Any]:
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()  # newer API
                if hasattr(obj, "as_dict"):
                    return obj.as_dict()  # legacy helper
                return vars(obj)  # best-effort fallback

            return [_to_dict(listen) for listen in listens_raw]
        return self._client.get_listens(username, min_ts=min_ts, max_ts=max_ts, count=count)

    def lookup_metadata(
            self,
            track_name: str,
            artist_name: str,
            *,
            incs: str | None = None,
    ) -> Dict[str, Any]:
        if self._backend == "pylb":
            return self._client.lookup_metadata(
                track_name=track_name,
                artist_name=artist_name,
                incs=incs,
            )
        return self._client.lookup_metadata(track_name, artist_name, incs=incs)

    def submit_listens(self, listen_doc: Dict[str, Any]) -> None:
        if self._backend == "pylb":
            self._client.submit_listens(listen_doc=listen_doc)
        else:
            self._client.submit_listens(listen_doc)


# --------------------------------------------------------------------------- #
# Export helper
# --------------------------------------------------------------------------- #
def export_listens_to_parquet(listens: List[Dict[str, Any]], output_path: str) -> None:
    """Convert ListenBrainz listens to a DataFrame and save as Parquet."""
    if not listens:
        logger.warning("No listens to export")
        return
    data = []
    for entry in listens:
        md = entry["track_metadata"]
        data.append({
            "Artist": md.get("artist_name", ""),
            "Album": md.get("release_name", ""),
            "Song": md.get("track_name", ""),
            "Datetime": entry.get("listened_at", 0)
        })
    df = pd.DataFrame(data)
    logger.info("Exporting %d scrobbles to %s", len(df), output_path)
    df.to_parquet(output_path, compression="zstd", index=False)
    logger.info("✅ Export completed")


# --------------------------------------------------------------------------- #
# CLI helper (so we can try things out quickly)
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Quick ListenBrainz helper")
    parser.add_argument("--user", required=True, help="MusicBrainz user ID")
    parser.add_argument("--count", type=int, default=5, help="number of listens to fetch")
    parser.add_argument("--export", type=str, help="export listens to parquet file")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    client = LBClient()
    # Use larger count for export
    count = args.count
    if args.export and count == 5:  # If default count and export requested
        count = 1000  # Use larger default for exports
    logger.info("Fetching %d listens for %s …", count, args.user)
    listens = client.get_listens(args.user, count=count)
    if not listens:
        print("No listens found.")
        sys.exit(0)
    if args.export:
        export_listens_to_parquet(listens, args.export)
    else:
        print(f"\nLast {len(listens)} listens:")
        print("-" * 72)
        for entry in listens:
            ts = datetime.fromtimestamp(entry["listened_at"], timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            md = entry["track_metadata"]
            print(f"{ts}  |  {md['artist_name']} — {md['track_name']}")
        print("-" * 72)


# --------------------------------------------------------------------------- #
# When run directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        _cli()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)

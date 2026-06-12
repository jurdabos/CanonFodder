"""
Provides versioned Parquet schema management for c9r.

Each managed table has a registered schema version.  Writers embed
``c9r_table`` and ``c9r_schema_version`` in Parquet key-value metadata
so readers can detect stale files and apply migrations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ── PyArrow schema definitions ────────────────────────────────────────────────
SCROBBLE_SCHEMA = pa.schema(
    [
        ("artist_name", pa.string()),
        ("album_title", pa.string()),
        ("track_title", pa.string()),
        ("artist_mbid", pa.string()),
        ("play_time", pa.timestamp("us", tz="UTC")),
    ]
)
ARTIST_INFO_SCHEMA = pa.schema(
    [
        ("artist_name", pa.string()),
        ("mbid", pa.string()),
        ("country", pa.string()),
        ("disambiguation_comment", pa.string()),
        ("aliases", pa.string()),
    ]
)
AVC_SCHEMA = pa.schema(
    [
        ("artist_variants_hash", pa.string()),
        ("artist_variants_text", pa.string()),
        ("canonical_name", pa.string()),
        ("to_link", pa.bool_()),
        ("comment", pa.string()),
        ("stamp", pa.timestamp("us", tz="UTC")),
    ]
)
UC_SCHEMA = pa.schema(
    [
        ("country_code", pa.string()),
        ("start_date", pa.date32()),
        ("end_date", pa.date32()),
    ]
)
GS_MB_SCHEMA = pa.schema(
    [
        ("variant_a", pa.string()),
        ("variant_b", pa.string()),
        ("to_link", pa.bool_()),
        ("source", pa.string()),
    ]
)

# ── Schema registry ───────────────────────────────────────────────────────────
SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    "scrobble": {
        "version": 1,
        "schema": SCROBBLE_SCHEMA,
        "cols": ["artist_name", "album_title", "track_title", "artist_mbid", "play_time"],
    },
    "artist_info": {
        "version": 1,
        "schema": ARTIST_INFO_SCHEMA,
        "cols": ["artist_name", "mbid", "country", "disambiguation_comment", "aliases"],
    },
    "avc": {
        "version": 1,
        "schema": AVC_SCHEMA,
        "cols": ["artist_variants_hash", "artist_variants_text", "canonical_name", "to_link", "comment", "stamp"],
    },
    "uc": {
        "version": 1,
        "schema": UC_SCHEMA,
        "cols": ["country_code", "start_date", "end_date"],
    },
    "gs_mb": {
        "version": 1,
        "schema": GS_MB_SCHEMA,
        "cols": ["variant_a", "variant_b", "to_link", "source"],
    },
}

# Mapping file path stems → table names
_PATH_TO_TABLE: dict[str, str] = {
    "scrobble.parquet": "scrobble",
    "artist_info.parquet": "artist_info",
    "avc.parquet": "avc",
    "uc.parquet": "uc",
    "gs_mb.parquet": "gs_mb",
}

# Metadata keys stored in Parquet files
_META_TABLE = b"c9r_table"
_META_VERSION = b"c9r_schema_version"


def current_version(table_name: str) -> int:
    """Returns the current schema version for *table_name*, or 0 if unknown."""
    entry = SCHEMA_REGISTRY.get(table_name)
    return entry["version"] if entry else 0


# ── Path → table resolution ──────────────────────────────────────────────────
def table_name_for_path(path: Path) -> str | None:
    """Infers the c9r table name from a Parquet file path.

    Handles both flat files (``avc.parquet``) and partitioned scrobble
    layout (``scrobble/year=YYYY/part.parquet``).
    """
    name = path.name
    if name in _PATH_TO_TABLE:
        return _PATH_TO_TABLE[name]
    # Partitioned scrobble directory (e.g. PQ/scrobble/)
    if name == "scrobble" and path.is_dir():
        return "scrobble"
    # Partitioned scrobble file: PQ/scrobble/**/part.parquet
    if name == "part.parquet":
        for parent in path.parents:
            if parent.name == "scrobble":
                return "scrobble"
    return None


# ── Metadata stamping / reading ───────────────────────────────────────────────
def stamp_metadata(table: pa.Table, table_name: str) -> pa.Table:
    """Returns a copy of *table* with c9r version metadata embedded."""
    version = current_version(table_name)
    existing = table.schema.metadata or {}
    merged = {
        **existing,
        _META_TABLE: table_name.encode(),
        _META_VERSION: str(version).encode(),
    }
    return table.replace_schema_metadata(merged)


def read_file_version(path: Path) -> tuple[str | None, int]:
    """Reads c9r metadata from a Parquet file without loading data.

    Returns ``(table_name, version)``.  Un-stamped files return
    ``(None, 0)``.
    """
    try:
        meta = pq.read_metadata(path)
    except Exception:
        return None, 0
    kv = meta.metadata or {}
    tbl = kv.get(_META_TABLE)
    ver = kv.get(_META_VERSION)
    tbl_str = tbl.decode() if tbl else None
    ver_int = int(ver.decode()) if ver else 0
    return tbl_str, ver_int


# ── Schema validation ─────────────────────────────────────────────────────────
def validate_schema(path: Path) -> dict[str, Any]:
    """Compares a Parquet file's embedded version and columns against the registry.

    Returns a status dict with keys: ``table``, ``file_version``,
    ``current_version``, ``status``, ``missing_cols``, ``extra_cols``.
    """
    tbl_name, file_ver = read_file_version(path)
    # Inferring table name from path if metadata absent
    if tbl_name is None:
        tbl_name = table_name_for_path(path)
    if tbl_name is None or tbl_name not in SCHEMA_REGISTRY:
        return {
            "table": tbl_name,
            "file_version": file_ver,
            "current_version": None,
            "status": "unmanaged",
            "missing_cols": [],
            "extra_cols": [],
        }
    entry = SCHEMA_REGISTRY[tbl_name]
    cur_ver = entry["version"]
    expected_cols = set(entry["cols"])
    # Reading actual columns from file metadata (cheap — no data load)
    try:
        schema = pq.read_schema(path)
        actual_cols = set(schema.names)
    except Exception:
        actual_cols = set()
    # Dropping synthetic partition column for scrobble
    actual_cols.discard("year")
    missing = sorted(expected_cols - actual_cols)
    extra = sorted(actual_cols - expected_cols)
    if file_ver == cur_ver and not missing:
        status = "ok"
    elif file_ver < cur_ver:
        status = "needs-migration"
    elif file_ver > cur_ver:
        status = "future-version"
    else:
        status = "column-mismatch"
    return {
        "table": tbl_name,
        "file_version": file_ver,
        "current_version": cur_ver,
        "status": status,
        "missing_cols": missing,
        "extra_cols": extra,
    }


# ── Migration registry ────────────────────────────────────────────────────────
_MIGRATIONS: dict[tuple[str, int, int], Callable[[Path], None]] = {}


def register_migration(table: str, from_ver: int, to_ver: int):
    """Decorator that registers a migration function for *(table, from→to)*."""

    def decorator(fn: Callable[[Path], None]) -> Callable[[Path], None]:
        _MIGRATIONS[(table, from_ver, to_ver)] = fn
        return fn

    return decorator


def _stamp_existing_file(path: Path, table_name: str) -> None:
    """Re-writes a Parquet file in place with c9r metadata added."""
    tbl = pq.read_table(path)
    stamped = stamp_metadata(tbl, table_name)
    pq.write_table(stamped, path, compression="zstd")
    logger.info("Stamped %s as %s v%d", path, table_name, current_version(table_name))


def _stamp_partitioned_scrobble(base_dir: Path) -> None:
    """Stamps all partition files under a scrobble directory."""
    for pf in sorted(base_dir.rglob("*.parquet")):
        _stamp_existing_file(pf, "scrobble")


# ── Built-in v0 → v1 migrations (metadata-only) ──────────────────────────────
@register_migration("scrobble", 0, 1)
def _migrate_scrobble_0_1(path: Path) -> None:
    """Stamps scrobble file(s) with v1 metadata."""
    if path.is_dir():
        _stamp_partitioned_scrobble(path)
    else:
        _stamp_existing_file(path, "scrobble")


@register_migration("artist_info", 0, 1)
def _migrate_artist_info_0_1(path: Path) -> None:
    """Stamps artist_info.parquet with v1 metadata."""
    _stamp_existing_file(path, "artist_info")


@register_migration("avc", 0, 1)
def _migrate_avc_0_1(path: Path) -> None:
    """Stamps avc.parquet with v1 metadata."""
    _stamp_existing_file(path, "avc")


@register_migration("uc", 0, 1)
def _migrate_uc_0_1(path: Path) -> None:
    """Stamps uc.parquet with v1 metadata."""
    _stamp_existing_file(path, "uc")


@register_migration("gs_mb", 0, 1)
def _migrate_gs_mb_0_1(path: Path) -> None:
    """Stamps gs_mb.parquet with v1 metadata."""
    _stamp_existing_file(path, "gs_mb")


# ── Migration execution ──────────────────────────────────────────────────────
def migrate_file(path: Path) -> int:
    """Applies the migration chain to bring *path* to the current version.

    Returns the final version after migration.
    """
    tbl_name, file_ver = read_file_version(path)
    if tbl_name is None:
        tbl_name = table_name_for_path(path)
    if tbl_name is None or tbl_name not in SCHEMA_REGISTRY:
        logger.warning("Cannot migrate unknown table: %s", path)
        return file_ver
    target = current_version(tbl_name)
    if file_ver >= target:
        logger.info("%s already at v%d", path, file_ver)
        return file_ver
    v = file_ver
    while v < target:
        next_v = v + 1
        fn = _MIGRATIONS.get((tbl_name, v, next_v))
        if fn is None:
            raise RuntimeError(f"No migration registered for {tbl_name} v{v} → v{next_v}")
        logger.info("Migrating %s v%d → v%d …", tbl_name, v, next_v)
        fn(path)
        v = next_v
    return v


def migrate_all(pq_dir: Path) -> dict[str, str]:
    """Migrates all known Parquet files in *pq_dir* to current versions.

    Returns a dict mapping relative file name → result string.
    """
    from helpers.io import SCROBBLE_PQ, SCROBBLE_PQ_DIR

    results: dict[str, str] = {}
    # Handling partitioned scrobble directory
    if SCROBBLE_PQ_DIR.exists() and any(SCROBBLE_PQ_DIR.rglob("*.parquet")):
        first_part = next(SCROBBLE_PQ_DIR.rglob("*.parquet"))
        _, ver = read_file_version(first_part)
        if ver < current_version("scrobble"):
            migrate_file(SCROBBLE_PQ_DIR)
            results["scrobble/"] = f"migrated → v{current_version('scrobble')}"
        else:
            results["scrobble/"] = f"ok (v{ver})"
    elif SCROBBLE_PQ.exists():
        _, ver = read_file_version(SCROBBLE_PQ)
        if ver < current_version("scrobble"):
            migrate_file(SCROBBLE_PQ)
            results["scrobble.parquet"] = f"migrated → v{current_version('scrobble')}"
        else:
            results["scrobble.parquet"] = f"ok (v{ver})"
    # Handling flat files
    for filename, tbl_name in _PATH_TO_TABLE.items():
        if tbl_name == "scrobble":
            continue  # to handle above
        filepath = pq_dir / filename
        if not filepath.exists():
            continue
        _, ver = read_file_version(filepath)
        target = current_version(tbl_name)
        if ver < target:
            migrate_file(filepath)
            results[filename] = f"migrated → v{target}"
        else:
            results[filename] = f"ok (v{ver})"
    return results

"""
Unit tests for helpers.schema (versioned Parquet schema management).
"""

import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers.schema import (
    current_version,
    migrate_file,
    read_file_version,
    stamp_metadata,
    table_name_for_path,
    validate_schema,
)


# ── stamp / read roundtrip ────────────────────────────────────────────────────
class TestStampAndRead:
    """Tests metadata stamping and reading helpers."""

    def test_stamp_and_read_roundtrip(self, tmp_path):
        """Stamps an Arrow table, writes it, reads version back."""
        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        stamped = stamp_metadata(table, "artist_info")
        path = tmp_path / "artist_info.parquet"
        pq.write_table(stamped, path, compression="zstd")
        tbl_name, ver = read_file_version(path)
        assert tbl_name == "artist_info"
        assert ver == current_version("artist_info")

    def test_read_legacy_file_returns_v0(self, tmp_path):
        """An un-stamped Parquet file returns (None, 0)."""
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "legacy.parquet"
        df.to_parquet(path, index=False)
        tbl_name, ver = read_file_version(path)
        assert tbl_name is None
        assert ver == 0

    def test_read_missing_file(self, tmp_path):
        """A non-existent path returns (None, 0)."""
        tbl_name, ver = read_file_version(tmp_path / "missing.parquet")
        assert tbl_name is None
        assert ver == 0


# ── table_name_for_path ───────────────────────────────────────────────────────
class TestTableNameForPath:
    """Tests path → table name resolution."""

    def test_flat_file(self, tmp_path):
        """Recognises avc.parquet as table 'avc'."""
        assert table_name_for_path(tmp_path / "avc.parquet") == "avc"

    def test_partitioned_scrobble(self, tmp_path):
        """Recognises scrobble/year=2024/part.parquet as table 'scrobble'."""
        path = tmp_path / "scrobble" / "year=2024" / "part.parquet"
        assert table_name_for_path(path) == "scrobble"

    def test_unknown_file(self, tmp_path):
        """Returns None for unrecognised filenames."""
        assert table_name_for_path(tmp_path / "random.parquet") is None


# ── validate_schema ───────────────────────────────────────────────────────────
class TestValidateSchema:
    """Tests schema validation against the registry."""

    def test_stamped_file_ok(self, tmp_path):
        """A properly stamped v1 file validates as 'ok'."""
        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        table = stamp_metadata(pa.Table.from_pandas(df, preserve_index=False), "artist_info")
        path = tmp_path / "artist_info.parquet"
        pq.write_table(table, path)
        info = validate_schema(path)
        assert info["status"] == "ok"
        assert info["table"] == "artist_info"
        assert info["file_version"] == 1
        assert info["missing_cols"] == []

    def test_legacy_file_needs_migration(self, tmp_path):
        """An un-stamped file with correct columns reports 'needs-migration'."""
        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        path = tmp_path / "artist_info.parquet"
        df.to_parquet(path, index=False)
        info = validate_schema(path)
        assert info["status"] == "needs-migration"
        assert info["file_version"] == 0

    def test_unmanaged_file(self, tmp_path):
        """An unrecognised file reports 'unmanaged'."""
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "unknown.parquet"
        df.to_parquet(path, index=False)
        info = validate_schema(path)
        assert info["status"] == "unmanaged"


# ── migrate_file ──────────────────────────────────────────────────────────────
class TestMigrateFile:
    """Tests the migration chain execution."""

    def test_migrate_v0_to_v1(self, tmp_path):
        """Migrates an un-stamped artist_info file to v1."""
        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        path = tmp_path / "artist_info.parquet"
        df.to_parquet(path, index=False)
        result_ver = migrate_file(path)
        assert result_ver == 1
        # Verifying metadata was stamped
        tbl_name, ver = read_file_version(path)
        assert tbl_name == "artist_info"
        assert ver == 1

    def test_migrate_already_current(self, tmp_path):
        """No-op when file is already at current version."""
        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        table = stamp_metadata(pa.Table.from_pandas(df, preserve_index=False), "artist_info")
        path = tmp_path / "artist_info.parquet"
        pq.write_table(table, path)
        result_ver = migrate_file(path)
        assert result_ver == 1

    def test_migrate_partitioned_scrobble(self, tmp_path):
        """Migrates a partitioned scrobble directory to v1."""
        scrobble_dir = tmp_path / "scrobble"
        year_dir = scrobble_dir / "year=2024"
        year_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "artist_name": ["A"],
                "album_title": ["Al"],
                "track_title": ["T"],
                "artist_mbid": [None],
                "play_time": pd.to_datetime(["2024-06-01"], utc=True),
            }
        )
        part = year_dir / "part.parquet"
        df.to_parquet(part, index=False)
        result_ver = migrate_file(scrobble_dir)
        assert result_ver == 1
        # Verifying the partition file got stamped
        tbl_name, ver = read_file_version(part)
        assert tbl_name == "scrobble"
        assert ver == 1


# ── dump_parquet integration ──────────────────────────────────────────────────
class TestDumpParquetStamps:
    """Tests that dump_parquet() automatically stamps metadata."""

    def test_dump_parquet_stamps_metadata(self, tmp_pq_dir):
        """Writing via dump_parquet embeds c9r version metadata."""
        import helpers.io as io_mod
        from helpers.io import dump_parquet

        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        dump_parquet(df, io_mod.ARTIST_INFO_PQ)
        tbl_name, ver = read_file_version(io_mod.ARTIST_INFO_PQ)
        assert tbl_name == "artist_info"
        assert ver == current_version("artist_info")


# ── read_parquet version checks ───────────────────────────────────────────────
class TestReadParquetVersionChecks:
    """Tests that read_parquet warns/raises on version mismatches."""

    def test_warns_on_stale_version(self, tmp_pq_dir, caplog):
        """Logs a warning when file version < current."""
        import helpers.io as io_mod

        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        # Writing without stamp (v0)
        df.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        with caplog.at_level(logging.WARNING, logger="helpers.io"):
            result = io_mod.read_parquet(io_mod.ARTIST_INFO_PQ)
        assert result is not None
        assert "schema migrate" in caplog.text

    def test_raises_on_future_version(self, tmp_pq_dir):
        """Raises RuntimeError when file version > current."""
        import helpers.io as io_mod

        df = pd.DataFrame(
            {"artist_name": ["A"], "mbid": ["x"], "country": ["DE"], "disambiguation_comment": [""], "aliases": [""]}
        )
        # Writing with a fake future version
        table = pa.Table.from_pandas(df, preserve_index=False)
        meta = {b"c9r_table": b"artist_info", b"c9r_schema_version": b"999"}
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, io_mod.ARTIST_INFO_PQ)
        with pytest.raises(RuntimeError, match="schema v999"):
            io_mod.read_parquet(io_mod.ARTIST_INFO_PQ)

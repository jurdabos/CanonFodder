"""
Unit tests for helpers.query (DuckDB analytics layer).
"""

import pandas as pd
from helpers.query import (
    artist_country_stats,
    artist_info_df,
    scrobble_count,
    scrobbles_between,
    top_artists,
    top_albums,
    unique_artists,
    user_country_scrobble_counts,
    user_country_top_entities,
)


class TestTopArtists:
    """Tests top_artists query."""

    def test_returns_top_n(self, populated_pq):
        """Returns a DataFrame with the top N artists by play count."""
        df = top_artists(n=5)
        assert not df.empty
        assert "artist_name" in df.columns
        assert "play_count" in df.columns
        # Bohren has 2 scrobbles, Ry Cooder has 1
        assert df.iloc[0]["artist_name"] == "Bohren & der Club of Gore"
        assert df.iloc[0]["play_count"] == 2


class TestScrobbleCount:
    """Tests scrobble_count query."""

    def test_counts_scrobbles(self, populated_pq):
        """Returns the total number of scrobbles."""
        assert scrobble_count() == 3


class TestUniqueArtists:
    """Tests unique_artists query."""

    def test_counts_unique_artists(self, populated_pq):
        """Returns the number of distinct artist names."""
        assert unique_artists() == 2


class TestArtistInfoDf:
    """Tests artist_info_df query."""

    def test_returns_artist_info(self, populated_pq):
        """Returns the full artist_info table."""
        df = artist_info_df()
        assert len(df) == 2
        assert "country" in df.columns

    def test_returns_empty_when_missing(self, tmp_pq_dir):
        """Returns empty DataFrame when artist_info.parquet is absent."""
        df = artist_info_df()
        assert df.empty


class TestScrobblesBetween:
    """Tests scrobbles_between date-range query."""

    def test_filters_by_date_range(self, populated_pq):
        """Returns only scrobbles within the given range."""
        df = scrobbles_between("2024-01-15T20:00:00+00:00", "2024-01-15T20:06:00+00:00")
        assert len(df) == 2  # 20:00 and 20:05 included, 20:10 excluded


# ── Canonical resolution ────────────────────────────────────────────────────
class TestCanonicalResolution:
    """Tests that queries resolve variant names through artist_info aliases."""

    @staticmethod
    def _write_variant_fixtures(pq_dir):
        """Writes scrobble + post-propagation artist_info with aliases."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["Bohren & der Club of Gore"] * 3 + ["Bohren und der Club of Gore"] * 2 + ["Autechre"],
                "album_title": ["Sunset Mission"] * 3 + ["Black Earth"] * 2 + ["Amber"],
                "track_title": [f"Track {i}" for i in range(6)],
                "artist_mbid": ["a4074512-87e0-4820-b609-0c4a18142a70"] * 5 + ["410c9baf-5469-44f6-9852-826524b80c61"],
                "play_time": pd.date_range("2024-06-01", periods=6, freq="5min", tz="UTC"),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        ai = pd.DataFrame(
            {
                "artist_name": ["Bohren & der Club of Gore", "Autechre"],
                "mbid": ["a4074512-87e0-4820-b609-0c4a18142a70", "410c9baf-5469-44f6-9852-826524b80c61"],
                "country": ["DE", "GB"],
                "disambiguation_comment": ["", ""],
                "aliases": ["Bohren und der Club of Gore", ""],
            }
        )
        ai.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)

    def test_top_artists_merges_variants(self, tmp_pq_dir):
        """Scrobbles with variant names are grouped under the canonical artist."""
        self._write_variant_fixtures(tmp_pq_dir)
        df = top_artists(n=5)
        assert df.iloc[0]["artist_name"] == "Bohren & der Club of Gore"
        assert df.iloc[0]["play_count"] == 5
        names = df["artist_name"].tolist()
        assert "Bohren und der Club of Gore" not in names

    def test_unique_artists_with_aliases(self, tmp_pq_dir):
        """Variant names resolve to fewer distinct artists."""
        self._write_variant_fixtures(tmp_pq_dir)
        assert unique_artists() == 2

    def test_top_albums_resolves_canonical(self, tmp_pq_dir):
        """Albums list the canonical artist name, not the variant."""
        self._write_variant_fixtures(tmp_pq_dir)
        df = top_albums(n=10)
        artists = df["artist_name"].unique().tolist()
        assert "Bohren und der Club of Gore" not in artists

    def test_artist_country_stats_with_aliases(self, tmp_pq_dir):
        """Country stats count variant scrobbles under the canonical artist."""
        self._write_variant_fixtures(tmp_pq_dir)
        df = artist_country_stats()
        de_row = df[df["country"] == "DE"]
        assert not de_row.empty
        assert int(de_row.iloc[0]["play_count"]) == 5


class TestArtistCountryStats:
    """Tests artist_country_stats join query."""

    def test_returns_stats(self, populated_pq):
        """Returns country-level aggregation."""
        df = artist_country_stats()
        assert not df.empty
        assert set(df.columns) == {"country", "play_count", "artist_count"}

    def test_returns_empty_when_no_info(self, tmp_pq_dir, sample_scrobble_df):
        """Returns empty when artist_info is missing."""
        import helpers.io as io_mod

        sample_scrobble_df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        df = artist_country_stats()
        assert df.empty


class TestUserCountryScrobbleCounts:
    """Tests user_country_scrobble_counts interval-join query."""

    @staticmethod
    def _write_uc_fixtures(pq_dir):
        """Writes scrobble + uc parquets for user-country testing."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["Alpha", "Beta", "Alpha", "Gamma"],
                "album_title": ["A1", "B1", "A2", "G1"],
                "track_title": ["T1", "T2", "T3", "T4"],
                "artist_mbid": [None, None, None, None],
                "play_time": pd.to_datetime(
                    [
                        "2020-03-15 10:00:00",
                        "2020-03-20 11:00:00",
                        "2022-06-01 14:00:00",
                        "2023-01-10 09:00:00",
                    ],
                    utc=True,
                ),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        uc = pd.DataFrame(
            {
                "country_code": ["DE", "HU"],
                "start_date": pd.to_datetime(["2019-01-01", "2022-01-01"]).date,
                "end_date": [pd.Timestamp("2021-12-31").date(), None],
            }
        )
        uc.to_parquet(io_mod.UC_PQ, index=False)

    def test_returns_counts_by_country(self, tmp_pq_dir):
        """Groups scrobbles by the user's country at play time."""
        self._write_uc_fixtures(tmp_pq_dir)
        df = user_country_scrobble_counts()
        assert not df.empty
        assert set(df.columns) == {"country_code", "scrobble_count"}
        codes = df["country_code"].tolist()
        assert "DE" in codes
        assert "HU" in codes

    def test_de_gets_two_scrobbles(self, tmp_pq_dir):
        """DE interval (2019–2021) matches two scrobbles from 2020."""
        self._write_uc_fixtures(tmp_pq_dir)
        df = user_country_scrobble_counts()
        de_row = df[df["country_code"] == "DE"]
        assert int(de_row.iloc[0]["scrobble_count"]) == 2

    def test_hu_gets_two_scrobbles(self, tmp_pq_dir):
        """HU interval (2022–open) matches two scrobbles."""
        self._write_uc_fixtures(tmp_pq_dir)
        df = user_country_scrobble_counts()
        hu_row = df[df["country_code"] == "HU"]
        assert int(hu_row.iloc[0]["scrobble_count"]) == 2

    def test_returns_empty_when_no_uc(self, tmp_pq_dir, sample_scrobble_df):
        """Returns empty when uc.parquet is missing."""
        import helpers.io as io_mod

        sample_scrobble_df.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        df = user_country_scrobble_counts()
        assert df.empty

    def test_returns_empty_when_no_scrobbles(self, tmp_pq_dir):
        """Returns empty when scrobble.parquet is missing."""
        df = user_country_scrobble_counts()
        assert df.empty


class TestUserCountryTopEntities:
    """Tests user_country_top_entities ranked medal query."""

    @staticmethod
    def _write_uc_fixtures(pq_dir):
        """Writes scrobble + uc parquets with varied albums/tracks."""
        import helpers.io as io_mod

        scrobbles = pd.DataFrame(
            {
                "artist_name": ["Alpha", "Alpha", "Beta", "Gamma", "Gamma"],
                "album_title": ["A1", "A1", "B1", "G1", "G2"],
                "track_title": ["T1", "T2", "T3", "T4", "T5"],
                "artist_mbid": [None] * 5,
                "play_time": pd.to_datetime(
                    [
                        "2020-03-15 10:00:00",
                        "2020-03-16 10:00:00",
                        "2020-03-17 11:00:00",
                        "2022-06-01 14:00:00",
                        "2022-06-02 14:00:00",
                    ],
                    utc=True,
                ),
            }
        )
        scrobbles.to_parquet(io_mod.SCROBBLE_PQ, index=False)
        uc = pd.DataFrame(
            {
                "country_code": ["DE", "HU"],
                "start_date": pd.to_datetime(["2019-01-01", "2022-01-01"]).date,
                "end_date": [pd.Timestamp("2021-12-31").date(), None],
            }
        )
        uc.to_parquet(io_mod.UC_PQ, index=False)

    def test_returns_three_categories(self, tmp_pq_dir):
        """Returns a dict with artists, albums, and tracks DataFrames."""
        self._write_uc_fixtures(tmp_pq_dir)
        result = user_country_top_entities(top_n=3)
        assert set(result.keys()) == {"artists", "albums", "tracks"}
        for key in result:
            assert "country_code" in result[key].columns
            assert "rank" in result[key].columns
            assert "play_count" in result[key].columns

    def test_artists_ranked_per_country(self, tmp_pq_dir):
        """DE has Alpha as #1 (2 scrobbles), Beta as #2."""
        self._write_uc_fixtures(tmp_pq_dir)
        artists = user_country_top_entities(top_n=3)["artists"]
        de = artists[artists["country_code"] == "DE"]
        assert not de.empty
        assert de.iloc[0]["artist_name"] == "Alpha"
        assert int(de.iloc[0]["play_count"]) == 2

    def test_hu_has_gamma(self, tmp_pq_dir):
        """HU has Gamma as #1."""
        self._write_uc_fixtures(tmp_pq_dir)
        artists = user_country_top_entities(top_n=3)["artists"]
        hu = artists[artists["country_code"] == "HU"]
        assert not hu.empty
        assert hu.iloc[0]["artist_name"] == "Gamma"

    def test_albums_have_title(self, tmp_pq_dir):
        """Album results include album_title column."""
        self._write_uc_fixtures(tmp_pq_dir)
        albums = user_country_top_entities(top_n=3)["albums"]
        assert "album_title" in albums.columns
        assert not albums.empty

    def test_tracks_have_title(self, tmp_pq_dir):
        """Track results include track_title column."""
        self._write_uc_fixtures(tmp_pq_dir)
        tracks = user_country_top_entities(top_n=3)["tracks"]
        assert "track_title" in tracks.columns
        assert not tracks.empty

    def test_respects_top_n(self, tmp_pq_dir):
        """Limits results to top_n per country."""
        self._write_uc_fixtures(tmp_pq_dir)
        result = user_country_top_entities(top_n=1)
        for key in ("artists", "albums", "tracks"):
            df = result[key]
            for cc in df["country_code"].unique():
                assert len(df[df["country_code"] == cc]) <= 1

    def test_returns_empty_when_no_data(self, tmp_pq_dir):
        """Returns empty DataFrames when parquets are missing."""
        result = user_country_top_entities(top_n=3)
        for key in ("artists", "albums", "tracks"):
            assert result[key].empty

"""
Unit tests for HTTP.lfAPI (Last.fm API helpers).
"""
import pandas as pd
from HTTP.lfAPI import (
    _clean_track,
    _normalise_tracks,
    _update_user_country,
    generate_lastfm_signature,
    iso2_for_en_name,
)


class TestCleanTrack:
    """Tests the _clean_track normaliser."""

    def test_valid_track(self):
        """Extracts artist, album, track, uts from a well-formed record."""
        rec = {
            "artist": {"#text": "Bohren", "mbid": "abc-123"},
            "album": {"#text": "Sunset Mission"},
            "name": "Prowler",
            "date": {"uts": "1700000000"},
        }
        result = _clean_track(rec)
        assert result["artist_name"] == "Bohren"
        assert result["track_title"] == "Prowler"
        assert result["uts"] == 1700000000

    def test_now_playing(self):
        """Returns uts=0 for a now-playing track (no date key)."""
        rec = {
            "artist": {"#text": "X"},
            "album": {"#text": "A"},
            "name": "T",
            "date": {},
        }
        result = _clean_track(rec)
        assert result["uts"] == 0

    def test_non_dict_returns_none(self):
        """Returns None for non-dict input."""
        assert _clean_track("not a dict") is None

    def test_missing_nested_fields(self):
        """Handles missing artist/album gracefully."""
        rec = {"name": "T", "date": {"uts": "100"}}
        result = _clean_track(rec)
        assert result["artist_name"] == ""
        assert result["album_title"] == ""


class TestNormaliseTracks:
    """Tests the _normalise_tracks batch normaliser."""

    def test_filters_now_playing(self):
        """Skips tracks without a 'date' key."""
        tracks = [
            {"artist": {"#text": "A", "mbid": ""}, "album": {"#text": "Al"}, "name": "T", "date": {"uts": "100"}},
            {"artist": {"#text": "B"}, "album": {"#text": "Al"}, "name": "T2"},  # no date → now playing
        ]
        result = _normalise_tracks(tracks)
        assert len(result) == 1
        assert result[0]["Artist"] == "A"

    def test_string_artist(self):
        """Handles artist as plain string instead of dict."""
        tracks = [{"artist": "PlainArtist", "album": {"#text": "Al"}, "name": "T", "date": {"uts": "200"}}]
        result = _normalise_tracks(tracks)
        assert result[0]["Artist"] == "PlainArtist"


class TestIso2ForEnName:
    """Tests the country-name → ISO-2 translator."""

    def test_returns_none_when_no_parquet(self, tmp_pq_dir):
        """Returns None when c.parquet does not exist."""
        assert iso2_for_en_name("Germany") is None

    def test_exact_match(self, tmp_pq_dir):
        """Finds exact (case-insensitive) match."""
        import helpers.io as io_mod
        df = pd.DataFrame({"ISO2": ["DE", "US"], "en_name": ["Germany", "United States"]})
        df.to_parquet(io_mod.C_PQ, index=False)
        assert iso2_for_en_name("germany") == "DE"

    def test_fuzzy_match(self, tmp_pq_dir):
        """Finds a close match within edit distance 1."""
        import helpers.io as io_mod
        df = pd.DataFrame({"ISO2": ["HU"], "en_name": ["Hungary"]})
        df.to_parquet(io_mod.C_PQ, index=False)
        assert iso2_for_en_name("Xungabc") is None  # edit distance >1 → no match
        assert iso2_for_en_name("Hungaru") == "HU"  # edit distance 1 → match

    def test_no_match(self, tmp_pq_dir):
        """Returns None when nothing matches."""
        import helpers.io as io_mod
        df = pd.DataFrame({"ISO2": ["DE"], "en_name": ["Germany"]})
        df.to_parquet(io_mod.C_PQ, index=False)
        assert iso2_for_en_name("Narnia") is None


class TestUpdateUserCountry:
    """Tests the uc.parquet update logic."""

    def test_creates_new_file(self, tmp_pq_dir):
        """Creates uc.parquet when it doesn't exist."""
        changed = _update_user_country("HU")
        assert changed is True
        import helpers.io as io_mod
        df = pd.read_parquet(io_mod.UC_PQ)
        assert len(df) == 1
        assert df.iloc[0]["country_code"] == "HU"

    def test_no_change_same_country(self, tmp_pq_dir):
        """Returns False when the country hasn't changed."""
        import helpers.io as io_mod
        from datetime import date
        today = str(date.today())
        existing = pd.DataFrame([{"country_code": "HU", "start_date": today, "end_date": None}])
        existing.to_parquet(io_mod.UC_PQ, index=False)
        assert _update_user_country("HU") is False

    def test_adds_new_row_on_change(self, tmp_pq_dir):
        """Adds a new row and closes the old one when country changes."""
        import helpers.io as io_mod
        existing = pd.DataFrame([{"country_code": "HU", "start_date": "2020-01-01", "end_date": None}])
        existing.to_parquet(io_mod.UC_PQ, index=False)
        changed = _update_user_country("DE")
        assert changed is True
        df = pd.read_parquet(io_mod.UC_PQ)
        assert len(df) == 2
        assert df.iloc[-1]["country_code"] == "DE"


class TestGenerateLastfmSignature:
    """Tests the Last.fm API signature generator."""

    def test_deterministic(self):
        """Produces the same hash for the same inputs."""
        params = {"method": "auth.getSession", "token": "abc123"}
        sig1 = generate_lastfm_signature(params, "secret")
        sig2 = generate_lastfm_signature(params, "secret")
        assert sig1 == sig2
        assert len(sig1) == 32  # MD5 hex

    def test_different_secret(self):
        """Produces different hash with different secret."""
        params = {"a": "1"}
        sig1 = generate_lastfm_signature(params, "secret1")
        sig2 = generate_lastfm_signature(params, "secret2")
        assert sig1 != sig2

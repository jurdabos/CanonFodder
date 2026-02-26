"""
Tests for corefunc.profile — data profiling functions.
"""
import pandas as pd
from corefunc.profile import (
    country_breakdown,
    overview_stats,
    population_vs_scrobbles,
    top_artists_profile,
    trusted_companions,
    user_country_medal_profile,
    user_country_profile,
    variant_candidates,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _scrobble_df(years=(2023, 2024, 2025)) -> pd.DataFrame:
    """Builds a scrobble DataFrame spanning multiple years with varied artists."""
    rows = []
    artists = [
        ("Bohren & der Club of Gore", "Sunset Mission", "Prowler", "a4074512-87e0-4820-b609-0c4a18142a70"),
        ("Bohren und der Club of Gore", "Black Earth", "Constant Fear", "a4074512-87e0-4820-b609-0c4a18142a70"),
        ("Autechre", "Amber", "Foil", "410c9baf-5469-44f6-9852-826524b80c61"),
        ("Radiohead", "OK Computer", "Paranoid Android", "a74b1b7f-71a5-4011-9441-d0b5e4122711"),
        ("Secret Chiefs 3", "Book of Horizons", "Zulfikar", "b5f3a039-10fa-44d6-99f2-27aeb5e5bfd0"),
    ]
    for year in years:
        for i, (artist, album, track, mbid) in enumerate(artists):
            # Varying play counts: Bohren & has more, Bohren und has fewer
            count = 10 if artist.startswith("Bohren &") else 5 if artist.startswith("Bohren und") else 8
            for j in range(count):
                rows.append({
                    "artist_name": artist,
                    "album_title": album,
                    "track_title": f"{track} #{j}",
                    "artist_mbid": mbid,
                    "play_time": pd.Timestamp(f"{year}-06-{(i * 5 + j + 1) % 28 + 1:02d} 20:00", tz="UTC"),
                })
    return pd.DataFrame(rows)


def _artist_info_df() -> pd.DataFrame:
    """Builds an artist_info DataFrame matching the scrobble artists."""
    return pd.DataFrame({
        "artist_name": [
            "Bohren & der Club of Gore", "Bohren und der Club of Gore",
            "Autechre", "Radiohead", "Secret Chiefs 3",
        ],
        "mbid": [
            "a4074512-87e0-4820-b609-0c4a18142a70", "a4074512-87e0-4820-b609-0c4a18142a70",
            "410c9baf-5469-44f6-9852-826524b80c61", "a74b1b7f-71a5-4011-9441-d0b5e4122711",
            "b5f3a039-10fa-44d6-99f2-27aeb5e5bfd0",
        ],
        "country": ["DE", "DE", "GB", "GB", "US"],
        "disambiguation_comment": ["", "", "", "", ""],
        "aliases": ["", "", "", "", ""],
    })


def _country_codes_df() -> pd.DataFrame:
    """Builds a minimal c.parquet with ISO-2 to English name mapping."""
    return pd.DataFrame({
        "ISO-2": ["DE", "GB", "US"],
        "ISO-3": ["DEU", "GBR", "USA"],
        "en_name": ["Germany", "United Kingdom", "United States"],
        "hu_name": ["Németország", "Egyesült Királyság", "Egyesült Államok"],
    })


def _write_fixtures(pq_dir, years=(2023, 2024, 2025)):
    """Writes scrobble + artist_info + c parquets and returns the directory."""
    _scrobble_df(years).to_parquet(pq_dir / "scrobble.parquet", index=False)
    _artist_info_df().to_parquet(pq_dir / "artist_info.parquet", index=False)
    _country_codes_df().to_parquet(pq_dir / "c.parquet", index=False)
    return pq_dir


# ── Overview stats ────────────────────────────────────────────────────────────
class TestOverviewStats:
    """Tests for overview_stats."""

    def test_missing_parquet(self, tmp_pq_dir):
        """Returns error dict when scrobble.parquet does not exist."""
        result = overview_stats()
        assert "error" in result

    def test_basic_stats(self, tmp_pq_dir):
        """Returns correct totals and distribution keys."""
        _write_fixtures(tmp_pq_dir)
        result = overview_stats()
        assert result["total_scrobbles"] > 0
        assert result["unique_artists"] == 5
        assert "distribution" in result
        assert "yearly" in result
        d = result["distribution"]
        assert d["total_artists"] == 5
        assert d["min"] > 0
        assert d["median"] >= d["min"]

    def test_yearly_breakdown(self, tmp_pq_dir):
        """Returns yearly totals for all years in data."""
        _write_fixtures(tmp_pq_dir, years=(2023, 2024))
        result = overview_stats()
        years = [int(y) for y, _ in result["yearly"]]
        assert 2023 in years
        assert 2024 in years

    def test_distribution_quartiles(self, tmp_pq_dir):
        """Quartile values are monotonically non-decreasing."""
        _write_fixtures(tmp_pq_dir)
        d = overview_stats()["distribution"]
        assert d["min"] <= d["q25"] <= d["median"] <= d["q75"] <= d["max"]


# ── Variant candidates ────────────────────────────────────────────────────────
class TestVariantCandidates:
    """Tests for variant_candidates (the Bohren problem)."""

    def test_missing_parquet(self, tmp_pq_dir):
        """Returns empty list when scrobble.parquet does not exist."""
        result = variant_candidates()
        assert result == []

    def test_finds_bohren_pair(self, tmp_pq_dir):
        """Detects the Bohren & / Bohren und near-duplicate pair."""
        _write_fixtures(tmp_pq_dir)
        result = variant_candidates(threshold=80, min_plays=3)
        assert len(result) >= 1
        # Finding the Bohren cluster
        bohren = [c for c in result if any("Bohren" in v["name"] for v in c["variants"])]
        assert len(bohren) >= 1
        pair = bohren[0]
        names = {v["name"] for v in pair["variants"]}
        assert "Bohren & der Club of Gore" in names
        assert "Bohren und der Club of Gore" in names

    def test_combined_count(self, tmp_pq_dir):
        """Combined count equals sum of individual variant play counts."""
        _write_fixtures(tmp_pq_dir)
        result = variant_candidates(threshold=80, min_plays=3)
        for c in result:
            assert c["combined_count"] == sum(v["plays"] for v in c["variants"])

    def test_similarity_above_threshold(self, tmp_pq_dir):
        """All returned pairs have similarity >= threshold."""
        _write_fixtures(tmp_pq_dir)
        threshold = 80
        result = variant_candidates(threshold=threshold, min_plays=3)
        for c in result:
            assert c["similarity"] >= threshold

    def test_high_threshold_filters(self, tmp_pq_dir):
        """A very high threshold should return fewer or no pairs."""
        _write_fixtures(tmp_pq_dir)
        strict = variant_candidates(threshold=99, min_plays=3)
        relaxed = variant_candidates(threshold=70, min_plays=3)
        assert len(strict) <= len(relaxed)

    def test_sorted_by_combined_count(self, tmp_pq_dir):
        """Results are sorted descending by combined_count."""
        _write_fixtures(tmp_pq_dir)
        result = variant_candidates(threshold=70, min_plays=3)
        counts = [c["combined_count"] for c in result]
        assert counts == sorted(counts, reverse=True)


# ── Top artists ───────────────────────────────────────────────────────────────
class TestTopArtistsProfile:
    """Tests for top_artists_profile."""

    def test_missing_parquet(self, tmp_pq_dir):
        """Returns error dict when scrobble.parquet does not exist."""
        result = top_artists_profile()
        assert "error" in result

    def test_raw_top(self, tmp_pq_dir):
        """Returns ranked artists with play counts."""
        _write_fixtures(tmp_pq_dir)
        result = top_artists_profile(n=3)
        raw = result["raw_top"]
        assert len(raw) == 3
        assert raw[0]["rank"] == 1
        assert raw[0]["plays"] >= raw[1]["plays"]

    def test_canonize_with_empty_aliases(self, tmp_pq_dir):
        """With empty aliases in artist_info, canon_top matches raw_top."""
        _write_fixtures(tmp_pq_dir)
        result = top_artists_profile(n=5, canonize=True)
        assert "raw_top" in result
        assert "canon_top" in result
        raw_names = {e["name"] for e in result["raw_top"]}
        canon_names = {e["name"] for e in result["canon_top"]}
        assert raw_names == canon_names

    def test_canonize_with_aliases(self, tmp_pq_dir):
        """Applies artist_info alias mapping and re-ranks artists."""
        _write_fixtures(tmp_pq_dir)
        # Overwriting artist_info to simulate post-propagation state
        import helpers.io as io_mod
        ai = pd.DataFrame({
            "artist_name": [
                "Bohren & der Club of Gore",
                "Autechre", "Radiohead", "Secret Chiefs 3",
            ],
            "mbid": [
                "a4074512-87e0-4820-b609-0c4a18142a70",
                "410c9baf-5469-44f6-9852-826524b80c61",
                "a74b1b7f-71a5-4011-9441-d0b5e4122711",
                "b5f3a039-10fa-44d6-99f2-27aeb5e5bfd0",
            ],
            "country": ["DE", "GB", "GB", "US"],
            "disambiguation_comment": ["", "", "", ""],
            "aliases": ["Bohren und der Club of Gore", "", "", ""],
        })
        ai.to_parquet(io_mod.ARTIST_INFO_PQ, index=False)
        result = top_artists_profile(n=5, canonize=True)
        assert "canon_top" in result
        canon_names = [e["name"] for e in result["canon_top"]]
        assert "Bohren & der Club of Gore" in canon_names
        assert "Bohren und der Club of Gore" not in canon_names
        ranks = [e["rank"] for e in result["canon_top"]]
        assert ranks == list(range(1, len(ranks) + 1))


# ── Trusted companions ────────────────────────────────────────────────────────
class TestTrustedCompanions:
    """Tests for trusted_companions."""

    def test_missing_parquet(self, tmp_pq_dir):
        """Returns error dict when scrobble.parquet does not exist."""
        result = trusted_companions()
        assert "error" in result

    def test_all_artists_present(self, tmp_pq_dir):
        """All 5 test artists appear in every year, so all are companions."""
        _write_fixtures(tmp_pq_dir, years=(2023, 2024, 2025))
        result = trusted_companions(start_year=2023, end_year=2025)
        assert result["year_count"] == 3
        assert len(result["companions"]) == 5

    def test_companion_sorted_by_std(self, tmp_pq_dir):
        """Companions are sorted by standard deviation ascending."""
        _write_fixtures(tmp_pq_dir)
        result = trusted_companions(start_year=2023, end_year=2025)
        stds = [c["std_dev"] for c in result["companions"]]
        assert stds == sorted(stds)

    def test_narrow_range_no_companions(self, tmp_pq_dir):
        """A year range with no data returns empty companions."""
        _write_fixtures(tmp_pq_dir, years=(2023,))
        result = trusted_companions(start_year=2020, end_year=2022)
        assert result["companions"] == []

    def test_yearly_plays_populated(self, tmp_pq_dir):
        """Each companion has yearly_plays dict with correct year keys."""
        _write_fixtures(tmp_pq_dir, years=(2023, 2024))
        result = trusted_companions(start_year=2023, end_year=2024)
        for c in result["companions"]:
            assert 2023 in c["yearly_plays"]
            assert 2024 in c["yearly_plays"]
            assert c["total_plays"] == sum(c["yearly_plays"].values())


# ── Country breakdown ─────────────────────────────────────────────────────────
class TestCountryBreakdown:
    """Tests for country_breakdown."""

    def test_missing_parquet(self, tmp_pq_dir):
        """Returns empty list when parquet files do not exist."""
        result = country_breakdown()
        assert result == []

    def test_returns_countries(self, tmp_pq_dir):
        """Returns countries with play counts, artist counts, and English names."""
        _write_fixtures(tmp_pq_dir)
        result = country_breakdown(top_n=10)
        assert len(result) > 0
        codes = [r["country"] for r in result]
        assert "DE" in codes
        assert "GB" in codes
        assert "US" in codes
        # Verifying English name from c.parquet
        de_row = next(r for r in result if r["country"] == "DE")
        assert de_row["name"] == "Germany"

    def test_sorted_by_play_count(self, tmp_pq_dir):
        """Results are sorted descending by play count."""
        _write_fixtures(tmp_pq_dir)
        result = country_breakdown(top_n=10)
        counts = [r["play_count"] for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_pct_sums_to_100(self, tmp_pq_dir):
        """Percentages sum to approximately 100 when all countries are included."""
        _write_fixtures(tmp_pq_dir)
        result = country_breakdown(top_n=100)
        total_pct = sum(r["pct"] for r in result)
        assert abs(total_pct - 100.0) < 0.5

    def test_top_n_limits(self, tmp_pq_dir):
        """Respects the top_n parameter."""
        _write_fixtures(tmp_pq_dir)
        result = country_breakdown(top_n=2)
        assert len(result) == 2


# ── Population vs. scrobbles ──────────────────────────────────────────────────
def _write_uc_fixtures(pq_dir, years=(2023, 2024, 2025)):
    """Writes scrobble + artist_info + c + uc parquets for user-country tests."""
    _scrobble_df(years).to_parquet(pq_dir / "scrobble.parquet", index=False)
    _artist_info_df().to_parquet(pq_dir / "artist_info.parquet", index=False)
    _country_codes_df().to_parquet(pq_dir / "c.parquet", index=False)
    uc = pd.DataFrame({
        "country_code": ["DE", "HU"],
        "start_date": pd.to_datetime(["2022-01-01", "2024-01-01"]).date,
        "end_date": [pd.Timestamp("2023-12-31").date(), None],
    })
    uc.to_parquet(pq_dir / "uc.parquet", index=False)
    return pq_dir


class TestPopulationVsScrobbles:
    """Tests for population_vs_scrobbles."""

    def test_missing_data(self, tmp_pq_dir):
        """Returns error dict when artist_info is absent."""
        result = population_vs_scrobbles()
        assert "error" in result

    def test_returns_rankings(self, tmp_pq_dir):
        """Returns both absolute and per-capita rankings."""
        _write_fixtures(tmp_pq_dir)
        result = population_vs_scrobbles(top_n=5)
        assert "by_absolute" in result
        assert "by_per_capita" in result
        assert result["total_countries"] > 0

    def test_absolute_sorted_descending(self, tmp_pq_dir):
        """Absolute ranking is sorted by play_count descending."""
        _write_fixtures(tmp_pq_dir)
        result = population_vs_scrobbles(top_n=10)
        counts = [r["play_count"] for r in result["by_absolute"]]
        assert counts == sorted(counts, reverse=True)

    def test_per_capita_sorted_descending(self, tmp_pq_dir):
        """Per-capita ranking is sorted by per_million descending."""
        _write_fixtures(tmp_pq_dir)
        result = population_vs_scrobbles(top_n=10)
        rates = [r["per_million"] for r in result["by_per_capita"]]
        assert rates == sorted(rates, reverse=True)

    def test_entries_have_population(self, tmp_pq_dir):
        """Each entry includes a positive population value."""
        _write_fixtures(tmp_pq_dir)
        result = population_vs_scrobbles(top_n=10)
        for r in result["by_absolute"]:
            assert r["population"] > 0


# ── User country profile ──────────────────────────────────────────────────────
class TestUserCountryProfile:
    """Tests for user_country_profile."""

    def test_missing_uc(self, tmp_pq_dir):
        """Returns error dict when uc.parquet is absent."""
        result = user_country_profile()
        assert "error" in result

    def test_returns_countries(self, tmp_pq_dir):
        """Returns country entries with counts and percentages."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_profile(top_n=10)
        assert "countries" in result
        assert len(result["countries"]) > 0
        assert result["total_scrobbles_matched"] > 0

    def test_pct_within_bounds(self, tmp_pq_dir):
        """Percentages are between 0 and 100."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_profile(top_n=10)
        for r in result["countries"]:
            assert 0.0 <= r["pct"] <= 100.0

    def test_includes_country_names(self, tmp_pq_dir):
        """Country entries include English names from c.parquet."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_profile(top_n=10)
        names = [r["name"] for r in result["countries"]]
        # At least one known name from the fixture should be present
        assert any(n in names for n in ["Germany", "Hungary"])

    def test_unique_countries_count(self, tmp_pq_dir):
        """Reports the correct number of unique countries."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_profile(top_n=10)
        assert result["unique_countries"] == 2


# ── User-country medal profile ────────────────────────────────────────────────
class TestUserCountryMedalProfile:
    """Tests for user_country_medal_profile."""

    def test_missing_uc(self, tmp_pq_dir):
        """Returns error dict when uc.parquet is absent."""
        result = user_country_medal_profile()
        assert "error" in result

    def test_returns_countries_with_medals(self, tmp_pq_dir):
        """Returns per-country medal entries for all 3 categories."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, ucn=2)
        assert "countries" in result
        assert len(result["countries"]) == 2
        for c in result["countries"]:
            assert "artists" in c
            assert "albums" in c
            assert "tracks" in c

    def test_medal_entries_have_rank_and_name(self, tmp_pq_dir):
        """Each medal entry has rank, name, and plays keys."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=2, ucn=1)
        c = result["countries"][0]
        for key in ("artists", "albums", "tracks"):
            for e in c[key]:
                assert "rank" in e
                assert "name" in e
                assert "plays" in e

    def test_ucn_limits_countries(self, tmp_pq_dir):
        """The ucn parameter limits the number of countries in the output."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, ucn=1)
        assert len(result["countries"]) == 1

    def test_includes_country_names(self, tmp_pq_dir):
        """Country entries include English names from c.parquet."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, ucn=2)
        names = [c["name"] for c in result["countries"]]
        assert any(n for n in names if n)  # to verify at least one name resolved

    def test_scrobble_count_per_country(self, tmp_pq_dir):
        """Each country entry reports its scrobble count."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, ucn=2)
        for c in result["countries"]:
            assert c["scrobble_count"] > 0

    def test_country_codes_filters_to_specified(self, tmp_pq_dir):
        """Passing country_codes returns only those countries."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, country_codes=["HU"])
        codes = [c["country"] for c in result["countries"]]
        assert codes == ["HU"]
        assert result["ucn"] == 1

    def test_country_codes_ignores_unknown(self, tmp_pq_dir):
        """Unknown codes are silently dropped from the output."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, country_codes=["XX", "DE"])
        codes = [c["country"] for c in result["countries"]]
        assert "XX" not in codes
        assert "DE" in codes

    def test_country_codes_case_insensitive(self, tmp_pq_dir):
        """Lowercase codes are normalised to uppercase."""
        _write_uc_fixtures(tmp_pq_dir)
        result = user_country_medal_profile(top_n=3, country_codes=["de"])
        codes = [c["country"] for c in result["countries"]]
        assert "DE" in codes

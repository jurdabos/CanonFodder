"""
Tests for the time-series query layer and temporal profile analytics.

Exercises monthly_scrobble_counts, yearly_top_n_artists, listening_clock,
daily_scrobble_dates (query layer) and monthly_summary,
yearly_top_artists_profile, streak_analysis, listening_clock_profile
(profile layer).
"""
from __future__ import annotations


# ── Query-layer tests ─────────────────────────────────────────────────────────
class TestMonthlyScrobbleCounts:
    """Tests helpers.query.monthly_scrobble_counts."""

    def test_returns_correct_months(self, temporal_pq):
        from helpers.query import monthly_scrobble_counts
        df = monthly_scrobble_counts()
        assert not df.empty
        # Fixture has data in months 1, 2, 7, 12
        months_present = sorted(df["month"].unique())
        assert 1 in months_present
        assert 7 in months_present
        assert 12 in months_present

    def test_counts_are_positive(self, temporal_pq):
        from helpers.query import monthly_scrobble_counts
        df = monthly_scrobble_counts()
        assert (df["scrobble_count"] > 0).all()

    def test_empty_when_no_data(self, tmp_pq_dir):
        from helpers.query import monthly_scrobble_counts
        df = monthly_scrobble_counts()
        assert df.empty


class TestYearlyTopNArtists:
    """Tests helpers.query.yearly_top_n_artists."""

    def test_returns_ranked_artists(self, temporal_pq):
        from helpers.query import yearly_top_n_artists
        df = yearly_top_n_artists(top_n=2)
        assert not df.empty
        assert set(df.columns) >= {"year", "rank", "artist_name", "play_count"}
        # Checking that rank is bounded by top_n
        assert df["rank"].max() <= 2

    def test_top1_per_year(self, temporal_pq):
        from helpers.query import yearly_top_n_artists
        df = yearly_top_n_artists(top_n=1)
        # Each year should have exactly one row
        assert df.groupby("year").size().max() == 1

    def test_alpha_dominates_2023(self, temporal_pq):
        """Alpha has 7 scrobbles in 2023 (3 Jan + 4 Dec) vs Beta's 2."""
        from helpers.query import yearly_top_n_artists
        df = yearly_top_n_artists(top_n=1)
        row_2023 = df[df["year"] == 2023]
        assert len(row_2023) == 1
        assert row_2023.iloc[0]["artist_name"] == "Alpha"


class TestListeningClock:
    """Tests helpers.query.listening_clock."""

    def test_hourly_buckets(self, temporal_pq):
        from helpers.query import listening_clock
        df = listening_clock(granularity="hour")
        assert not df.empty
        assert "hour" in df.columns
        assert "scrobble_count" in df.columns
        # Fixture has scrobbles at various hours (8–23 range)
        assert df["hour"].min() >= 0
        assert df["hour"].max() <= 23

    def test_weekday_buckets(self, temporal_pq):
        from helpers.query import listening_clock
        df = listening_clock(granularity="weekday")
        assert not df.empty
        assert "weekday" in df.columns
        assert df["weekday"].min() >= 0
        assert df["weekday"].max() <= 6


class TestDailyScrobbleDates:
    """Tests helpers.query.daily_scrobble_dates."""

    def test_returns_distinct_dates(self, temporal_pq):
        from helpers.query import daily_scrobble_dates
        df = daily_scrobble_dates()
        assert not df.empty
        assert df["play_date"].is_unique


# ── Profile-layer tests ───────────────────────────────────────────────────────
class TestMonthlySummary:
    """Tests corefunc.profile.monthly_summary."""

    def test_returns_all_12_months(self, temporal_pq):
        from corefunc.profile import monthly_summary
        result = monthly_summary()
        assert "months" in result
        assert len(result["months"]) == 12

    def test_strongest_and_weakest(self, temporal_pq):
        from corefunc.profile import monthly_summary
        result = monthly_summary()
        assert result["strongest"] is not None
        assert result["weakest"] is not None
        assert result["strongest"]["mean"] >= result["weakest"]["mean"]

    def test_january_has_data(self, temporal_pq):
        """Fixture has scrobbles in Jan across 2023, 2024, 2025."""
        from corefunc.profile import monthly_summary
        result = monthly_summary()
        jan = result["months"][0]
        assert jan["month"] == 1
        assert jan["year_count"] == 3
        assert jan["total"] == 10  # 3 + 5 + 2

    def test_error_when_empty(self, tmp_pq_dir):
        from corefunc.profile import monthly_summary
        result = monthly_summary()
        assert "error" in result


class TestYearlyTopArtistsProfile:
    """Tests corefunc.profile.yearly_top_artists_profile."""

    def test_returns_year_entries(self, temporal_pq):
        from corefunc.profile import yearly_top_artists_profile
        result = yearly_top_artists_profile(top_n=2)
        assert "years" in result
        assert len(result["years"]) >= 3  # 2023, 2024, 2025

    def test_each_year_has_ranked_artists(self, temporal_pq):
        from corefunc.profile import yearly_top_artists_profile
        result = yearly_top_artists_profile(top_n=2)
        for yr in result["years"]:
            assert "artists" in yr
            assert len(yr["artists"]) <= 2
            for a in yr["artists"]:
                assert "rank" in a
                assert "name" in a
                assert "plays" in a


class TestStreakAnalysis:
    """Tests corefunc.profile.streak_analysis."""

    def test_returns_streak_keys(self, temporal_pq):
        from corefunc.profile import streak_analysis
        result = streak_analysis()
        assert "total_active_days" in result
        assert "longest_streak" in result
        assert "current_streak" in result
        assert "longest_gap_days" in result

    def test_active_days_plausible(self, temporal_pq):
        from corefunc.profile import streak_analysis
        result = streak_analysis()
        # Fixture has scrobbles on ~19 distinct days
        assert result["total_active_days"] >= 15

    def test_longest_streak_positive(self, temporal_pq):
        from corefunc.profile import streak_analysis
        result = streak_analysis()
        assert result["longest_streak"] >= 1

    def test_error_when_empty(self, tmp_pq_dir):
        from corefunc.profile import streak_analysis
        result = streak_analysis()
        assert "error" in result


class TestListeningClockProfile:
    """Tests corefunc.profile.listening_clock_profile."""

    def test_returns_hours_and_weekdays(self, temporal_pq):
        from corefunc.profile import listening_clock_profile
        result = listening_clock_profile()
        assert "hours" in result
        assert "weekdays" in result
        assert len(result["hours"]) > 0
        assert len(result["weekdays"]) > 0

    def test_peak_and_quiet(self, temporal_pq):
        from corefunc.profile import listening_clock_profile
        result = listening_clock_profile()
        assert result["peak_hour"] is not None
        assert result["quiet_hour"] is not None
        assert result["peak_hour"]["count"] >= result["quiet_hour"]["count"]

    def test_percentages_sum_to_100(self, temporal_pq):
        from corefunc.profile import listening_clock_profile
        result = listening_clock_profile()
        hourly_pct = sum(h["pct"] for h in result["hours"])
        assert abs(hourly_pct - 100.0) < 1.0  # allowing rounding tolerance

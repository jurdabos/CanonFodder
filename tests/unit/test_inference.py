"""
Unit tests for helpers.inference (inference-time feature engineering).
"""
import pickle
from pathlib import Path
import pandas as pd
import pytest
import helpers.inference as inf_mod
from helpers.inference import (
    _compute_disco_features,
    _compute_interaction_features,
    _compute_melo_features,
    _fuzzy_overlap,
    _jaccard,
    _parse_delimited,
    compute_inference_features,
    invalidate_catalogue_cache,
    load_model,
)


class _FakePipeline:
    """Picklable stub standing in for a fitted sklearn Pipeline."""
    feature_names_in_ = ["feat_a", "feat_b", "feat_c"]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensures the catalogue cache is fresh for every test."""
    invalidate_catalogue_cache()
    yield
    invalidate_catalogue_cache()


# ── _parse_delimited ──────────────────────────────────────────────────────────
class TestParseDelimited:
    """Tests for {-delimited string parsing."""

    def test_empty_string_returns_empty_list(self):
        """Returns [] for empty input."""
        assert _parse_delimited("") == []

    def test_none_returns_empty_list(self):
        """Returns [] for None input."""
        assert _parse_delimited(None) == []

    def test_nan_returns_empty_list(self):
        """Returns [] for NaN input."""
        assert _parse_delimited(float("nan")) == []

    def test_single_value_no_delimiter(self):
        """Returns single-element list when no delimiter is present."""
        assert _parse_delimited("Album One") == ["Album One"]

    def test_multiple_values(self):
        """Splits on { delimiter correctly."""
        assert _parse_delimited("A{B{C") == ["A", "B", "C"]

    def test_filters_blank_segments(self):
        """Filters out whitespace-only segments between delimiters."""
        assert _parse_delimited("A{{ {B") == ["A", "B"]


# ── _jaccard ──────────────────────────────────────────────────────────────────
class TestJaccard:
    """Tests for Jaccard index computation."""

    def test_both_empty_returns_zero(self):
        """Returns 0.0 when both sets are empty."""
        assert _jaccard(set(), set()) == 0.0

    def test_identical_sets(self):
        """Returns 1.0 for identical sets."""
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """Returns 0.0 for completely disjoint sets."""
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        """Returns correct ratio for overlapping sets."""
        # {a,b} ∩ {b,c} = {b}, |union| = 3 → 1/3
        assert _jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)

    def test_subset(self):
        """Returns |subset|/|superset| when one is a subset of the other."""
        assert _jaccard({"a"}, {"a", "b"}) == pytest.approx(0.5)


# ── _fuzzy_overlap ────────────────────────────────────────────────────────────
class TestFuzzyOverlap:
    """Tests for fuzzy string matching overlap."""

    def test_empty_list_a(self):
        """Returns (0, 0.0) when list_a is empty."""
        assert _fuzzy_overlap([], ["x"], 80) == (0, 0.0)

    def test_empty_list_b(self):
        """Returns (0, 0.0) when list_b is empty."""
        assert _fuzzy_overlap(["x"], [], 80) == (0, 0.0)

    def test_both_empty(self):
        """Returns (0, 0.0) when both lists are empty."""
        assert _fuzzy_overlap([], [], 80) == (0, 0.0)

    def test_exact_matches_above_threshold(self):
        """Counts exact matches that exceed the threshold."""
        n, ratio = _fuzzy_overlap(["Foo", "Bar"], ["Foo", "Bar"], 80)
        assert n == 2
        assert ratio == pytest.approx(1.0)

    def test_no_matches_below_threshold(self):
        """Returns (0, 0.0) when nothing exceeds the threshold."""
        n, ratio = _fuzzy_overlap(["Aaa"], ["Zzz"], 80)
        assert n == 0
        assert ratio == 0.0

    def test_case_insensitive_fuzzy(self):
        """Matches case-insensitive variants above threshold."""
        n, ratio = _fuzzy_overlap(["Sunset Mission"], ["sunset mission"], 80)
        assert n == 1
        assert ratio > 0.0

    def test_partial_match(self):
        """Counts only items exceeding the threshold."""
        n, ratio = _fuzzy_overlap(
            ["Album A", "Completely Different"],
            ["Album A", "Something Else"],
            80,
        )
        assert n == 1


# ── _compute_disco_features ───────────────────────────────────────────────────
class TestComputeDiscoFeatures:
    """Tests for discography (album) feature computation."""

    def test_both_empty(self):
        """Returns zeros for empty album lists."""
        feats = _compute_disco_features([], [])
        assert feats["disco_fuzzy_album_ratio"] == 0.0
        assert feats["disco_has_fuzzy_album_match"] == 0.0
        assert feats["disco_n_fuzzy_album_matches"] == 0
        assert feats["disco_exact_album_jaccard"] == 0.0
        assert feats["disco_min_album_count"] == 0

    def test_identical_albums(self):
        """Returns full overlap for identical album lists."""
        albums = ["Album A", "Album B"]
        feats = _compute_disco_features(albums, albums)
        assert feats["disco_exact_album_jaccard"] == 1.0
        assert feats["disco_has_fuzzy_album_match"] == 1.0
        assert feats["disco_min_album_count"] == 2

    def test_disjoint_albums(self):
        """Returns zero overlap for completely different albums."""
        feats = _compute_disco_features(["Xxx"], ["Yyy"])
        assert feats["disco_exact_album_jaccard"] == 0.0

    def test_returns_five_features(self):
        """Returns exactly 5 disco features, all prefixed with 'disco_'."""
        feats = _compute_disco_features(["A"], ["B"])
        assert len(feats) == 5
        assert all(k.startswith("disco_") for k in feats)

    def test_asymmetric_lists(self):
        """Handles lists of different lengths correctly."""
        feats = _compute_disco_features(["A", "B", "C"], ["A"])
        assert feats["disco_min_album_count"] == 1
        assert feats["disco_exact_album_jaccard"] == pytest.approx(1 / 3)


# ── _compute_melo_features ────────────────────────────────────────────────────
class TestComputeMeloFeatures:
    """Tests for melography (track) feature computation."""

    def test_both_empty(self):
        """Returns zeros for empty track lists."""
        feats = _compute_melo_features([], [])
        assert feats["melo_fuzzy_track_ratio"] == 0.0
        assert feats["melo_has_fuzzy_track_match"] == 0.0
        assert feats["melo_n_fuzzy_track_matches"] == 0
        assert feats["melo_exact_track_jaccard"] == 0.0
        assert feats["melo_min_track_count"] == 0

    def test_identical_tracks(self):
        """Returns full overlap for identical track lists."""
        tracks = ["Prowler", "Midnight Walker"]
        feats = _compute_melo_features(tracks, tracks)
        assert feats["melo_exact_track_jaccard"] == 1.0
        assert feats["melo_has_fuzzy_track_match"] == 1.0

    def test_returns_five_features(self):
        """Returns exactly 5 melo features, all prefixed with 'melo_'."""
        feats = _compute_melo_features(["A"], ["B"])
        assert len(feats) == 5
        assert all(k.startswith("melo_") for k in feats)

    def test_disjoint_tracks(self):
        """Returns zero exact overlap for different tracks."""
        feats = _compute_melo_features(["Aaa"], ["Zzz"])
        assert feats["melo_exact_track_jaccard"] == 0.0


# ── _compute_catalogue_features ───────────────────────────────────────────────
class TestComputeCatalogueFeatures:
    """Tests for combined catalogue feature computation."""

    def test_unknown_names_return_zeros(self, monkeypatch):
        """Returns zeros when neither name exists in the cache."""
        monkeypatch.setattr(inf_mod, "_catalogue_cache", {
            "albums": {}, "tracks": {},
        })
        feats = inf_mod._compute_catalogue_features("Unknown_A", "Unknown_B")
        assert feats["disco_min_album_count"] == 0
        assert feats["melo_min_track_count"] == 0

    def test_known_names_with_overlap(self, monkeypatch):
        """Returns positive features when artists share catalogue entries."""
        monkeypatch.setattr(inf_mod, "_catalogue_cache", {
            "albums": {"A": ["Album1", "Album2"], "B": ["Album1"]},
            "tracks": {"A": ["Track1"], "B": ["Track1", "Track2"]},
        })
        feats = inf_mod._compute_catalogue_features("A", "B")
        assert feats["disco_has_fuzzy_album_match"] == 1.0
        assert feats["melo_has_fuzzy_track_match"] == 1.0

    def test_returns_ten_features(self, monkeypatch):
        """Returns exactly 10 catalogue features (5 disco + 5 melo)."""
        monkeypatch.setattr(inf_mod, "_catalogue_cache", {
            "albums": {}, "tracks": {},
        })
        feats = inf_mod._compute_catalogue_features("X", "Y")
        assert len(feats) == 10
        disco = [k for k in feats if k.startswith("disco_")]
        melo = [k for k in feats if k.startswith("melo_")]
        assert len(disco) == 5
        assert len(melo) == 5

    def test_one_known_one_unknown(self, monkeypatch):
        """Returns zeros when only one name has catalogue data."""
        monkeypatch.setattr(inf_mod, "_catalogue_cache", {
            "albums": {"A": ["Album1"]}, "tracks": {"A": ["Track1"]},
        })
        feats = inf_mod._compute_catalogue_features("A", "Z")
        assert feats["disco_exact_album_jaccard"] == 0.0
        assert feats["melo_exact_track_jaccard"] == 0.0


# ── _compute_interaction_features ─────────────────────────────────────────────
class TestComputeInteractionFeatures:
    """Tests for pairwise interaction feature computation."""

    def test_six_scores_produce_30_features(self):
        """Returns C(6,2)*2 = 30 interaction features for 6 similarity scores."""
        base = {
            "ratio": 0.8, "partial_ratio": 0.9,
            "token_sort_ratio": 0.7, "token_set_ratio": 0.85,
            "WRatio": 0.75, "QRatio": 0.65,
        }
        feats = _compute_interaction_features(base)
        assert len(feats) == 30

    def test_diff_and_product_values(self):
        """Computes correct diff and product for a known pair."""
        base = {"ratio": 0.8, "partial_ratio": 0.6}
        feats = _compute_interaction_features(base)
        diff_keys = [k for k in feats if "minus" in k]
        prod_keys = [k for k in feats if "mul" in k]
        assert len(diff_keys) == 1
        assert len(prod_keys) == 1
        assert feats[diff_keys[0]] == pytest.approx(0.2)
        assert feats[prod_keys[0]] == pytest.approx(0.48)

    def test_fewer_scores_fewer_features(self):
        """Returns C(2,2)*2 = 2 features when base has only 2 sim scores."""
        base = {"ratio": 0.5, "WRatio": 0.4}
        feats = _compute_interaction_features(base)
        assert len(feats) == 2

    def test_no_sim_scores_returns_empty(self):
        """Returns empty dict when base has no recognised sim scores."""
        feats = _compute_interaction_features({"some_other_feature": 1.0})
        assert feats == {}

    def test_single_score_returns_empty(self):
        """Returns empty dict when only one sim score is present."""
        feats = _compute_interaction_features({"ratio": 0.9})
        assert feats == {}

    def test_feature_names_are_unique(self):
        """Produces unique column names for all 30 features."""
        base = {
            "ratio": 0.1, "partial_ratio": 0.2,
            "token_sort_ratio": 0.3, "token_set_ratio": 0.4,
            "WRatio": 0.5, "QRatio": 0.6,
        }
        feats = _compute_interaction_features(base)
        assert len(feats) == len(set(feats.keys()))


# ── compute_inference_features ────────────────────────────────────────────────
class TestComputeInferenceFeatures:
    """Tests for the main compute_inference_features entry point."""

    def test_returns_dict_of_numerics(self, monkeypatch):
        """Returns a flat dict with numeric values."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {f"cat_{i}": 0.0 for i in range(10)},
        )
        feats = compute_inference_features("Bohren", "Bohren")
        assert isinstance(feats, dict)
        assert all(isinstance(v, (int, float)) for v in feats.values())

    def test_none_inputs_handled(self, monkeypatch):
        """Handles None inputs without raising."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {},
        )
        feats = compute_inference_features(None, None)
        assert isinstance(feats, dict)
        assert "ratio" in feats

    def test_empty_string_inputs(self, monkeypatch):
        """Handles empty string inputs without raising."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {},
        )
        feats = compute_inference_features("", "")
        assert isinstance(feats, dict)

    def test_includes_base_features(self, monkeypatch):
        """Includes base pairwise features from compute_pair_features."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {},
        )
        feats = compute_inference_features("Alice", "Bob")
        assert "ratio" in feats
        assert "WRatio" in feats
        assert "jaro_winkler" in feats

    def test_includes_interaction_features(self, monkeypatch):
        """Includes interaction (diff/product) features."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {},
        )
        feats = compute_inference_features("Alice", "Bob")
        interaction_keys = [k for k in feats if "minus" in k or "mul" in k]
        assert len(interaction_keys) > 0

    def test_includes_catalogue_features(self, monkeypatch):
        """Includes catalogue features when provided."""
        mock_cat = {"disco_fuzzy_album_ratio": 0.5, "melo_fuzzy_track_ratio": 0.3}
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: mock_cat,
        )
        feats = compute_inference_features("X", "Y")
        assert "disco_fuzzy_album_ratio" in feats
        assert feats["disco_fuzzy_album_ratio"] == 0.5

    def test_approximate_feature_count(self, monkeypatch):
        """Returns approximately 63 features (23 base + 30 interaction + 10 catalogue)."""
        monkeypatch.setattr(
            inf_mod, "_compute_catalogue_features",
            lambda a, b: {f"cat_{i}": 0.0 for i in range(10)},
        )
        feats = compute_inference_features("Foo", "Bar")
        # 23 base + 30 interaction + 10 catalogue = 63
        assert len(feats) >= 60


# ── invalidate_catalogue_cache ────────────────────────────────────────────────
class TestInvalidateCatalogueCache:
    """Tests for cache invalidation."""

    def test_clears_populated_cache(self, monkeypatch):
        """Sets _catalogue_cache to None when it was populated."""
        monkeypatch.setattr(inf_mod, "_catalogue_cache", {"albums": {}, "tracks": {}})
        invalidate_catalogue_cache()
        assert inf_mod._catalogue_cache is None

    def test_idempotent_on_none(self):
        """Clears without error when cache is already None."""
        invalidate_catalogue_cache()
        assert inf_mod._catalogue_cache is None


# ── load_model ────────────────────────────────────────────────────────────────
class TestLoadModel:
    """Tests for model loading."""

    def test_missing_file_raises(self, tmp_path):
        """Raises FileNotFoundError when pickle file is absent."""
        with pytest.raises(FileNotFoundError, match="Model pickle not found"):
            load_model(tmp_path / "nonexistent.pkl")

    def test_loads_valid_pickle(self, tmp_path):
        """Loads a valid pickle and returns the pipeline object."""
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as fh:
            pickle.dump(_FakePipeline(), fh)
        result = load_model(model_path)
        assert hasattr(result, "feature_names_in_")
        assert len(result.feature_names_in_) == 3

    def test_default_path_used(self, monkeypatch, tmp_path):
        """Uses MODEL_PATH when no path argument is given."""
        monkeypatch.setattr(inf_mod, "MODEL_PATH", tmp_path / "absent.pkl")
        with pytest.raises(FileNotFoundError):
            load_model()


# ── _load_catalogue_cache ─────────────────────────────────────────────────────
class TestLoadCatalogueCache:
    """Tests for catalogue cache building."""

    def test_returns_cached_when_populated(self, monkeypatch):
        """Returns cached values without rebuilding."""
        fake = {"albums": {"A": ["x"]}, "tracks": {"A": ["t"]}}
        monkeypatch.setattr(inf_mod, "_catalogue_cache", fake)
        albums, tracks = inf_mod._load_catalogue_cache()
        assert albums == {"A": ["x"]}
        assert tracks == {"A": ["t"]}

    def test_empty_scrobbles_returns_empty(self, monkeypatch):
        """Returns empty dicts when no scrobble data exists."""
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", Path("/tmp/nonexistent.parquet"))
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: None)
        albums, tracks = inf_mod._load_catalogue_cache()
        assert albums == {}
        assert tracks == {}

    def test_empty_scrobble_df_returns_empty(self, monkeypatch):
        """Returns empty dicts when scrobble DataFrame is empty."""
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", Path("/tmp/nonexistent.parquet"))
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: pd.DataFrame())
        albums, tracks = inf_mod._load_catalogue_cache()
        assert albums == {}
        assert tracks == {}

    def test_scrobble_only_fallback(self, monkeypatch):
        """Builds lookups from scrobble data when disco file is absent."""
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", Path("/tmp/nonexistent.parquet"))
        scrobbles = pd.DataFrame({
            "artist_name": ["ArtistA", "ArtistA", "ArtistB"],
            "album_title": ["Album1", "Album2", "Album3"],
            "track_title": ["Track1", "Track2", "Track3"],
            "artist_mbid": [None, None, None],
        })
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: scrobbles)
        albums, tracks = inf_mod._load_catalogue_cache()
        assert "ArtistA" in albums
        assert "ArtistB" in albums
        assert set(albums["ArtistA"]) == {"Album1", "Album2"}
        assert tracks["ArtistB"] == ["Track3"]

    def test_disco_preferred_over_scrobble(self, monkeypatch, tmp_path):
        """Uses disco data when the artist has a matching MBID in the disco file."""
        disco_path = tmp_path / "disco.parquet"
        proper_mbid = "a4074512-87e0-4820-b609-0c4a18142a70"
        disco_df = pd.DataFrame({
            "mbid": [proper_mbid],
            "albums_str": ["DiscoAlbum1{DiscoAlbum2"],
            "tracks_str": ["DiscoTrack1{DiscoTrack2"],
        })
        disco_df.to_parquet(disco_path, index=False)
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", disco_path)
        scrobbles = pd.DataFrame({
            "artist_name": ["ArtistA"],
            "album_title": ["ScrobbleAlbum"],
            "track_title": ["ScrobbleTrack"],
            "artist_mbid": [proper_mbid],
        })
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: scrobbles)
        albums, tracks = inf_mod._load_catalogue_cache()
        # Disco data should take precedence
        assert albums["ArtistA"] == ["DiscoAlbum1", "DiscoAlbum2"]
        assert tracks["ArtistA"] == ["DiscoTrack1", "DiscoTrack2"]

    def test_scrobble_fallback_when_no_mbid(self, monkeypatch, tmp_path):
        """Falls back to scrobble data when artist has no MBID."""
        disco_path = tmp_path / "disco.parquet"
        disco_df = pd.DataFrame({
            "mbid": ["some-other-mbid-that-wont-match-00"],
            "albums_str": ["DiscoAlbum"],
            "tracks_str": ["DiscoTrack"],
        })
        disco_df.to_parquet(disco_path, index=False)
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", disco_path)
        scrobbles = pd.DataFrame({
            "artist_name": ["NoMbidArtist"],
            "album_title": ["FallbackAlbum"],
            "track_title": ["FallbackTrack"],
            "artist_mbid": [None],
        })
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: scrobbles)
        albums, tracks = inf_mod._load_catalogue_cache()
        assert albums["NoMbidArtist"] == ["FallbackAlbum"]
        assert tracks["NoMbidArtist"] == ["FallbackTrack"]

    def test_blank_album_titles_excluded(self, monkeypatch):
        """Excludes blank and whitespace-only album titles from lookups."""
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", Path("/tmp/nonexistent.parquet"))
        scrobbles = pd.DataFrame({
            "artist_name": ["A", "A", "A"],
            "album_title": ["Good Album", "", "   "],
            "track_title": ["T1", "T2", "T3"],
            "artist_mbid": [None, None, None],
        })
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: scrobbles)
        albums, _ = inf_mod._load_catalogue_cache()
        assert albums["A"] == ["Good Album"]

    def test_populates_global_cache(self, monkeypatch):
        """Sets _catalogue_cache so subsequent calls skip rebuilding."""
        monkeypatch.setattr(inf_mod, "SOLO_DISCO_PQ", Path("/tmp/nonexistent.parquet"))
        monkeypatch.setattr(inf_mod, "read_scrobble_df", lambda: None)
        inf_mod._load_catalogue_cache()
        assert inf_mod._catalogue_cache is not None
        assert "albums" in inf_mod._catalogue_cache
        assert "tracks" in inf_mod._catalogue_cache

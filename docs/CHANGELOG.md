# Changelog

All notable changes to c9r (CanonFodder) in reverse chronological order.                                                                                   

---

## 2026-06-13: Pin CI to Python 3.12 (unblocks lint workflow)
- .python-version — added with `3.12`; the new lint.yml runs `uv python install` without an argument, which previously picked CPython 3.14.6 on the GitHub runner.
- pyproject.toml — tightened `requires-python` from `>=3.12` to `>=3.12,<3.14`. `ruamel-yaml-clib==0.2.12` (transitive via `prefect`) does `from ast import Str` in its setup.py, which 3.12 removed and 3.14 still doesn't reinstate; without the upper bound, any contributor or future CI job running bare `uv python install` would re-hit the same build failure.
- uv.lock, requirements.txt — re-locked and re-exported against the new constraint (no transitive version changes; only the metadata bound moved).

---

## 2026-06-12: acidbase CI baseline + `c9r push` integration
- pyproject.toml — extended `[tool.ruff]` to the canonical a6a config (extend-exclude, `select = ["E", "W", "F", "I"]`, per-file-ignores for tests/scripts/__init__, formatter block); added `acidbase` to the dev dependency group from the public mirror (`git+https://github.com/jurdabos/acidbase`).
- main.py — attached the shared acidbase `push_command` as `uv run c9r push` (guarded import keeps the CLI usable when acidbase is absent).
- .github/workflows/lint.yml — new canonical lint workflow: ruff check + format-check via uv, plus the MIT gitleaks CLI pinned at 8.30.1 (full-history scan, `--redact`).
- .github/workflows/test.yml — removed the duplicate lint job (linting now lives in lint.yml); dropped the `needs: lint` gates.
- .gitleaks.toml — new canonical secret-scan config (extends default ruleset, ecosystem allowlist).
- .pre-commit-config.yaml — added ruff + ruff-format hooks; bumped gitleaks hook to v8.30.1; kept repo-specific uv-lock/uv-export hooks.
- Codebase — 172 ruff autofixes (mostly import sorting) and repo-wide `ruff format` (33 files, line-ending/quote normalisation); wrapped 3 over-long lines in corefunc/canon/tcn_trainer.py and main.py; marked the deliberate post-constant imports in helpers/io.py with `noqa: E402`.
- tests/unit/test_qa_ml_extra.py — fixed pre-existing `test_verify_mlflow` failure under mlflow ≥ 3.13 by also mocking the function-local mlflow import (the real call hit the deprecated ./mlruns file store).
- requirements.txt — re-exported (`uv export --frozen`) to include acidbase and the click bump.
- README.md — documented the new `push` command under CLI Usage.

---

## 2026-06-12: Airflow dependency fully removed
- pyproject.toml — dropped `apache-airflow>=3.2.0` from dependencies; Prefect (migrated 2025-09) is the sole orchestrator.
- uv.lock, requirements.txt — re-locked and re-exported via `uv export --frozen --output-file=requirements.txt`; 61 Airflow-related packages removed from the dependency tree.
- GitHub: dismissed 17 stale Dependabot alerts (reason `not_used`) that were pinned to the deleted `requirements-airflow.txt` manifest — GitHub's dependency graph never auto-resolves alerts when an entire manifest file is deleted.

---

## 2026-02-28: Prediction log schema and feature quantile drift detection
- corefunc/canon/workflow.py — `_log_predictions()` now includes a `features_json` column (JSON-serialised ~63-feature dict from `compute_inference_features`) alongside `timestamp`, `variant_a`, `variant_b`, `probability`. Every `canon machine` run now persists the full feature vector for downstream drift detection.
- corefunc/qa.py — `qa_predictions()` extended with feature quantile drift detection via new `_feature_quantile_drift()` helper. Parses `features_json`, computes per-feature medians for baseline and recent windows, and flags features whose median shift exceeds `DRIFT_FEATURE_QUANTILE_THRESHOLD` (0.15). The report now includes a `feature_quantiles` dict and any feature-level warnings.
- tests/unit/test_qa_ml_extra.py — all `TestQaPredictions` fixtures updated to include the full 5-column schema (`timestamp`, `variant_a`, `variant_b`, `probability`, `features_json`). `test_drift_detected` now verifies feature quantile warnings.
- tests/unit/test_enrich_extra.py — `test_log_predictions` updated to include `features_json` and assert on the full column set.
- Deleted stale `PQ/predictions_log.parquet` (had only 2 columns: `timestamp`, `probability`). File regenerates with the correct schema on next `canon machine` run.

---

## 2026-02-28: Documentation overhaul
- README.md — updated tech stack (LightGBM primary, ListenBrainz recommended), added all missing commands (`tune`, `schema`, `migrate-scrobbles`, `train run`/`train tcn`), documented every CLI option with default values, added drift detection and schema management sections.
- WARP.md — full rewrite from MySQL/SQLAlchemy/Alembic-era content to match current Parquet+DuckDB+Click+LightGBM architecture. Covers data flow, ML pipeline, drift detection, schema management, directory layout.

---

## 2026-02-28: New test files
- tests/unit/test_trainer_ml.py — 73 tests covering trainer.py utility functions (parsing, Jaccard, fuzzy overlap, proportional features, cross-tier interactions), feature dispatch, evaluation helpers, CV loop, GPU fallback, MLflow helpers, and all data builders (_dispatch_data_build, mixed, group, mbdb, dbscan, etc.)
- tests/unit/test_tcn_tuner_runner.py — 27 tests covering SiameseTCN/HybridTCN forward passes, NamePairDataset/HybridDataset, prediction/evaluation helpers, training loops (smoke tests), tuner search spaces + objective function, and experiment_runner CV/GPU fallback
- tests/unit/test_final_coverage.py — 22 tests covering device.py GPU probe, _build_feature_sep/mbdb_max data builders, unicode script detection, script mismatch flags, and legacy compatibility shims

---

## 2026-02-28: LightGBM as default model across the repo
- corefunc/canon/model.py (legacy module): Replaced XGBClassifier with LGBMClassifier, switched model artefact paths to lgbm_legacy.pkl / lgbm_columns.json, updated hyperparams (using is_unbalance=True instead of scale_pos_weight), changed save to pickle, removed unused get_device import.
- corefunc/canon/trainer.py (unified training pipeline): Added DEFAULT_MODELS = ["LightGBM"] so c9r train run trains only LightGBM by default (override with --models). Changed catalogue_design default from "presence" to "proportional" to use the 10-feature proportional catalogue (matching inference features). Added cat_design parameter to compute_all_features().
- corefunc/canon/tuner.py (Optuna tuning): Added _DEFAULT_TUNE_MODELS = ["LightGBM"] so c9r tune tunes only LightGBM by default. Updated compute_all_features() calls to use cat_design="proportional".
- main.py (CLI): Updated --catalogue-design default to "proportional", updated --models help text for both train run and tune to say (default: LightGBM).
- tests/unit/test_canon.py: Updated assertions from XGBoost step names/report headers to LightGBM equivalents.

The serving path (helpers/inference.py, corefunc/model_server.py) and discovery path (corefunc/canon/workflow.py) already pointed to lightgbm_best.pkl with proportional catalogue features — no changes needed there.

---

## 2026-02-27: Versioned Parquet schema feature implemented
- helpers/schema.py — schema registry, version tracking, metadata stamp/read, validation, migration framework with v0→v1 migrations
- helpers/io.py — auto-stamps on write, warns on stale / raises on future version
- main.py — *c9r schema show* and *c9r schema migrate* CLI commands
- tests/unit/test_schema.py — 15 tests covering all schema functionality

---

## 2026-02-27: Year-partitioned scrobble implementation
- Core I/O (helpers/io.py): Added SCROBBLE_PQ_DIR, plus helpers read_scrobble_df(), scrobble_duckdb_from(), dump_scrobble_df(), scrobble_data_exists(), and migrate_scrobble_to_partitioned(). Rewrote ingest_scrobbles() to write PQ/scrobble/year=YYYY/part.parquet partitions. All helpers fall back to the legacy single file for backward compat.
- Consumers updated (10 files): DuckDB queries (query.py, profile.py, canon/workflow.py) use scrobble_duckdb_from(). Pandas readers (enrich.py, data_cleaning.py, qa.py, trainer.py, mb_local.py, inference.py, lfAPI.py) use read_scrobble_df()/dump_scrobble_df().
- CLI (main.py): Added *c9r migrate-scrobbles [--remove-legacy]* command.
- Tests: Updated conftest.py to patch SCROBBLE_PQ_DIR, fixed assertions in test_io.py, test_data_cleaning.py, and test_workflow.py. 454/455 tests pass; the one failure is a pre-existing TestTrainCommand issue.

---

## 2026-02-27: Prefect flow updated to cover functional reqs from SRD
- propagate_avc_task (FR-05) — runs after canonise_batch to apply decided AVC mappings to artist_info.parquet, so canonisation doesn't just flag variants but actually applies them.
- augment_gs_task (FR-06) — refreshes gold-standard pairs from the local MB mirror; skips gracefully if the Docker mirror is down.
- retrain_model_task (FR-07) — retrains the canonisation model on current data; skips if there are fewer than 20 decided AVC rows.
The flow now covers FR-01 through FR-07 and FR-10 directly. FR-08 (ASGI server) and FR-09 (interactive BI/profiling) are inherently outside a batch pipeline — they're served by c9r serve and c9r dashboard respectively.

## 2026-02-27: SRD requirement around backoff fulfilled
- HTTP/client.py — sleep(2) → sleep(min(2 ** attempt, 120)) giving 2s, 4s, 8s … capped at 120s. Also removed the dead attempt = attempt + 1 line.
- HTTP/mbAPI.py — stop_after_attempt(5) → 8; raised max wait from 10s to 60s so later retries aren't artificially compressed.
- flows/cf_ingest.py — constant retry_delay_seconds=30 → exponential_backoff(backoff_factor=10) (10s, 20s, 40s, 80s …) on both fetch_scrobbles and enrich_artists, keeping the existing jitter.

---

## 2026-02-27: Updates to ML reenactment logic

1. trainer.py — new data builders (_build_dbscan_capped_training_data, _build_feature_sep_training_data), feature-separation logic, expanded run_training() params
2. tcn_trainer.py (new) — consolidated Siamese TCN + Hybrid TCN module with run_tcn_training() entry point
3. main.py — new --data-source=dbscan-capped, --cluster-cap, --neg-ratio, --feature-strategy, --neg-matching, --neg-count options on train run; new train tcn subcommand with --model, --epochs, --batch-size, --lr, --patience, --experiment, --run-name
4. __init__.py — exports run_tcn_training

---

## 2026-02-27: Update best-model choosing score to composite 0.4 × HiP_P + 0.3 × HiP_F1 + 0.3 × AUC all through the repo, e.g.:
- Experiment scripts (exp6, 7, 7b, 8, 11, 12, 13, 14, 15): replaced max(..., key=x["auc"]) with the composite c9r score
- corefunc/canon/trainer.py: same (dict-of-dicts style)
- corefunc/canon/tuner.py run_tuning: replaced opt_prec selection with c9r score
- corefunc/canon/tuner.py save_best_historical_models: replaced AUC-based comparison with c9r score computed from MLflow metrics (hiprec_precision, hiprec_f1, auc)

---

## 2026-02-27: c9r train options expanded
- main.py — c9r train is now a command group. c9r train run exposes 11 new options beyond the existing 5:  --data-source, --split, --test-source, --features, --catalogue-source, --catalogue-design, --group-features, --wratio-lower, --wratio-upper, --experiment, --include-composites.
- corefunc/canon/trainer.py — run_training() accepts all new parameters and dispatches to the correct data builder, feature computation, and catalogue design. New functions:
  - _dispatch_data_build() — routes to the right data source
  - _build_mbdb_training_data() (Exp 4–5), _build_dbscan_training_data() (Exp 6–8), _build_mbdb_max_training_data() (Exp 11–12), _build_mixed_training_data() (Exp 1–3), _build_avc_group_split() (Exp 13), _build_avc_full_test() (cross-domain test)
  - _load_scrobble_only_lookups() — scrobble-only catalogue (Exp 13)
  - _add_proportional_catalogue_features() — Jaccard/fuzzy_ratio catalogue design (Exp 12–13)
  - _compute_features_for_split() — dispatches base / interaction / full features
  - _add_base_features_only() — base 23 without interaction terms

---

## 2026-02-27: Integration added
- New file: helpers/inference.py — Single-pair feature engineering for inference. Produces all 63 raw features (base 23 + interaction 30 + catalogue 10) so any pruned model pickle can select its subset via feature_names_in_. Includes session-level catalogue cache (MBDB solo-credit + scrobble fallback) and load_model() helper. Smoke test confirms all 41 features the Exp 15 LightGBM expects are covered, with correct predictions (Beatles/The Beatles → p=0.994, Mozart/Metallica → p≈0.000).
- Updated corefunc/model_server.py — Replaced XGBoost load_model() + separate xgb_columns.json with a single pickle.load() of the self-contained Pipeline. The pipeline carries its own column order in feature_names_in_.
- Updated corefunc/canon/workflow.py — discover_candidates() now uses compute_inference_features() for full 41-feature scoring instead of the old 6-feature fuzzy_scores(). Loads the pickle directly when no model is passed. Every prediction is logged to PQ/predictions_log.parquet (timestamp, pair, probability). Model probability stored in AVC comment field as p=0.XXXX. Results sorted by descending probability.
- Updated main.py — canon machine simplified: loads pickle directly, no MLflow run selection UI. Shows probability alongside each candidate. canon human now parses p=X.XXXX from the comment field, sorts groups by descending probability, and displays [model p=X.XXXX] in the review prompt.
- Updated flows/cf_ingest.py — Added canonise_batch as the 5th Prefect task. Loads the pickle in-process (no HTTP dependency). Gracefully skips when the pickle is missing. Flow returns flagged_for_review and skipped counts.
- Updated corefunc/qa.py — Added qa_predictions() for drift detection. Compares baseline vs recent prediction windows on mean probability shift (threshold: 0.10) and ambiguous-band proportion growth (threshold: 2×).

---

## 2026-02-27: c9r tune added
### Historical model pickles saved (5 best-AUC models from Exp 14-16):
- extratrees_best.pkl — AUC=0.980, opt_P=0.837
- lightgbm_best.pkl — AUC=0.979, opt_P=0.923
- xgboost_best.pkl — AUC=0.977, opt_P=0.864
- randomforest_best.pkl — AUC=0.980, opt_P=0.826
- gradientboosting_best.pkl — AUC=0.965, opt_P=0.780
### Optuna tuning pipeline verified end-to-end (5-trial smoke test on LightGBM):
- Best trial scored 0.944 (mean CV precision=0.944, worst fold=0.909)
- Tuned model: lightgbm_tuned.pkl — AUC=0.967, opt_P=0.919
### New files/changes:
- corefunc/canon/tuner.py — Optuna tuning module with precision-biased objective (mean_precision - 0.5 * penalty), search spaces for LightGBM/XGBoost/ExtraTrees, post-tuning retrain with full 5-fold CV + MLflow logging, and save_best_historical_models() for pickle export.
- main.py — new c9r tune command with --models, --trials, --folds, --min-precision, --catalogue/--no-catalogue.
- corefunc/canon/__init__.py — exports run_tuning.

---

## 2026-02-27: Tier 3 of c9r enrich is added (for enrich --rebuild)
### Example run:
- Tier 1: 15,805 artists by MBID (13,118 unique MBIDs)
- Tier 2: 6 artists by exact name match
- Tier 3: 37 artists by catalogue overlap
- Stubs: 4,953 truly unresolved
- Coverage: 76.2% of artist_info entries have MBID; 90.1% of scrobble rows have MBID
- Backfill: 678 MBIDs propagated into scrobble.parquet

---

## 2026-02-25 – Adding country_code filter list to profile uc
- corefunc/profile.py — user_country_medal_profile now accepts an optional country_codes: list[str] | None. When provided, it filters the scrobble counts to only those ISO-2 codes (case-insensitive) instead of using the top-ucn by volume.
- main.py — Added -c option to profile uc plus a _parse_country_codes helper that accepts both HU,ES,DK and (HU, ES, DK) formats.
- tests — 3 new unit tests (filter-to-specified, unknown-code drop, case-insensitivity) and 2 new CLI tests (single code, parenthesised multi-code).
- Usage:
uv run c9r profile uc -n 3 -c "(HU, ES, DK, TH, VN, CN, NZ, CO, IE)"
uv run c9r profile uc -n 3 -s artist                     # artists only
uv run c9r profile uc -n 3 -s "(artist, track)"          # artists + tracks
uv run c9r profile uc -n 3 -c "(HU, ES)" -s album        # albums only, filtered countries

---

## 2026-02-25 – Startup of ML capabilities
### Added files:
- helpers/device.py — GPU probe that tries XGBoost CUDA, caches result, falls back to CPU. Every model constructor calls get_device().
- helpers/features.py — Three-tier compute_pair_features(a, b) returning 23 features: 10 whole-string (6 existing RapidFuzz + Levenshtein, Jaro-Winkler, length ratio, abs length diff), 5 token-level (count diff, Jaccard, shared ratio, LCS, Kendall τ displacement), 8 character-level (bigram/trigram Jaccard, edit op breakdown, shared prefix/suffix, script mismatch flag).
- corefunc/canon/experiment_runner.py — Multi-model orchestrator that trains 8 models (XGBoost GPU, Random Forest, Extra Trees, LightGBM, GradientBoosting, VotingClassifier, StackingClassifier, BaggingXGB) in a single MLflow parent run with nested child runs, 5-fold stratified CV with per-fold logging, GPU fallback safety, SHAP/confusion matrix/feature importance artifacts.
### Modified files:
- corefunc/canon/model.py — Now uses 23-feature compute_pair_features instead of the old 6-score path, and device=get_device() for GPU-accelerated XGBoost.
- helpers/experiment.py — Added log_confusion_matrix(), log_feature_importance(), and log_shap_summary() artifact helpers.
- main.py — Added c9r canon experiment CLI command; fixed c9r mlflow-ui to pass --workers 1 (critical for WSL2 stability).
- pyproject.toml / uv.lock — Added lightgbm, shap, optuna dependencies.

---

## 2026-02-25 – User country to entities tables
- helpers/query.py — user_country_top_entities(top_n): runs 3 DuckDB queries (artists, albums, tracks) each with the uc interval join + canonical name resolution + ROW_NUMBER() OVER (PARTITION BY country_code). Returns a dict of 3 DataFrames, ranked per country.
- corefunc/profile.py — user_country_medal_profile(top_n, ucn): gets the top-ucn countries from user_country_scrobble_counts(), fetches medal data from the query function, resolves country names from c.parquet, and assembles a structured dict with per-country artist/album/track medal lists. Also extracted _country_name_map() helper.
- main.py — profile uc CLI command with -n (medal count, default 3) and --ucn (user-country count, default 5). Output mirrors dashboard yearly style — each country gets a section header with scrobble count, then Artists/Albums/Tracks sub-sections with GOLD/SILVER/BRONZE labels.
- Tests — 7 query tests (TestUserCountryTopEntities), 6 profile tests (TestUserCountryMedalProfile), 4 CLI tests (TestProfileUc).

---

## 2026-02-25 – Population-weighted country ranking and user country rankings
- conftest.py — added UC_PQ to q_mod patch list so the new query function's path constant gets redirected to the temp directory during tests
- main.py — added two CLI commands:
  - profile population — displays artist-origin countries ranked by absolute scrobble count and by per-capita (scrobbles per million population)
  - profile where — displays the user's physical country at scrobble time, ranked by scrobble count with share percentages
- test_query.py — added TestUserCountryScrobbleCounts (5 tests) covering interval-join logic, correct DE/HU counts, and empty-data edge cases
- test_profile.py — added TestPopulationVsScrobbles (5 tests) and TestUserCountryProfile (5 tests) covering rankings, sorting, percentages, country names, and missing-data paths
- test_cli.py — added TestProfilePopulation (3 tests) and TestProfileWhere (3 tests) covering error states, formatted output, and data display
- Fixed pre-existing lint issues: removed unused imports (PQ_DIR, C_PQ, UC_PQ, pytest) and extraneous f-prefixes

---

## 2026-02-25 – Time series backbone and temporal analysis integration
- Query backbone (helpers/query.py) — 4 new DuckDB functions:
  - monthly_scrobble_counts() — year/month aggregation for the temporal backbone
  - yearly_top_n_artists(top_n) — per-year ranked artists with canonical name resolution
  - listening_clock(granularity) — hour-of-day and day-of-week bucketing
  - daily_scrobble_dates() — distinct scrobble dates for streak computation
- Profile analytics (corefunc/profile.py) — 4 new functions:
  - monthly_summary() — per-calendar-month stats (min/max/mean/total) with strongest/weakest identification, mirroring the ridgeline insight
  - yearly_top_artists_profile(top_n) — gold/silver/bronze per year, mirroring the stacked bar chart from pics/
  - streak_analysis() — longest streak, current streak, longest gap, total active days
  - listening_clock_profile() — hourly and weekday distributions with peak/quiet identification
- CLI commands (main.py):
  - c9r profile timeline — monthly summary table
  - c9r profile streaks — streak/gap stats
  - c9r profile clock — when-do-you-listen with bar visualisation
  - c9r dashboard yearly — year-by-year top artists in GOLD/SILVER/BRONZE format

---

## 2026-02-25 – DuckDB CTE for joining scrobble with artist_info for analytics purposes
- Core change: _canonical_cte() in helpers/query.py — a DuckDB CTE that builds a variant→canonical mapping from artist_info.aliases. When artist_info is absent, returns an empty CTE so queries degrade gracefully.
- Updated queries in helpers/query.py — top_artists, unique_artists, top_albums, top_tracks, recent_scrobbles, artist_country_stats all now LEFT JOIN through canonical_map and use COALESCE(cm.canonical_name, s.artist_name). scrobble_count and scrobbles_between stay raw (they don't aggregate by artist).
- Updated queries in corefunc/profile.py — overview_stats, trusted_companions, country_breakdown all resolve through canonical names. top_artists_profile now uses artist_info aliases instead of AVC for its canonize=True path. variant_candidates intentionally stays raw (its job is to find unresolved variants).
- Updated main.py — profile top --canonized label and fallback message updated.
- Tests — 4 new canonization tests in test_query.py, 2 updated tests in test_profile.py (alias-based instead of AVC-based).

---

## 2026-02-25 – c9r profile top --custom addition
- _parse_rank_ranges() — parses a string like "(1, 5), (27, 29)" into sorted (start, end) tuples with validation (start ≥ 1, start ≤ end).
- _echo_ranged_entries() — prints filtered entries for the given ranges, inserting ... between discontinuous ranges.
- --custom flag on profile top — when provided, fetches enough rows to cover the max rank and displays only the selected ranges. Works with --canonized too.
- 9 new tests covering the parser (valid/invalid inputs) and the CLI integration (single range, multi-range with ellipsis, invalid input).

---

## 2026-02-25 – c9r canon sketched out

### Added
- corefunc/avc_seed.py — MySQL dump parser that seeded PQ/avc.parquet with 488 rows (177 linked, 311 skipped) from the old gold standard
- tests/unit/test_canonize.py — 15 unit tests covering all functions (all passing)

### Changed
- corefunc/canon.py — business logic for all four subcommands: avc_summary(), propagate_avc(), undecided_rows() / update_avc_decision(), list_mlflow_runs() / load_run_model() / discover_candidates() / write_new_candidates()
- main.py — replaced the placeholder canonise and review commands with the canon group housing:
- c9r canon avc show (with --decided / --undecided / --last N filters, tabular output with variants column last)
- c9r canon avc propagate (renames artist_info rows based on avc.parquet (NOT the training file gs_mb.parquet, appends to aliases)
  - It only touches artist_info, the dimension table, as canonisation belongs on the dimension — scrobble rows then resolve through artist_info at query time.
- c9r canon avc seed <sql_path> (one-time migration command)
- c9r canon human (interactive review of NULL-to_link rows with progress counter and [q]uit)
- c9r canon machine (lists MLflow runs with metrics, user picks a model, RapidFuzz pre-filter → ML classification → union-find grouping → writes new candidates with to_link=NULL)

---

## 2026-02-25 — MLflow experiment tracking + dashboard expansion (v0.6.2)

### Added
- **MLflow integration** — `helpers/experiment.py` wraps MLflow for experiment tracking. `corefunc/canon.py` now logs hyperparameters, metrics (precision, recall, F1, AUC), and model artefacts to a local SQLite-backed MLflow store (`mlruns.db`). `log_cv_fold()` helper ready for stratified k-fold CV.
- **`c9r train --run-name`** — optional MLflow run name for training sessions.
- **`c9r mlflow-ui`** — launches the MLflow tracking UI for experiment comparison.
- **`c9r dashboard` command group** — expanded from a single command to four subcommands:
  - `c9r dashboard artist [-n N]` — top artists by play count (original behaviour)
  - `c9r dashboard album [-n N]` — top albums as `artist_name: album_title`
  - `c9r dashboard track [-n N]` — top tracks as `artist_name: track_title (album_title)`
  - `c9r dashboard recent [-n N]` — most recent scrobbles with timestamps
- Query functions in `helpers/query.py`: `top_albums()`, `top_tracks()`, `recent_scrobbles()`.

### Changed
- `evaluate()` in `corefunc/canon.py` now returns a `dict[str, float]` with precision, recall, f1, and auc (still prints the full report).
- `train_model()` accepts an optional `run_name` parameter.
- MLflow backend uses SQLite (`sqlite:///mlruns.db`) instead of the deprecated filesystem store.

---

## 2026-02-24 — Profiling, local enrichment, QA, purge (v0.6.1)

### Added
- **`c9r profile` command group** with 5 subcommands:
  - `overview` — total scrobbles, unique artists/tracks/albums, date range, yearly bar chart, play-count distribution
  - `variants` — fuzzy-similar artist name pairs that split scrobble counts (the Bohren problem)
  - `top [-n 20] [--canonized]` — top N artists, optionally re-ranked after AVC canonisation
  - `companions [--start 2006] [--end 2025]` — artists present in every year, ranked by consistency (lowest σ)
  - `countries [-n 15]` — country breakdown via artist_info join
- **`c9r enrich-local [--rebuild]`** — bulk artist enrichment via local MusicBrainz PostgreSQL mirror using `docker exec` + `psql`. Two-tier resolution: MBID match (15,691 rows), then exact name match (114 more).
- **`c9r qa` subcommands** — `a_i` (artist_info), `avc`, `uc` checks added alongside existing `scrobble` checks. `show` command improved with source/target display.
- **`c9r purge`** — three modes: interactive picker, `--all` with confirmation, `--all --yes` for scripting.

### Changed
- `corefunc/qa.py` refactored: `_check_schema`, `_check_nulls`, `_check_mbids`, `_check_encoding` accept column/target params. `_persist_report` stores a `target` column.
- Column constants added to `helpers/io.py`: `ARTIST_INFO_COLS`, `AVC_COLS`, `UC_COLS`.

---

## 2026-02 — Dual-source ingestion: Last.fm + ListenBrainz

### Added
- **`HTTP/lblink.py`** — `fetch_scrobbles_since()` paginates via `max_ts`, normalises listens into the standard column set. Extracts `artist_mbid` from `track_metadata.additional_info.artist_mbids[0]`.
- **`--source` / `-s` CLI option** on `ingest`, `enrich`, `flow` — accepts `lastfm` (default), `listenbrainz`, or `lb`. Env-var fallback: `LASTFM_USER` or `LB_USER`.

### Changed
- `lfAPI.py` — moved `LASTFM_API_KEY` check from module-level into `lastfm_request()` so importing doesn't crash when only LB credentials are configured.
- `corefunc/workflow.py` — `source` parameter; skips `sync_user_country` for ListenBrainz.
- `flows/cf_ingest.py` — `source` threaded through tasks and flow.

---

## 2026-02 — v0.6.0: Parquet + DuckDB migration

Major migration from MySQL/SQLAlchemy + curses architecture to Parquet/DuckDB + Click.

### Removed
- Entire `DB/` package, `alembic/` directory, `setup.py`, `requirements.txt`, `alembic.ini`
- Dependencies: sqlalchemy, pymysql, alembic, windows-curses, questionary, openai, python-docx, jupyter, flask, werkzeug, folium, branca, fonttools, pillow, cryptography, starlette
- `helpers/cli_interface.py` (curses GUI)
- 18 dead test files, `CSV/`, `dev/`, `scripts/`, `tests/e2e/`, `tests/setup/`, and 14 dead source files

### Added
- Dependencies: click, duckdb, fastapi, uvicorn
- `helpers/io.py` — sole Parquet I/O module: `append_to_parquet`, `dump_parquet`, `read_parquet` with PyArrow schemas, zstd compression, app-layer dedup
- `helpers/query.py` — DuckDB-over-Parquet analytical queries
- Click CLI (`main.py`) with 9 subcommands: `ingest`, `enrich`, `canonise`, `review`, `train`, `serve`, `dashboard`, `purge`, `flow`
- `corefunc/canon.py` — promoted from `dev/canon.py` with `train_model()`, `evaluate()`
- `corefunc/model_server.py` — FastAPI app with `/predict`, `/predict_batch`, `/health`
- `flows/cf_ingest.py` — Prefect 3.x flow with 3 tasks (fetch, enrich, clean)
- Multi-stage Dockerfile (python:3.12-slim + uv)

### Changed
- All `engine.connect()` + `pd.read_sql()` calls replaced with DuckDB equivalents
- HTTP modules (`lfAPI.py`, `mbAPI.py`) stripped of all DB/SQLAlchemy imports
- `helpers/stats.py` — removed ~550 lines of DB-dependent code; kept pure statistical utilities
- `helpers/cli.py` prompts migrated from questionary → Click

---

## 2025-09 — Airflow → Prefect migration

### Removed
- `apache-airflow~=3.0.1` and all related providers, `/dags` directory, `requirements-airflow.txt`

### Added
- `prefect>=3.0`, `prefect-sqlalchemy>=0.5.0`
- `/flows/cf_ingest.py`, `tests/test_prefect_flow.py`, `docs/PREFECT_SETUP.md`

### Changed
- `PythonOperator` → `@task`, `DAG` → `@flow`, XCom → direct Python return values, Airflow schedule → Prefect `CronSchedule`
- Resolved SQLAlchemy version conflict (Airflow 3.x required < 2.0, conflicting with core app's >= 2.0.40)

---

## 2025-08 — Database migration fix
- Added initial migration to create all tables before adding columns.
- Fixed migration sequence to ensure tables exist before modification.
- Resolved "Table 'canonfodder.artist_variants_canonized' doesn't exist" error.

---

## 2025-07 — SQLAlchemy compatibility layer
- Added compatibility layer in `DB/models.py` for SQLAlchemy 1.4.x (Airflow) + 2.0 (core app).
- Implemented fallback mechanisms for SQLAlchemy 2.0 features under 1.4.x.

---

## 2025-06 — Orchestration and pipeline improvements
- Added weekly autofetch with Airflow integration.
- Created pull-based pipeline (`corefunc/pipeline.py`).
- Implemented conflict handling policies for data merging.
- Created Airflow DAG in `dags/` directory.
- Refactored data gathering for incremental updates; improved MusicBrainz rate-limit handling.

---

## Earlier — Modularisation, serialisation, UX
- Reorganised API modules into `HTTP/`; dev scripts into `dev/`; created `corefunc/` package.
- Added JSON config files for colour palettes and feature selection.
- Implemented XGBoost model serialisation; updated Parquet file structure.
- Major overhaul of menu logic; added progress tracking for long-running operations.
- Added comprehensive testing infrastructure.

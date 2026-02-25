# Changelog

All notable changes to c9r (CanonFodder) in reverse chronological order.

---

## 2026-02-25 – c9r canon sketched out

### Added
- corefunc/avc_seed.py — MySQL dump parser that seeded PQ/avc.parquet with 488 rows (177 linked, 311 skipped) from the old gold standard
- tests/unit/test_canonize.py — 15 unit tests covering all functions (all passing)

### Changed
- corefunc/canon.py — business logic for all four subcommands: avc_summary(), propagate_avc(), undecided_rows() / update_avc_decision(), list_mlflow_runs() / load_run_model() / discover_candidates() / write_new_candidates()
- main.py — replaced the placeholder canonise and review commands with the canon group housing:
- c9r canon avc show (with --decided / --undecided / --last N filters, tabular output with variants column last)
- c9r canon avc propagate (renames artist_info rows, appends to aliases without overwriting)
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

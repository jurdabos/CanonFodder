# c9r Migration Log

Consolidated migration history in reverse chronological order.

---

## February 2026: Dual-source support — Last.fm + ListenBrainz ingestion

### Problem
The pipeline hardcoded Last.fm as the only scrobble source. Users who use ListenBrainz instead (or in addition) could not ingest their listening history.

### Changes ✅
1. **Safe `lfAPI.py` import** — moved `LASTFM_API_KEY` check from module-level into `lastfm_request()` so importing `lfAPI` no longer crashes when only LB credentials are configured.
2. **`fetch_scrobbles_since()` in `HTTP/lblink.py`** — new function that paginates via `max_ts`, normalises listens into the same column set (`Artist`, `Song`, `Album`, `uts`, `artist_mbid`) that `normalise_scrobble_df` already accepts. Extracts `artist_mbid` from `track_metadata.additional_info.artist_mbids[0]` when available.
3. **`--source` / `-s` CLI option** on `ingest`, `enrich`, `flow` — accepts `lastfm` (default) or `listenbrainz` (alias `lb`). Env-var fallback: `LASTFM_USER` or `LB_USER`.
4. **`corefunc/workflow.py`** — `source` parameter on `run_data_gathering_workflow()`. Skips `sync_user_country` for ListenBrainz.
5. **`flows/cf_ingest.py`** — `source` threaded through tasks and flow. Default `"lastfm"`, fallback env var `C9R_SOURCE`.
6. **`.env.example` and `README.md`** updated with `LB_USER`, `C9R_SOURCE`, and `--source` usage examples.
7. **Tests** — 225 passing, 86% coverage.

### Files changed
`HTTP/lfAPI.py`, `HTTP/lblink.py`, `main.py`, `corefunc/workflow.py`, `flows/cf_ingest.py`, `.env.example`, `README.md`, `tests/unit/test_lblink_extra.py`, `tests/unit/test_cli.py`, `tests/integration/test_workflow.py`

---

## February 2026: c9r v0.6 — Parquet + DuckDB migration

Transition from MySQL/SQLAlchemy + curses architecture to Parquet/DuckDB + Click.

### Previous state
- Storage: MySQL via SQLAlchemy ORM (with SQLite fallback), Alembic migrations
- CLI: argparse + windows-curses GUI, questionary prompts
- HTTP: lfAPI.py, mbAPI.py — both dependent on DB/ for caching
- Workflow: corefunc/workflow.py — dependent on DB.ops
- Parquet: helpers/io.py wrote Parquet as a secondary cache alongside MySQL
- Dependencies: SQLAlchemy, pymysql, windows-curses, questionary, alembic, openai, python-docx, jupyter, etc.

### Phase 1 — Foundation: pyproject.toml, .env, project metadata
- Renamed project to c9r, bumped version to 0.6.0
- Removed: sqlalchemy, pymysql, alembic, windows-curses, questionary, openai, python-docx, jupyter, flask, werkzeug, folium, branca, fonttools, pillow, cryptography, starlette
- Added: click, duckdb, fastapi, uvicorn
- Kept: pandas, numpy, pyarrow, requests, musicbrainzngs, tenacity, rapidfuzz, xgboost, scikit-learn, python-dotenv, prefect, matplotlib, seaborn
- Updated .env.example: dropped DB\_URL, renamed TOKEN → LB\_TOKEN, added PQ\_DIR
- Removed setup.py, requirements.txt, alembic.ini
- Updated pyproject.toml `[project.scripts]` to `c9r = "main:cli"`

### Phase 2 — Storage layer: remove DB/, upgrade PQ/ and helpers/io.py
- Deleted entire DB/ package and alembic/ directory
- Rewrote helpers/io.py as the sole Parquet I/O module: `append_to_parquet`, `dump_parquet`, `read_parquet`
- Constants: PQ\_DIR, SCROBBLE\_PQ, ARTIST\_INFO\_PQ, AVC\_PQ, C\_PQ, UC\_PQ
- Schemas enforced via PyArrow; all zstd-compressed, app-layer dedup, 10k-row chunk inserts

### Phase 3 — DuckDB query layer
- Created helpers/query.py with `duckdb_over_parquet(sql, *, pq_dir)` helper
- Exposed common analytics queries (top artists, ascii\_freq, etc.) as functions
- Replaced every `engine.connect()` + `pd.read_sql()` call with DuckDB equivalents

### Phase 4 — Source layer: refactor HTTP/ modules
- HTTP/lfAPI.py → removed all DB/SQLAlchemy imports; `fetch_scrobbles_since()` returns DataFrame
- HTTP/mbAPI.py → removed DB caching; `_cache_artist()` writes to artist\_info.parquet
- HTTP/client.py → kept as-is (resilient HTTP GET with retries)
- dev/lblink.py → moved to HTTP/lblink.py (LBClient for ListenBrainz)

### Phase 5 — CLI: replace argparse+curses with Click
- Rewrote main.py as a Click group: `ingest`, `enrich`, `canonise`, `review`, `train`, `serve`, `dashboard`, `purge`, `flow`
- Deleted helpers/cli\_interface.py (curses GUI)
- Migrated helpers/cli.py prompts from questionary → Click

### Phase 6 — ELT / core functions
- corefunc/workflow.py → rewritten for Parquet I/O
- corefunc/data\_cleaning.py, corefunc/data\_gathering.py → updated imports
- corefunc/enrich.py → MB enrichment writing to artist\_info.parquet
- helpers/cli.py: `unify_artist_names_cli()` → Click prompts, writes to avc.parquet

### Phase 7 — Model pipeline & server
- dev/canon.py → promoted to corefunc/canon.py with `train_model()`, `evaluate()`
- corefunc/model\_server.py → FastAPI app with /predict and /health endpoints
- Model persisted as ML/xgb.json, feature columns as ML/xgb\_columns.json

### Phase 8 — Prefect flow, Docker, tests ✅
- flows/cf\_ingest.py → 3 lean tasks (fetch\_scrobbles, enrich\_artists, clean\_artists)
- Dockerfile added (python:3.12-slim + uv, multi-stage)
- conftest.py → Parquet temp-dir fixtures; patches helpers.io and helpers.query constants
- 18 dead test files deleted; 6 new test modules created (14 + 8 + 7 + 3 + 4 + 5 tests)
- 36 passed, 1 skipped, 0 failed

### Phase 9 — Cleanup ✅
- Deleted dead directories (CSV/, dev/, scripts/, tests/e2e/, tests/setup/) and 14 dead source files
- Rewrote helpers/stats.py — removed all DB/SQLAlchemy imports (~550 lines); kept pure statistical utilities
- Removed dead `fetch_lastfm_with_progress()` from HTTP/lfAPI.py
- Rewrote README.md for c9r v0.6
- Updated .gitignore — added ML/, pics/, visualizations/; removed alembic/ entry
- 36 passed, 1 skipped, 0 failed

---

## September 2025: Airflow → Prefect migration (BREAKING)

### Summary
Migrated workflow orchestration from Apache Airflow to Prefect 3.0+ to resolve fundamental dependency incompatibilities (Airflow 3.x requires SQLAlchemy < 2.0, conflicting with CanonFodder's SQLAlchemy >= 2.0.40).

### Dependencies
- **Removed:** `apache-airflow~=3.0.1` and all related providers, `requirements-airflow.txt`
- **Added:** `prefect>=3.0`, `prefect-sqlalchemy>=0.5.0`

### File structure
- **Removed:** `/dags` directory, Airflow test files and docs
- **Added:** `/flows/cf_ingest.py`, `tests/test_prefect_flow.py`, `docs/PREFECT_SETUP.md`

### Key differences from Airflow
1. `PythonOperator` → `@task` decorator
2. `DAG` object → `@flow` decorator
3. XCom → direct Python return values
4. Airflow schedule expressions → Prefect `CronSchedule`
5. No separate webserver/scheduler processes

### Benefits
Simpler architecture, better dev experience, full async/await support, no SQLAlchemy version conflicts, easier testing.

---

## August 2025: Database migration fix
- Added new initial migration to create all tables before adding columns
- Fixed migration sequence to ensure tables exist before modification
- Resolved "Table 'canonfodder.artist\_variants\_canonized' doesn't exist" error
- Improved Docker container startup reliability

---

## July 2025: SQLAlchemy compatibility layer
- Added compatibility layer in DB/models.py for SQLAlchemy 1.4.x (Airflow) + 2.0 (core app)
- Implemented fallback mechanisms for SQLAlchemy 2.0 features under 1.4.x
- Fixed `DeclarativeBase` import error in Airflow environment
- Improved Docker container compatibility with Airflow

---

## June 2025: Orchestration and pipeline improvements
- Added weekly autofetch with Airflow integration
- Created pull-based pipeline (`corefunc/pipeline.py`) — manual or Airflow-triggered
- Implemented conflict handling policies for data merging
- Created Airflow DAG in `dags/` directory
- Added "Full Pipeline" option to CLI; improved error handling and progress reporting
- Refactored data gathering for incremental updates; improved MusicBrainz rate-limit handling

---

## Earlier: Modularisation, serialisation, UX
- Reorganised API modules into HTTP/ directory; dev scripts into dev/; created corefunc/ package
- Added JSON config files for colour palettes and feature selection
- Implemented XGBoost model serialisation; updated Parquet file structure
- Major overhaul of menu logic; added progress tracking for long-running operations
- Moved setup.py to project root; removed redundant config/ directory
- Added comprehensive testing infrastructure

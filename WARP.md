# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

c9r (CanonFodder) is a reproducible data-engineering pipeline that ingests music scrobbles from ListenBrainz or Last.fm, enriches them with MusicBrainz metadata, resolves artist-name ambiguity through ML-assisted record linkage, and delivers interactive analytics. All data lives in zstd-compressed Parquet files queryable via DuckDB.

## Common Development Commands

### Environment Setup
```bash
uv sync                      # core dependencies
uv sync --extra dev          # add pytest, ruff, httpx
uv sync --extra all          # everything including Prefect
```

### Running the Application
All commands use the `c9r` entry point:
```bash
uv run c9r --help            # list all commands
uv run c9r -v <command>      # enable debug logging
```

### Testing
```bash
uv run pytest                # full suite with coverage
uv run pytest tests/unit/    # unit tests only
uv run pytest tests/unit/test_qa_ml_extra.py -v  # single file
```

Coverage target is 80 % (configured in pyproject.toml).

### Linting
```bash
uv run ruff check .
uv run ruff format .
```

## Architecture & Code Organisation

### Core Data Flow
1. **Ingest** (`HTTP/lblink.py`, `HTTP/lfAPI.py`) → year-partitioned `PQ/scrobble/year=YYYY/`
2. **Enrich** (`corefunc/enrich.py`, `HTTP/mbAPI.py`) → `PQ/artist_info.parquet`
3. **Canonise** (`corefunc/canon/workflow.py`) → `PQ/avc.parquet`, `PQ/predictions_log.parquet`
4. **Profile / Dashboard** (`corefunc/profile.py`, `helpers/query.py`) → DuckDB over Parquet
5. **QA** (`corefunc/qa.py`) → `PQ/qa_report.parquet`

### API Integration
- **Last.fm API** (`HTTP/lfAPI.py`): paginated scrobble fetch, artist MBID enrichment
- **ListenBrainz API** (`HTTP/lblink.py`): paginated listen fetch with `max_ts`
- **MusicBrainz API** (`HTTP/mbAPI.py`): artist lookup with 1 req/s rate limiting via tenacity
- **Resilient HTTP** (`HTTP/client.py`): retries with exponential backoff (2^n, capped at 120 s)

### ML / Canonisation
- **Feature engineering** (`helpers/features.py`, `helpers/inference.py`): 3-tier pairwise features (whole-string, token, character) + interaction terms + catalogue (disco/melo) features — ~63 raw, pruned to 41
- **Training** (`corefunc/canon/trainer.py`): unified pipeline with multiple data sources, feature tiers, catalogue designs, k-fold CV, MLflow tracking
- **TCN training** (`corefunc/canon/tcn_trainer.py`): Siamese and Hybrid TCN architectures
- **Tuning** (`corefunc/canon/tuner.py`): Optuna precision-biased hyperparameter search
- **Experiment runner** (`corefunc/canon/experiment_runner.py`): multi-model benchmarking with 8 models
- **Inference** (`helpers/inference.py`): `compute_inference_features()` + `load_model()`
- **Model server** (`corefunc/model_server.py`): FastAPI with `/predict`, `/predict_batch`, `/health`
- **Drift detection** (`corefunc/qa.py`): `qa_predictions()` compares baseline vs recent windows on mean probability, ambiguous-band proportion, and per-feature median quantile shifts

### Workflow Orchestration
- **Prefect flow** (`flows/cf_ingest.py`): fetch → fix-encoding → enrich → clean → canonise → propagate → augment → retrain — with exponential backoff retries

### Data Storage (all git-ignored)
- **`PQ/`**: Parquet data store — `scrobble/` (partitioned), `artist_info`, `avc`, `c`, `gs_mb`, `predictions_log`, `qa_report`, `uc`
- **`ML/`**: trained model pickles (e.g. `lightgbm_best.pkl`)
- **`JSON/`**: colour palettes and configuration

## Key Technical Details

### Environment Variables (.env)
Required:
- `LASTFM_API_KEY`: from https://www.last.fm/api/account/create (for Last.fm source)
- `LB_TOKEN`: ListenBrainz user token (for LB source)

Optional:
- `LASTFM_USER` / `LB_USER`: default username so `--user` can be omitted
- `C9R_SOURCE`: default data source (`lastfm` or `listenbrainz`)

### Dependency Management
- `uv` for venv + packaging (PEP 621 in `pyproject.toml`, locked via `uv.lock`)
- Core deps include Click, DuckDB, FastAPI, LightGBM, MLflow, Optuna, pandas, PyArrow, RapidFuzz, scikit-learn, XGBoost, PyTorch
- Prefect 3.x for orchestration

### Parquet Schema Management
- `helpers/schema.py`: versioned `SCHEMA_REGISTRY` with typed PyArrow schemas
- Writers auto-stamp files with `c9r_table` and `c9r_schema_version` metadata
- Readers check stamps on load — warn on stale, reject future versions
- CLI: `c9r schema show` / `c9r schema migrate`

### Canonisation Pipeline
1. RapidFuzz WRatio pre-filter on scrobble artist names
2. `compute_inference_features()` produces ~63 pairwise features
3. LightGBM pipeline (`ML/lightgbm_best.pkl`) scores each pair
4. Union-find groups positive pairs
5. New candidates written to `avc.parquet` with `to_link=NULL`
6. Predictions logged to `predictions_log.parquet` (timestamp, pair, probability, feature vector)
7. Human reviews via `c9r canon human`; decisions applied via `c9r canon avc propagate`

### Testing Strategy
- pytest suite in `tests/` — `unit/` and `integration/` subdirectories
- Fixtures in `tests/conftest.py` create temp `PQ/` dirs and monkeypatch all path constants
- Coverage gate: 80 % (fail_under in pyproject.toml)

## Directory Layout
- `corefunc/` — core pipeline: `canon/` (training, tuning, workflow), `enrich.py`, `qa.py`, `profile.py`, `model_server.py`
- `helpers/` — I/O (`io.py`), DuckDB queries (`query.py`), features (`features.py`, `inference.py`), schema (`schema.py`), stats, clustering, device
- `HTTP/` — API clients: `client.py`, `lfAPI.py`, `lblink.py`, `mbAPI.py`
- `flows/` — Prefect flow definitions
- `tests/` — pytest suite (unit + integration)
- `docs/` — CHANGELOG and project documentation
- `JSON/` — configuration files
- `PQ/` — Parquet data files (git-ignored)
- `ML/` — model artefacts (git-ignored)

# c9r (CanonFodder)

## Overview

c9r is a reproducible data-engineering pipeline that ingests music listening events (scrobbles), enriches them with metadata, stores them in columnar Parquet files queryable via DuckDB, and provides tools for artist-name canonisation through fuzzy matching and ML.

## Motivation

Scrobble service providers often struggle with data quality — the same artist appears under multiple name variants. c9r addresses this by building a record-linkage pipeline that clusters and standardises artist names, ensuring accurate music listening analytics.

For demonstration purposes the default instance uses Last.fm scrobbles from https://www.last.fm/user/jurda (active since 2006). Any new instance can be pointed at a different Last.fm user via the CLI.

## Technical Stack

- **Storage**: Apache Parquet files + DuckDB for ad-hoc analytical queries
- **APIs**: Last.fm for scrobble retrieval, MusicBrainz for metadata enrichment
- **ML**: XGBoost + scikit-learn for artist-name record linkage
- **CLI**: Click command groups exposed as `c9r` entry point via `uv run c9r`
- **Model server**: FastAPI + Uvicorn for serving predictions over HTTP
- **Orchestration**: Prefect (optional) for scheduled/automated flows
- **Packaging**: uv for dependency management and virtualenvs

## Repository Structure

- **corefunc/** — core pipeline logic: canonisation, data cleaning, enrichment, workflow, model server
- **helpers/** — I/O (Parquet read/write), DuckDB query layer, statistics, clustering utilities
- **HTTP/** — Last.fm and MusicBrainz API clients
- **flows/** — Prefect flow definitions
- **tests/** — pytest suite (unit, integration)
- **JSON/** — configuration files including colour palettes
- **PQ/** — Parquet data files (git-ignored)
- **ML/** — trained model artefacts (git-ignored)
- **docs/** — project documentation

## Installation

### Prerequisites

- Python 3.12+
- Git
- uv (Python package manager)

### Installing uv

```powershell
# Windows
winget install Astral-sh.Uv

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

```shell
git clone https://github.com/jurdabos/canonfodder.git
cd canonfodder
uv sync                      # core dependencies
uv sync --extra prefect      # add Prefect for orchestration
uv sync --extra dev          # add pytest, ruff, httpx
uv sync --extra all          # everything
```

### Configuration

1. Copy `.env.example` to `.env`
2. Set `LASTFM_API_KEY` (get one free at https://www.last.fm/api/account/create)
3. Optionally set `LASTFM_USER` so the `--user` flag can be omitted

## CLI Usage

All commands are available through the `c9r` entry point:

```shell
uv run c9r <command> [options]
```

Pass `--verbose` / `-v` before any subcommand for debug logging.

### Ingesting scrobbles

```shell
# Incremental fetch (default — resumes from the last stored timestamp)
uv run c9r ingest --user jurda

# Full history fetch
uv run c9r ingest --user jurda --full

# ListenBrainz instead of Last.fm
uv run c9r ingest --user jurda --source listenbrainz
```

### Enriching metadata

```shell
# Default: local MusicBrainz mirror
uv run c9r enrich

# Remote MusicBrainz API
uv run c9r enrich --mbapi

# Last.fm API for MBIDs + remote MB for metadata
uv run c9r enrich --lastfmapi

# Also sync user country
uv run c9r enrich --country --user jurda

# Rebuild artist_info from scratch
uv run c9r enrich --rebuild
```

### Canonisation (`canon`)

Artist-name record linkage lives under `c9r canon`.

#### AVC table operations (`canon avc`)

```shell
# Show the full AVC (Artist Variants Canonized) table
uv run c9r canon avc show

# Only decided or undecided rows
uv run c9r canon avc show --decided
uv run c9r canon avc show --undecided

# Apply canonisation decisions to artist_info
uv run c9r canon avc propagate

# Seed AVC from a legacy MySQL dump (one-time migration)
uv run c9r canon avc seed path/to/dump.sql

# Extract training pairs from the local MB mirror
uv run c9r canon avc augment --pos-limit 5000 --neg-limit 5000
```

#### Interactive and ML-driven review

```shell
# Human review of undecided variant groups
uv run c9r canon human

# ML-assisted variant discovery (picks a trained model interactively)
uv run c9r canon machine --cutoff 75 --threshold 0.5
```

#### Multi-model experiment

```shell
# Run the full experiment (all 8 models, 5-fold CV, logged to MLflow)
uv run c9r canon experiment --augment

# Run specific models only
uv run c9r canon experiment --models "XGBoost,LightGBM,RandomForest"

# Custom fold count and run name
uv run c9r canon experiment --folds 10 --run-name "baseline-v2"
```

### Training a single model

```shell
uv run c9r train
uv run c9r train --run-name "xgb-baseline" --augment
```

### MLflow UI

The MLflow tracking UI is a **viewer** — it is not required before training.
Launch it after `train` or `canon experiment` to compare runs:

```shell
uv run c9r mlflow-ui                    # http://127.0.0.1:5000
uv run c9r mlflow-ui --port 5050        # custom port
```

### Model server

```shell
uv run c9r serve                        # http://127.0.0.1:8000
uv run c9r serve --port 9000
```

### Dashboard

```shell
uv run c9r dashboard artist --top 20    # top artists by play count
uv run c9r dashboard album --top 10     # top albums
uv run c9r dashboard track --top 10     # top tracks
uv run c9r dashboard recent -n 15       # most recent scrobbles
uv run c9r dashboard yearly --top 3     # gold/silver/bronze per year
```

### Profiling

```shell
uv run c9r profile overview             # high-level stats and yearly bar chart
uv run c9r profile top -n 20            # top 20 artists (raw)
uv run c9r profile top --canonized      # re-ranked after alias resolution
uv run c9r profile top --custom "(1,5),(27,29)"  # custom rank ranges
uv run c9r profile variants             # fuzzy near-duplicate artist names
uv run c9r profile companions           # artists present every year
uv run c9r profile countries             # scrobbles by artist-origin country
uv run c9r profile population           # per-capita country ranking
uv run c9r profile where                # user country at scrobble time
uv run c9r profile uc                   # medal tables per user country
uv run c9r profile timeline             # monthly summary across all years
uv run c9r profile streaks              # listening streaks and gaps
uv run c9r profile clock                # hour-of-day and day-of-week patterns
```

### Quality assurance

```shell
uv run c9r qa scrobble                  # QA checks on scrobble.parquet
uv run c9r qa scrobble --hours 24       # only last 24 h
uv run c9r qa a_i                       # artist_info checks
uv run c9r qa avc                       # AVC table checks
uv run c9r qa gs_mb                     # gold-standard pair checks
uv run c9r qa uc                        # user country summary
uv run c9r qa show --last 10            # recent QA reports
uv run c9r qa show --fail-only          # failures only
```

### Maintenance

```shell
# Repair encoding-corrupted strings
uv run c9r fix-encoding

# Interactive Parquet file purge
uv run c9r purge

# Delete all Parquet files (with confirmation)
uv run c9r purge --all

# Skip confirmation
uv run c9r purge --all --yes
```

### Orchestration

```shell
# Run the full Prefect ingest → enrich → clean flow
uv run c9r flow
uv run c9r flow --full --source listenbrainz
```

## Docker

```shell
docker build -t c9r .
docker run --env-file .env c9r ingest --user jurda
```

## Testing

```shell
uv run pytest
```

Tests live in `tests/` and are organised into `unit/` and `integration/` subdirectories.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgements

- Last.fm for providing the scrobble API
- MusicBrainz for their comprehensive music metadata database
- Ben Foxall's [lastfm-to-csv](https://github.com/benfoxall/lastfm-to-csv) for inspiration on scrobble data extraction
- Research by Elsden et al. (2016) on personal music tracking and lifelogging

## Contact

For questions or feedback, please contact the project maintainer.

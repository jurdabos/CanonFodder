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
- **CLI**: Click command group with composable subcommands
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

All commands are available via `uv run python main.py <command>`:

```shell
# Fetching scrobbles (incremental by default)
uv run python main.py ingest --user jurda

# Full history fetch
uv run python main.py ingest --user jurda --full

# Enriching artist MBIDs and user country
uv run python main.py enrich --user jurda

# Training the XGBoost canonisation model
uv run python main.py train

# Starting the FastAPI model server
uv run python main.py serve --port 8000

# Quick text dashboard
uv run python main.py dashboard --top 20

# Running the full Prefect flow
uv run python main.py flow

# Purging all Parquet data files
uv run python main.py purge --all

# Log and view results
uv run c9r mlflow-ui

# Run the full experiment (all 8 models)
uv run c9r canon experiment --augment

# Run specific models only
uv run c9r canon experiment --models "XGBoost,LightGBM,RandomForest"
```

Pass `--verbose` / `-v` before any subcommand for debug logging.

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

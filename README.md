# c9r (CanonFodder)

## Overview

c9r is a reproducible data-engineering pipeline that ingests music listening events (scrobbles), enriches them with metadata, stores them in columnar Parquet files queryable via DuckDB, and provides tools for artist-name canonisation through fuzzy matching and ML.

## Motivation

Scrobble service providers often struggle with data quality — the same artist appears under multiple name variants. c9r addresses this by building a record-linkage pipeline that clusters and standardises artist names, ensuring accurate music listening analytics.

## Technical Stack

- **Storage**: Apache Parquet (zstd-compressed, year-partitioned scrobbles) + DuckDB for ad-hoc analytical queries
- **APIs**: ListenBrainz (recommended) and Last.fm for scrobble retrieval; MusicBrainz for metadata enrichment (local Docker mirror or remote API)
- **ML**: LightGBM + scikit-learn pipeline for artist-name record linkage; ~41-feature pairwise classification with 3-tier feature engineering (whole-string, token, character) + catalogue features; Optuna for tuning; MLflow for experiment tracking
- **CLI**: Click command groups exposed as `c9r` entry point via `uv run c9r`
- **Model server**: FastAPI + Uvicorn (`/predict`, `/predict_batch`, `/health`)
- **Orchestration**: Prefect 3.x for scheduled/automated flows
- **Packaging**: uv for dependency management and virtualenvs
- **Drift detection**: prediction logging with feature vectors; QA compares baseline vs recent windows on probability, ambiguous-band proportion, and feature quantiles

## Repository Structure

- **corefunc/** — core pipeline: `canon/` (training, tuning, experiment runner, workflow), `enrich.py`, `qa.py`, `profile.py`, `data_cleaning.py`, `model_server.py`
- **helpers/** — I/O (`io.py`), DuckDB queries (`query.py`), features (`features.py`, `inference.py`), schema versioning (`schema.py`), stats, clustering, device
- **HTTP/** — API clients: `client.py` (resilient HTTP), `lfAPI.py` (Last.fm), `lblink.py` (ListenBrainz), `mbAPI.py` (MusicBrainz)
- **flows/** — Prefect flow definitions (`cf_ingest.py`)
- **tests/** — pytest suite (`unit/`, `integration/`)
- **JSON/** — configuration files including colour palettes
- **PQ/** — Parquet data store (git-ignored): `scrobble/` (partitioned), `artist_info`, `avc`, `c`, `gs_mb`, `predictions_log`, `qa_report`, `uc`
- **ML/** — trained model pickles (git-ignored), e.g. `lightgbm_best.pkl`
- **docs/** — CHANGELOG and project documentation

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

### Ingesting scrobbles (`ingest`)

```shell
# Incremental fetch (default — resumes from the last stored timestamp)
uv run c9r ingest --user jurda

# Full history fetch
uv run c9r ingest --user jurda --full

# ListenBrainz instead of Last.fm
uv run c9r ingest --user jurda --source listenbrainz
```

Options:
- `-u, --user TEXT` — username (falls back to env `LASTFM_USER` or `LB_USER`)
- `-s, --source [lastfm|listenbrainz|lb]` — data source (falls back to env `C9R_SOURCE`)
- `--full` — fetch full history instead of incremental

### Enriching metadata (`enrich`)

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

Options:
- `--mbapi` — use remote MusicBrainz API instead of local mirror
- `--lastfmapi` — use Last.fm API for MBIDs + remote MB API for metadata
- `--country` — sync user country to `uc.parquet` (requires `--user`)
- `--rebuild` — rebuild `artist_info.parquet` from scratch
- `-u, --user TEXT` — username (only for `--country`)
- `-s, --source [lastfm|listenbrainz|lb]` — data source (only for `--country`)

### Canonisation (`canon`)

Artist-name record linkage lives under `c9r canon`.

#### AVC table operations (`canon avc`)

```shell
# Show the full AVC (Artist Variants Canonized) table
uv run c9r canon avc show

# Only decided or undecided rows
uv run c9r canon avc show --decided
uv run c9r canon avc show --undecided

# Show the last N rows
uv run c9r canon avc show --last 20

# Apply canonisation decisions to artist_info
uv run c9r canon avc propagate

# Seed AVC from a legacy SQL dump (one-time migration)
uv run c9r canon avc seed path/to/dump.sql

# Extract training pairs from the local MB mirror
uv run c9r canon avc augment --pos-limit 5000 --neg-limit 5000
```

`canon avc show` options:
- `--decided` — show only decided rows (to_link 0 or 1)
- `--undecided` — show only undecided rows (to_link NULL)
- `--last INTEGER` — show only the last N rows

`canon avc augment` options:
- `--pos-limit INTEGER` — max positive (alias→canonical) pairs
- `--neg-limit INTEGER` — max negative pairs
- `--similarity-floor INTEGER` — WRatio floor for hard negatives (0–100)

#### Interactive and ML-driven review

```shell
# Human review of undecided variant groups
uv run c9r canon human

# ML-assisted variant discovery using the trained LightGBM pipeline
uv run c9r canon machine
uv run c9r canon machine --cutoff 75 --threshold 0.5 --min-plays 2 --limit 2000
```

`canon machine` options:
- `--cutoff INTEGER` — RapidFuzz WRatio pre-filter cutoff (0–100)
- `--threshold FLOAT` — ML model probability threshold
- `--min-plays INTEGER` — minimum play count to consider
- `--limit INTEGER` — max artists to scan

#### Multi-model experiment (`canon experiment`)

```shell
# Run the full experiment (all 8 models, 5-fold CV, logged to MLflow)
uv run c9r canon experiment --augment

# Run specific models only
uv run c9r canon experiment --models "XGBoost,LightGBM,RandomForest"

# Custom fold count and run name
uv run c9r canon experiment --folds 10 --run-name "baseline-v2"
```

Options:
- `--run-name TEXT` — MLflow parent run name
- `--augment / --no-augment` — include MBDB pairs from `gs_mb.parquet`
- `--folds INTEGER` — number of CV folds
- `--models TEXT` — comma-separated model names to run (default: all)

### Training (`train`)

The `train` command group has two subcommands:

#### `train run` — unified training pipeline

```shell
# Default: train LightGBM with 5-fold CV
uv run c9r train run

# Train specific models with custom options
uv run c9r train run --models "LightGBM,XGBoost" --folds 10 --run-name "exp-v3"

# Different data sources and feature tiers
uv run c9r train run --data-source mbdb --features full --catalogue-source unified
```

Key options:
- `--run-name TEXT` — MLflow parent run name (auto-generated if omitted)
- `--folds INTEGER` — number of CV folds
- `--test-size FLOAT` — held-out test fraction
- `--models TEXT` — comma-separated model names (default: `LightGBM`)
- `--catalogue / --no-catalogue` — include catalogue features
- `--data-source [avc|mbdb|mbdb-max|dbscan|dbscan-capped|mixed]` — training data origin
- `--split [pair|group]` — split strategy (pair or group level)
- `--test-source [holdout|avc-full]` — test data origin
- `--features [base|interaction|full]` — feature tiers: base (23), interaction (53), full (71)
- `--catalogue-source [none|scrobble|mbdb|unified]` — catalogue data origin
- `--catalogue-design [proportional|presence]` — catalogue feature style (default: `proportional`)
- `--group-features / --no-group-features` — include group-level length_stats features
- `--wratio-lower INTEGER` — WRatio band lower bound
- `--wratio-upper INTEGER` — WRatio band upper bound
- `--experiment INTEGER` — experiment number for backfill labelling
- `--include-composites` — include composite models (Voting, Stacking, Bagging)
- `--cluster-cap INTEGER` — max cluster size for dbscan-capped (default: 30)
- `--neg-ratio INTEGER` — target neg:pos ratio for dbscan-capped (default: 10)
- `--feature-strategy [standard|separated]` — standard or separated feature strategy
- `--neg-matching [none|distribution]` — negative matching strategy
- `--neg-count INTEGER` — target count for distribution-matched negatives

#### `train tcn` — TCN-based architectures

```shell
uv run c9r train tcn --model siamese --epochs 50 --batch-size 256
uv run c9r train tcn --model hybrid --lr 3e-4
```

Options:
- `--model [siamese|hybrid]` — TCN architecture
- `--epochs INTEGER` — max training epochs
- `--batch-size INTEGER` — mini-batch size (default: 256 siamese, 512 hybrid)
- `--lr FLOAT` — learning rate (default: 1e-3 siamese, 3e-4 hybrid)
- `--patience INTEGER` — early stopping patience
- `--experiment INTEGER` — experiment number for MLflow labelling
- `--run-name TEXT` — MLflow run name

### Hyperparameter tuning (`tune`)

```shell
# Default: tune LightGBM with Optuna
uv run c9r tune

# Custom trial count and models
uv run c9r tune --models "LightGBM,XGBoost" --trials 200 --min-precision 0.95
```

Options:
- `--run-name TEXT` — MLflow parent run name (auto-generated if omitted)
- `--models TEXT` — comma-separated model names (default: `LightGBM`)
- `--trials INTEGER` — Optuna trials per model
- `--folds INTEGER` — CV folds for tuning inner loop
- `--test-size FLOAT` — held-out test fraction
- `--min-precision FLOAT` — precision floor for the objective
- `--catalogue / --no-catalogue` — include catalogue features

### MLflow UI (`mlflow-ui`)

The MLflow tracking UI is a **viewer** — it is not required before training.
Launch it after `train run` or `canon experiment` to compare runs:

```shell
uv run c9r mlflow-ui                    # http://127.0.0.1:5000
uv run c9r mlflow-ui --port 5050        # custom port
```

Options:
- `--host TEXT` — bind address
- `-p, --port INTEGER` — port to listen on

### Model server (`serve`)

```shell
uv run c9r serve                        # http://127.0.0.1:8000
uv run c9r serve --port 9000
```

Options:
- `--host TEXT` — bind address
- `-p, --port INTEGER` — port to listen on

### Dashboard (`dashboard`)

```shell
uv run c9r dashboard artist --top 20    # top artists by play count
uv run c9r dashboard album --top 10     # top albums
uv run c9r dashboard track --top 10     # top tracks
uv run c9r dashboard recent -n 15       # most recent scrobbles
uv run c9r dashboard yearly --top 3     # gold/silver/bronze per year
```

Each subcommand accepts `-n` / `--top INTEGER` to control the number of results.

### Profiling (`profile`)

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

`profile top` options:
- `-n INTEGER` — number of top artists
- `--canonized` — apply alias-based canonisation before ranking
- `--custom TEXT` — custom rank ranges, e.g. `"(1,5),(27,29)"`

`profile variants` options:
- `-t, --threshold INTEGER` — minimum fuzzy similarity score (0–100) (default: 91)
- `-m, --min-plays INTEGER` — minimum play count to consider
- `-l, --limit INTEGER` — max artists to compare
- `-n, --top INTEGER` — number of results to show

`profile companions` options:
- `--start INTEGER` — start year (inclusive)
- `--end INTEGER` — end year (inclusive)
- `-n INTEGER` — number of companions to show

`profile countries` / `profile population` / `profile where` options:
- `-n INTEGER` — number of top countries to show

`profile uc` options:
- `-n INTEGER` — number of entries per category (medal count)
- `--ucn INTEGER` — number of top user-countries to include
- `-c TEXT` — comma-separated country codes to filter, e.g. `"(HU, ES, DK)"`
- `-s, --show TEXT` — categories to display: `artist`, `album`, `track`, e.g. `"(artist, track)"`

### Quality assurance (`qa`)

```shell
uv run c9r qa scrobble                  # QA checks on scrobble.parquet
uv run c9r qa scrobble --hours 24       # only last 24 h
uv run c9r qa a_i                       # artist_info checks
uv run c9r qa avc                       # AVC table checks
uv run c9r qa gs_mb                     # gold-standard pair checks
uv run c9r qa uc                        # user country summary
uv run c9r qa show --last 10            # recent QA reports
uv run c9r qa show --all                # all reports
uv run c9r qa show --fail-only          # failures only
```

`qa scrobble` options:
- `-h, --hours INTEGER` — only check scrobbles from the last N hours
- `-s, --source [lastfm|listenbrainz|lb]` — data source label

`qa show` options:
- `--last INTEGER` — number of recent reports to show
- `--all` — show all reports (overrides `--last`)
- `--fail-only` — only show failed reports

Prediction drift detection runs automatically as part of `qa_predictions()`, comparing baseline (default: 30 days) vs recent (default: 7 days) windows on mean probability shift (threshold: 0.10), ambiguous-band proportion (threshold: 2×), and per-feature median quantile shift (threshold: 0.15).

### Schema management (`schema`)

```shell
uv run c9r schema show                  # display schema version status of all Parquet files
uv run c9r schema migrate               # migrate all Parquet files to current schema versions
```

### Maintenance

```shell
# Repair encoding-corrupted strings
uv run c9r fix-encoding

# Convert legacy single-file scrobble.parquet to year-partitioned layout
uv run c9r migrate-scrobbles
uv run c9r migrate-scrobbles --remove-legacy  # delete legacy file after migration

# Interactive Parquet file purge
uv run c9r purge

# Delete all Parquet files (with confirmation)
uv run c9r purge --all

# Skip confirmation
uv run c9r purge --all --yes
```

### Orchestration (`flow`)

```shell
# Run the full Prefect flow: ingest → fix-encoding → enrich → clean → canonise → propagate → augment → retrain
uv run c9r flow
uv run c9r flow --full --source listenbrainz
```

Options:
- `-s, --source [lastfm|listenbrainz|lb]` — data source
- `--full` — fetch full history instead of incremental

### Commit & push workflow (`push`)

```shell
# Stage, commit, and push everything in one go (pre-commit-hook-aware)
uv run c9r push -m "feat: my change"

# Preview without making changes
uv run c9r push --dry-run
```

Options:
- `-m, --message TEXT` — commit message (auto-generated fallback when omitted)
- `--dry-run` — preview without making changes

The command comes from the shared [acidbase](https://github.com/jurdabos/acidbase) toolkit (`acidbase.push.push_command`): it retries the commit when pre-commit hooks modify files, amends when a post-commit hook leaves the tree dirty, and activates DVC ingest and dual-remote publishing automatically when those are configured.

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

- Last.fm for providing a scrobble API
- MusicBrainz for their comprehensive music metadata database
- Ben Foxall's [lastfm-to-csv](https://github.com/benfoxall/lastfm-to-csv) for inspiration on scrobble data extraction
- Research by Elsden et al. (2016) on personal music tracking and lifelogging

## Contact

For questions or feedback, please contact the project maintainer.

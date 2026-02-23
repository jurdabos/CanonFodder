# Prefect Setup Guide

Setting up and running the c9r data pipeline with Prefect 3.x.

## Prerequisites

- Python 3.12+
- Prefect 3.0+
- A Last.fm API key **or** a ListenBrainz token (at least one)

## Installation

```powershell
uv sync --extra prefect
```

Copy `.env.example` to `.env` and set the relevant variables:

**Last.fm source:**
- `LASTFM_API_KEY` — from https://www.last.fm/api/account/create
- `LASTFM_USER` — target username

**ListenBrainz source:**
- `LB_TOKEN` — personal API token from https://listenbrainz.org/settings/
- `LB_USER` — target username

**Source selection (optional):**
- `C9R_SOURCE` — `lastfm` (default) or `listenbrainz`

**Prefect server (optional):**
- `PREFECT_API_URL` — e.g. `http://localhost:4200/api`

## Running the Pipeline

### Local execution

```powershell
# Default source (Last.fm)
uv run python flows/cf_ingest.py

# ListenBrainz source
C9R_SOURCE=listenbrainz uv run python flows/cf_ingest.py
```

### Via the CLI

```powershell
uv run python main.py flow                      # Last.fm (default)
uv run python main.py flow --source listenbrainz # ListenBrainz
```

### Using Prefect UI

1. **Start the Prefect server:**
   ```powershell
   uv run prefect server start
   ```
   UI at http://localhost:4200

2. **Serve the flow with a weekly schedule:**
   ```python
   from flows.cf_ingest import weekly_ingest_flow

   weekly_ingest_flow.serve(
       name="weekly-ingest",
       cron="0 0 * * 0",
       tags=["c9r", "scrobble"],
   )
   ```

## Flow Structure

`flows/cf_ingest.py` defines one flow with three tasks:

1. **fetch_scrobbles** — ingests scrobbles from the selected source via `corefunc.workflow`; persists to `scrobble.parquet`
2. **enrich_artists** — looks up country/MBID for unresolved artists via MusicBrainz; persists to `artist_info.parquet`
3. **clean_artists** — deduplicates `artist_info.parquet` via `corefunc.data_cleaning`

Each task uses automatic retries (up to 8) with exponential back-off and jitter.

## Scheduling

Common cron patterns for `flow.serve(cron=...)`:
- Weekly (Sunday midnight): `"0 0 * * 0"`
- Daily: `"0 0 * * *"`
- Monthly: `"0 0 1 * *"`

## Troubleshooting

1. **"LASTFM_USER not set" / "LB_USER not set"** — set the matching env var in `.env` for the source you chose.
2. **MusicBrainz rate limiting** — retry logic handles transient 503s; persistent failures may mean the MB endpoint is overloaded.
3. **Debug logging:**
   ```powershell
   uv run python main.py -v flow
   ```

## Development

1. Edit tasks in `flows/cf_ingest.py`
2. Underlying logic lives in `corefunc/workflow.py`, `corefunc/enrich.py`, `corefunc/data_cleaning.py`
3. Run tests: `uv run pytest tests/test_prefect_flow.py -v`

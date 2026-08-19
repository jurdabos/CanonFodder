# Prefect Setup Guide

Setting up and running the c9r data pipeline with Prefect 3.x.

## Prerequisites

- Python 3.12+
- Prefect 3.0+
- A ListenBrainz token **or** a Last.fm API key (at least one)

## Installation

```shell
uv sync                      # Prefect is a core dependency
```

Copy `.env.example` to `.env` and set the relevant variables:

**ListenBrainz source (recommended):**
- `LB_TOKEN` — personal API token from https://listenbrainz.org/settings/
- `LB_USER` — target username

**Last.fm source:**
- `LASTFM_API_KEY` — from https://www.last.fm/api/account/create
- `LASTFM_USER` — target username

**Source selection (optional):**
- `C9R_SOURCE` — `lastfm` or `listenbrainz`

**Prefect server (optional):**
- `PREFECT_API_URL` — e.g. `http://localhost:4200/api`

## Running the Pipeline

### Via the CLI

```shell
uv run c9r flow                                  # uses C9R_SOURCE env var
uv run c9r flow --source listenbrainz            # explicit source
uv run c9r flow --full --source listenbrainz     # full history fetch
```

Add `-v` before `flow` for debug logging:

```shell
uv run c9r -v flow
```

### Direct execution

```shell
uv run python flows/cf_ingest.py
```

Running the file directly starts a Prefect `.serve()` deployment with a weekly cron schedule (Mondays at 03:00 UTC).

### Using Prefect UI

1. **Start the Prefect server:**
   ```shell
   uv run prefect server start
   ```
   UI at http://localhost:4200

2. **Serve the flow with a custom schedule:**
   ```python
   from flows.cf_ingest import weekly_ingest_flow

   weekly_ingest_flow.serve(
       name="c9r-weekly-ingest",
       cron="0 3 * * 1",  # every Monday at 03:00 UTC
       tags=["c9r", "scrobble"],
   )
   ```

## Flow Structure

`flows/cf_ingest.py` defines one flow (`c9r_ingest`) with a housekeeping step plus eight tasks, covering functional requirements FR-01 through FR-07 and FR-10:

0. **janitor_zombie_runs** — marks `c9r_ingest` runs stuck in Running/Pending for over 6 h as Crashed (a serve() runner only writes a run's final state while its process is alive, so host/process deaths would otherwise leave zombies forever). Exempts the calling run; failures only log a warning.
1. **fetch_scrobbles** (FR-01→03) — ingests scrobbles from the selected source; persists to year-partitioned `PQ/scrobble/year=YYYY/`. Runs post-ingestion QA checks (schema, timestamps, duplicates, encoding). Retries: 8, exponential backoff (factor 10) + jitter.
2. **fix_encoding_task** — repairs encoding-corrupted strings in scrobble and artist_info. Retries: 3.
3. **enrich_artists** (FR-04) — looks up country/MBID for unresolved artists via the local MusicBrainz mirror; persists to `artist_info.parquet`. Retries: 8, exponential backoff (factor 10) + jitter.
4. **clean_artists** — deduplicates `artist_info.parquet`. Retries: 3.
5. **canonise_batch** (FR-05) — loads `ML/lightgbm_best.pkl` in-process, runs batch variant discovery via RapidFuzz pre-filter + ML classification, writes new candidate groups to `avc.parquet` with `to_link=NULL`. Logs every prediction (timestamp, pair, probability, feature vector) to `predictions_log.parquet`. Skips gracefully when the model pickle is missing.
6. **propagate_avc_task** (FR-05) — applies decided AVC mappings to `artist_info.parquet` (renames variants, appends aliases).
7. **augment_gs_task** (FR-06) — refreshes gold-standard training pairs from the local MusicBrainz Docker mirror into `gs_mb.parquet`. Skips gracefully when the mirror is unreachable.
8. **retrain_model_task** (FR-07) — retrains the LightGBM pipeline on current gold-standard data with 5-fold CV and MLflow logging. Exports the best model pickle(s) to `ML/`. Skips when fewer than 20 decided AVC rows exist.

## Scheduling

The default deployment (via `python flows/cf_ingest.py`) uses `cron="0 3 * * 1"` (Mondays at 03:00 UTC). The flow carries `timeout_seconds=8700` (2 h 25 min), so a hung run fails instead of lingering.

Common cron patterns for `flow.serve(cron=...)`:
- Weekly (Monday 03:00 UTC): `"0 3 * * 1"`
- Daily: `"0 0 * * *"`
- Monthly: `"0 0 1 * *"`

## Return Value

The flow returns a dict with these keys:
- `new_scrobbles` — number of scrobbles fetched
- `encoding_fixed` — number of encoding repairs
- `enriched_artists` — artists enriched from MB
- `cleaned` / `remaining` — dedup stats
- `flagged_for_review` / `skipped` — canonisation counts
- `avc_propagated` / `avc_aliases_added` — propagation stats
- `gs_rows_written` — gold-standard refresh count
- `models_trained` — retraining count
- `zombies_crashed` — stale runs marked Crashed by the janitor
- `duration` — wall-clock time

## Troubleshooting

1. **"LASTFM_USER not set" / "LB_USER not set"** — set the matching env var in `.env` for the source you chose.
2. **MusicBrainz rate limiting** — retry logic handles transient 503s; persistent failures may mean the MB endpoint is overloaded. The local Docker mirror bypasses the 1 req/s API limit.
3. **Canonisation skipped** — the model pickle `ML/lightgbm_best.pkl` must exist. Run `uv run c9r train run` first.
4. **Gold-standard augmentation skipped** — requires a running local MusicBrainz Docker mirror (see MetaBrainz/musicbrainz-docker).
5. **Retraining skipped** — needs at least 20 decided (non-NULL `to_link`) rows in `avc.parquet`. Run `uv run c9r canon human` to review pending candidates.

## Development

1. Edit tasks in `flows/cf_ingest.py`
2. Underlying logic lives in `corefunc/canon/workflow.py`, `corefunc/enrich.py`, `corefunc/data_cleaning.py`, `corefunc/canon/trainer.py`, `corefunc/canon/augment.py`
3. Run tests: `uv run pytest tests/ -v --no-cov -k prefect`

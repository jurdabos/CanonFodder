# Prefect Setup Guide

This guide explains how to set up and run the CanonFodder data pipeline using Prefect.

## Prerequisites

- Python 3.12+
- Prefect 3.0+
- MySQL or compatible database
- Last.fm API key
- MusicBrainz API access (optional)

## Installation

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```
   
   Required variables:
   - `LASTFM_API_KEY`: Your Last.fm API key
   - `LASTFM_USER`: Your Last.fm username
   - `DB_URL`: Database connection string
   
   Optional Prefect configuration:
   - `PREFECT_API_URL`: URL of Prefect server (if using)
   - `PREFECT_API_KEY`: API key for authentication

## Running the Pipeline

### Local Execution

To run the pipeline immediately:

```bash
python flows/cf_ingest.py
```

### Using Prefect UI

1. **Start the Prefect server:**
   ```bash
   prefect server start
   ```
   
   The UI will be available at http://localhost:4200

2. **Create a deployment:**
   ```python
   from flows.cf_ingest import weekly_ingest_flow
   from prefect.deployments import Deployment
   from prefect.schedules import CronSchedule
   
   deployment = Deployment.build_from_flow(
       flow=weekly_ingest_flow,
       name="weekly-ingest",
       schedule=CronSchedule(cron="0 0 * * 0", timezone="UTC"),
       work_queue_name="default",
       tags=["canonfodder", "lastfm", "musicbrainz"]
   )
   deployment.apply()
   ```

3. **Start a worker:**
   ```bash
   prefect worker start --pool default
   ```

### Manual Flow Execution

To run the flow manually via Prefect CLI:

```bash
python -m prefect run flows.cf_ingest:weekly_ingest_flow
```

## Flow Structure

The CanonFodder pipeline consists of the following tasks:

1. **fetch_new_scrobbles**: Fetches new tracks from Last.fm
2. **enrich_artist_info**: Enriches artist data from MusicBrainz
3. **clean_artist_data**: Removes duplicates and orphaned records
4. **run_canonization**: Groups artist name variants
5. **export_to_parquet**: Exports data to Parquet format
6. **run_data_profiling**: Generates analytics for BI dashboards

Each task includes:
- Automatic retries with exponential backoff
- Comprehensive logging
- Error handling
- Progress tracking

## Monitoring

### Prefect UI

The Prefect UI provides:
- Real-time flow execution monitoring
- Task status and logs
- Flow run history
- Performance metrics

### Logs

Prefect logs are available in:
- The Prefect UI
- Console output when running locally
- Configured log handlers (if set up)

## Scheduling

The pipeline is configured to run weekly by default. To modify the schedule:

1. Update the deployment configuration
2. Change the `CronSchedule` pattern
3. Re-apply the deployment

Common schedule patterns:
- Daily: `"0 0 * * *"`
- Weekly: `"0 0 * * 0"`
- Monthly: `"0 0 1 * *"`

## Troubleshooting

### Common Issues

1. **No Last.fm username error:**
   - Ensure `LASTFM_USER` is set in your `.env` file
   
2. **Database connection failed:**
   - Verify `DB_URL` is correctly configured
   - Check database is running and accessible
   
3. **MusicBrainz rate limiting:**
   - The pipeline includes automatic retry logic
   - Consider reducing request frequency if persistent

### Debug Mode

To enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

The pipeline is designed to:
- Process 1 million scrobbles in under 15 minutes
- Handle API rate limits gracefully
- Minimize database queries through bulk operations
- Support incremental data loading

## Development

To modify the pipeline:

1. Edit tasks in `flows/cf_ingest.py`
2. Update pipeline functions in `corefunc/pipeline.py`
3. Run tests: `pytest tests/test_prefect_flow.py`
4. Apply changes to deployment

## Migration from Airflow

If you're migrating from Airflow:

1. The DAGs have been replaced with Prefect flows
2. XCom is replaced with direct task returns
3. Operators are replaced with `@task` decorators
4. The scheduler is replaced with Prefect deployments

Key differences:
- No DAG files needed
- Simpler dependency management
- Better local development experience
- Modern Python async support

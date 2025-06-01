# CanonFodder Airflow Integration

This directory contains the Airflow DAGs for the CanonFodder project. The DAGs orchestrate the data pipeline for fetching, processing, and analyzing music scrobble data from Last.fm.

## Overview

The main DAG (`cf_ingest`) implements a complete data pipeline that covers all functional requirements:

1. **Fetch scrobbles (FR-01)** - Pull recent tracks for a user since the last stored timestamp and persist raw JSON
2. **Normalize scrobbles (FR-02)** - Rename columns, convert UTS→UTC datetime, and remove duplicates
3. **Bulk insert (FR-03)** - Load normalized rows into scrobble table with dialect-aware conflict ignore
4. **Artist enrichment (FR-04)** - For new MBIDs, fetch country & aliases; cache in artistcountry
5. **Canonization (FR-05)** - Group artist name variants, store mapping in AVC, apply to scrobble history
6. **Parquet export (FR-06)** - Dump star schema to parquet files on demand or based on schedule
7. **BI frontend (FR-07)** - Provide menu to launch data gathering and open interactive dashboards

The DAG includes proper retry and back-off mechanisms (FR-08) and is designed to complete within 15 minutes for 1 million scrobbles (FR-09).

## DAG Structure

The `cf_ingest` DAG consists of the following tasks:

1. `fetch_new_scrobbles` - Fetches new scrobbles from Last.fm
2. `enrich_artist_info` - Enriches artist information from MusicBrainz
3. `clean_artist_data` - Cleans up artist data by removing duplicates and orphaned records
4. `run_canonization` - Runs canonization to group artist name variants
5. `export_to_parquet` - Exports data to parquet files
6. `run_data_profiling` - Runs data profiling to generate analytics

The tasks are executed in sequence, with each task depending on the successful completion of the previous task.

## Running with Airflow

### Using Docker

The easiest way to run the CanonFodder Airflow integration is using Docker:

1. Make sure you have Docker and Docker Compose installed
2. Create a `.env` file with your Last.fm API key and username (see `.env.example`)
3. Run `docker-compose up -d` to start the services
4. Access the Airflow web interface at http://localhost:8080
5. Log in with username `admin` and password `admin`
6. Enable the `cf_ingest` DAG
7. Trigger the DAG manually or wait for the scheduled run

### Manual Setup

If you prefer to run Airflow without Docker:

1. Install Airflow: `pip install apache-airflow~=3.0.1`
2. Set the `AIRFLOW_HOME` environment variable to your preferred location
3. Initialize the Airflow database: `airflow db init`
4. Create an admin user: `airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin`
5. Set the `AIRFLOW__CORE__DAGS_FOLDER` environment variable to the `dags` directory in your CanonFodder project
6. Start the Airflow webserver: `airflow webserver -p 8080`
7. Start the Airflow scheduler: `airflow scheduler`
8. Access the Airflow web interface at http://localhost:8080
9. Log in with the credentials you created
10. Enable the `cf_ingest` DAG
11. Trigger the DAG manually or wait for the scheduled run

## Configuration

The DAG is configured to run weekly by default. You can change the schedule by modifying the `schedule_interval` parameter in the DAG definition.

The retry mechanism is configured with 8 retries and exponential back-off, starting with a 30-second delay and capping at 10 minutes. This ensures robust handling of network issues and API rate limits.

## Environment Variables

The following environment variables are used by the DAG:

- `LASTFM_USER` - Your Last.fm username
- `LASTFM_API_KEY` - Your Last.fm API key
- `DB_URL` - Database connection URL

Make sure these variables are set in your environment or in the `.env` file if using Docker.

## Monitoring

You can monitor the progress of the DAG runs in the Airflow web interface. The interface provides detailed information about task execution, logs, and any errors that may occur.

## Troubleshooting

If you encounter issues with the DAG:

1. Check the task logs in the Airflow web interface
2. Verify that your environment variables are set correctly
3. Ensure that the Last.fm API is accessible
4. Check that the database connection is working

For more detailed troubleshooting, refer to the Airflow documentation.
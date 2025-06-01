"""
Airflow DAG for CanonFodder data pipeline.
This DAG orchestrates the complete data pipeline for CanonFodder, including:
- Fetching new scrobbles from Last.fm (FR-01)
- Normalizing scrobbles (FR-02)
- Bulk inserting into database (FR-03)
- Artist enrichment from MusicBrainz (FR-04)
- Canonization of artist name variants (FR-05)
- Parquet export (FR-06)
- Data profiling for BI frontend (FR-07)

The DAG includes proper retry and back-off mechanisms (FR-08) and
is designed to complete within 15 minutes for 1 million scrobbles (FR-09).
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import sys
import logging
import math

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CanonFodder modules
from HTTP import mbAPI

# Initialize MusicBrainz API
mbAPI.init()

# Default arguments for the DAG with exponential back-off retry mechanism (FR-08)
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 8,  # Up to 8 retries as per FR-08
    'retry_delay': timedelta(seconds=30),  # Start with 30 seconds
    'retry_exponential_backoff': True,  # Enable exponential back-off
    'max_retry_delay': timedelta(minutes=10),  # Cap at 10 minutes
}

# Create the DAG
dag = DAG(
    'cf_ingest',  # Renamed to cf_ingest as per FR-09
    default_args=default_args,
    description='Complete CanonFodder data pipeline covering FR-01 to FR-07',
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,
    tags=['canonfodder', 'lastfm', 'musicbrainz', 'data-pipeline'],
)

def fetch_new_scrobbles(**context):
    """
    Fetch new scrobbles from Last.fm since the last run (FR-01, FR-02, FR-03).

    This task:
    - Pulls recent tracks for a user since the last stored timestamp
    - Persists raw JSON
    - Normalizes scrobbles (rename columns, convert UTS→UTC datetime)
    - Removes duplicates
    - Bulk inserts into database with conflict handling
    """
    logging.info("Starting fetch of new scrobbles")

    # Get username from environment variable
    username = os.getenv("LASTFM_USER")
    if not username:
        raise ValueError("No Last.fm username provided. Set LASTFM_USER environment variable.")

    # Use the pipeline module to fetch new data
    from corefunc.pipeline import fetch_new_data

    result = fetch_new_data(username)

    if result['status'] == 'error':
        raise ValueError(f"Error fetching new data: {result['message']}")

    if result['new_scrobbles'] == 0:
        logging.info("No new scrobbles since last run – nothing to do.")
        return None

    logging.info(f"Fetched {result['new_scrobbles']} new scrobbles")

    return result['new_scrobbles']

def enrich_artist_info(**context):
    """
    Enrich artist information from MusicBrainz (FR-04).

    This task:
    - For new MBIDs, fetches country & aliases
    - Caches results in artistcountry table
    """
    logging.info("Starting artist info enrichment")

    # Get the number of new scrobbles from the previous task
    ti = context['ti']
    new_scrobbles = ti.xcom_pull(task_ids='fetch_new_scrobbles')

    if new_scrobbles is None:
        logging.info("No new scrobbles to process - continuing with artist enrichment anyway")
    else:
        logging.info(f"Processing artist info for {new_scrobbles} new scrobbles")

    # Use the pipeline module to enrich artist data
    from corefunc.pipeline import enrich_artist_data

    result = enrich_artist_data()

    if result['status'] == 'error':
        raise ValueError(f"Error enriching artist data: {result['message']}")

    logging.info(f"Artist info enrichment complete: processed={result['processed']}, created={result['created']}, updated={result['updated']}")

    return {
        'processed': result['processed'],
        'created': result['created'],
        'updated': result['updated']
    }

def clean_artist_data(**context):
    """
    Clean up artist data by removing duplicates and orphaned records.
    """
    logging.info("Starting artist data cleanup")

    # Use the pipeline module to clean artist data
    from corefunc.pipeline import clean_artist_data as clean_artist_data_func

    result = clean_artist_data_func()

    if result['status'] == 'error':
        raise ValueError(f"Error cleaning artist data: {result['message']}")

    logging.info(f"Artist data cleanup complete: removed {result['cleaned']} records, {result['remaining']} remain")

    return {
        'cleaned': result['cleaned'],
        'remaining': result['remaining']
    }


def run_canonization(**context):
    """
    Run canonization to group artist name variants (FR-05).

    This task:
    - Groups artist name variants
    - Stores mapping in ArtistVariantsCanonized table
    - Applies canonization to scrobble history
    """
    logging.info("Starting artist name canonization")

    # Use the pipeline module to run canonization
    from corefunc.pipeline import run_canonization as run_canonization_func

    result = run_canonization_func()

    if result['status'] == 'error':
        raise ValueError(f"Error running canonization: {result['message']}")

    logging.info(f"Canonization complete: processed {result['row_count']} rows with {result['artist_count']} unique artists")

    return {
        'row_count': result['row_count'],
        'artist_count': result['artist_count'],
        'data_source': result['data_source']
    }


def export_to_parquet(**context):
    """
    Export data to parquet files (FR-06).

    This task:
    - Dumps star schema to parquet files
    - Used for analytics and BI dashboards
    """
    logging.info("Starting parquet export")

    # Use the pipeline module to export to parquet
    from corefunc.pipeline import export_to_parquet as export_to_parquet_func

    result = export_to_parquet_func()

    if result['status'] == 'error':
        raise ValueError(f"Error exporting to parquet: {result['message']}")

    logging.info(f"Parquet export complete: {result['parquet_path']}")

    return {
        'parquet_path': str(result['parquet_path'])
    }

def run_data_profiling(**context):
    """
    Run data profiling to generate analytics (FR-07).

    This task:
    - Generates analytics for BI dashboards
    - Prepares data for interactive exploration
    """
    logging.info("Starting data profiling")

    # Use the pipeline module to run data profiling
    from corefunc.pipeline import run_data_profiling as run_data_profiling_func

    result = run_data_profiling_func()

    if result['status'] == 'error':
        raise ValueError(f"Error running data profiling: {result['message']}")

    logging.info("Data profiling complete")

    return True

# Define the tasks
fetch_task = PythonOperator(
    task_id='fetch_new_scrobbles',
    python_callable=fetch_new_scrobbles,
    provide_context=True,
    dag=dag,
)

enrich_task = PythonOperator(
    task_id='enrich_artist_info',
    python_callable=enrich_artist_info,
    provide_context=True,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_artist_data',
    python_callable=clean_artist_data,
    provide_context=True,
    dag=dag,
)

canonization_task = PythonOperator(
    task_id='run_canonization',
    python_callable=run_canonization,
    provide_context=True,
    dag=dag,
)

parquet_task = PythonOperator(
    task_id='export_to_parquet',
    python_callable=export_to_parquet,
    provide_context=True,
    dag=dag,
)

profile_task = PythonOperator(
    task_id='run_data_profiling',
    python_callable=run_data_profiling,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
fetch_task >> enrich_task >> clean_task >> canonization_task >> parquet_task >> profile_task
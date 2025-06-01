"""
Airflow DAG for weekly CanonFodder data pipeline.

This DAG orchestrates the weekly autofetch of new data from Last.fm,
merges it with previous data, and enriches artist information from MusicBrainz.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import sys
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CanonFodder modules
from HTTP import mbAPI

# Initialize MusicBrainz API
mbAPI.init()

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'canonfodder_weekly_autofetch',
    default_args=default_args,
    description='Weekly autofetch of new data from Last.fm and enrichment from MusicBrainz',
    schedule_interval='@weekly',
    start_date=days_ago(1),
    catchup=False,
    tags=['canonfodder', 'lastfm', 'musicbrainz'],
)

def fetch_new_scrobbles(**context):
    """
    Fetch new scrobbles from Last.fm since the last run.
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
    Enrich artist information from MusicBrainz.
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

def run_data_profiling(**context):
    """
    Run data profiling to generate analytics.
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

profile_task = PythonOperator(
    task_id='run_data_profiling',
    python_callable=run_data_profiling,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
fetch_task >> enrich_task >> clean_task >> profile_task

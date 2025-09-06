"""
Tests for the Prefect flow in flows/cf_ingest.py.

This module tests the CanonFodder Prefect flow, including individual tasks
and the complete flow execution with proper mocking of external dependencies.
"""

import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, patch, call
from prefect.testing.utilities import prefect_test_harness

# Import the flow and tasks
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flows.cf_ingest import (
    fetch_new_scrobbles,
    enrich_artist_info,
    clean_artist_data,
    run_canonization,
    export_to_parquet,
    run_data_profiling,
    weekly_ingest_flow
)


@pytest.fixture(autouse=True)
def prefect_test_fixture():
    """Setting up Prefect test harness for all tests."""
    with prefect_test_harness():
        yield


@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    with patch.dict(os.environ, {"LASTFM_USER": "test_user"}):
        yield


@pytest.fixture
def mock_pipeline_functions():
    """Mock all pipeline functions."""
    with patch("flows.cf_ingest.fetch_new_data") as mock_fetch, \
         patch("flows.cf_ingest.enrich_artist_data") as mock_enrich, \
         patch("flows.cf_ingest.clean_artist_data_func") as mock_clean, \
         patch("flows.cf_ingest.run_canonization_func") as mock_canon, \
         patch("flows.cf_ingest.export_to_parquet_func") as mock_export, \
         patch("flows.cf_ingest.run_data_profiling_func") as mock_profile:
        
        # Set up return values
        mock_fetch.return_value = {
            'status': 'success',
            'message': 'Successfully fetched 100 new scrobbles',
            'new_scrobbles': 100,
            'latest_timestamp': 1234567890
        }
        
        mock_enrich.return_value = {
            'status': 'success',
            'message': 'Successfully enriched artist data',
            'processed': 50,
            'created': 10,
            'updated': 5
        }
        
        mock_clean.return_value = {
            'status': 'success',
            'message': 'Successfully cleaned artist data',
            'cleaned': 3,
            'remaining': 47
        }
        
        mock_canon.return_value = {
            'status': 'success',
            'message': 'Successfully applied canonization',
            'row_count': 100,
            'artist_count': 45,
            'data_source': 'parquet'
        }
        
        mock_export.return_value = {
            'status': 'success',
            'message': 'Successfully exported to parquet',
            'parquet_path': '/path/to/parquet'
        }
        
        mock_profile.return_value = {
            'status': 'success',
            'message': 'Successfully ran data profiling'
        }
        
        yield {
            'fetch': mock_fetch,
            'enrich': mock_enrich,
            'clean': mock_clean,
            'canon': mock_canon,
            'export': mock_export,
            'profile': mock_profile
        }


def test_fetch_new_scrobbles_success(mock_environment, mock_pipeline_functions):
    """Test successful fetching of new scrobbles."""
    result = fetch_new_scrobbles()
    
    assert result == 100
    mock_pipeline_functions['fetch'].assert_called_once_with("test_user")


def test_fetch_new_scrobbles_no_user():
    """Test fetch_new_scrobbles when no username is provided."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="No Last.fm username provided"):
            fetch_new_scrobbles()


def test_fetch_new_scrobbles_error(mock_environment):
    """Test fetch_new_scrobbles when an error occurs."""
    with patch("flows.cf_ingest.fetch_new_data") as mock_fetch:
        mock_fetch.return_value = {
            'status': 'error',
            'message': 'API error occurred',
            'new_scrobbles': 0,
            'latest_timestamp': None
        }
        
        with pytest.raises(ValueError, match="Error fetching new data"):
            fetch_new_scrobbles()


def test_fetch_new_scrobbles_no_new_data(mock_environment):
    """Test fetch_new_scrobbles when there are no new scrobbles."""
    with patch("flows.cf_ingest.fetch_new_data") as mock_fetch:
        mock_fetch.return_value = {
            'status': 'success',
            'message': 'No new scrobbles',
            'new_scrobbles': 0,
            'latest_timestamp': 1234567890
        }
        
        result = fetch_new_scrobbles()
        assert result is None


def test_enrich_artist_info_success(mock_pipeline_functions):
    """Test successful artist info enrichment."""
    result = enrich_artist_info(100)
    
    assert result['processed'] == 50
    assert result['created'] == 10
    assert result['updated'] == 5
    mock_pipeline_functions['enrich'].assert_called_once()


def test_enrich_artist_info_no_scrobbles(mock_pipeline_functions):
    """Test artist info enrichment with no new scrobbles."""
    result = enrich_artist_info(None)
    
    assert result['processed'] == 50
    mock_pipeline_functions['enrich'].assert_called_once()


def test_clean_artist_data_success(mock_pipeline_functions):
    """Test successful artist data cleaning."""
    enrichment_result = {'processed': 50, 'created': 10, 'updated': 5}
    result = clean_artist_data(enrichment_result)
    
    assert result['cleaned'] == 3
    assert result['remaining'] == 47
    mock_pipeline_functions['clean'].assert_called_once()


def test_run_canonization_success(mock_pipeline_functions):
    """Test successful canonization."""
    cleanup_result = {'cleaned': 3, 'remaining': 47}
    result = run_canonization(cleanup_result)
    
    assert result['row_count'] == 100
    assert result['artist_count'] == 45
    assert result['data_source'] == 'parquet'
    mock_pipeline_functions['canon'].assert_called_once()


def test_export_to_parquet_success(mock_pipeline_functions):
    """Test successful parquet export."""
    canonization_result = {
        'row_count': 100,
        'artist_count': 45,
        'data_source': 'parquet'
    }
    result = export_to_parquet(canonization_result)
    
    assert result['parquet_path'] == '/path/to/parquet'
    mock_pipeline_functions['export'].assert_called_once()


def test_run_data_profiling_success(mock_pipeline_functions):
    """Test successful data profiling."""
    parquet_result = {'parquet_path': '/path/to/parquet'}
    result = run_data_profiling(parquet_result)
    
    assert result is True
    mock_pipeline_functions['profile'].assert_called_once()


def test_weekly_ingest_flow_complete(mock_environment, mock_pipeline_functions):
    """Test the complete weekly ingest flow."""
    result = weekly_ingest_flow()
    
    # Check that all tasks were called
    mock_pipeline_functions['fetch'].assert_called_once()
    mock_pipeline_functions['enrich'].assert_called_once()
    mock_pipeline_functions['clean'].assert_called_once()
    mock_pipeline_functions['canon'].assert_called_once()
    mock_pipeline_functions['export'].assert_called_once()
    mock_pipeline_functions['profile'].assert_called_once()
    
    # Check the result
    assert result['new_scrobbles'] == 100
    assert result['enrichment']['processed'] == 50
    assert result['cleanup']['cleaned'] == 3
    assert result['canonization']['row_count'] == 100
    assert result['parquet_path'] == '/path/to/parquet'
    assert result['profiling_complete'] is True
    assert 'duration' in result


def test_weekly_ingest_flow_no_new_scrobbles(mock_environment):
    """Test the flow when there are no new scrobbles."""
    with patch("flows.cf_ingest.fetch_new_data") as mock_fetch, \
         patch("flows.cf_ingest.enrich_artist_data") as mock_enrich, \
         patch("flows.cf_ingest.clean_artist_data_func") as mock_clean, \
         patch("flows.cf_ingest.run_canonization_func") as mock_canon, \
         patch("flows.cf_ingest.export_to_parquet_func") as mock_export, \
         patch("flows.cf_ingest.run_data_profiling_func") as mock_profile:
        
        # No new scrobbles
        mock_fetch.return_value = {
            'status': 'success',
            'message': 'No new scrobbles',
            'new_scrobbles': 0,
            'latest_timestamp': 1234567890
        }
        
        # Other functions should still work
        mock_enrich.return_value = {'status': 'success', 'processed': 0, 'created': 0, 'updated': 0}
        mock_clean.return_value = {'status': 'success', 'cleaned': 0, 'remaining': 50}
        mock_canon.return_value = {'status': 'success', 'row_count': 0, 'artist_count': 0, 'data_source': 'database'}
        mock_export.return_value = {'status': 'success', 'parquet_path': '/path/to/parquet'}
        mock_profile.return_value = {'status': 'success'}
        
        result = weekly_ingest_flow()
        
        # Verify the flow completed even with no new scrobbles
        assert result['new_scrobbles'] is None
        assert 'duration' in result


def test_task_retry_on_failure(mock_environment):
    """Test that tasks retry on failure."""
    with patch("flows.cf_ingest.fetch_new_data") as mock_fetch:
        # First call fails, second succeeds
        mock_fetch.side_effect = [
            {'status': 'error', 'message': 'Temporary failure'},
            {'status': 'success', 'new_scrobbles': 50, 'latest_timestamp': 1234567890}
        ]
        
        # This should fail and trigger retry logic
        # In a real Prefect environment, this would retry automatically
        # For testing, we just verify the error is raised
        with pytest.raises(ValueError):
            fetch_new_scrobbles()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=flows", "--cov-report=term-missing"])

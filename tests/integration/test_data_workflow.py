"""
Integration tests for the data gathering workflow.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import run_data_gathering_workflow


class TestDataGatheringWorkflow:
    """Test the data gathering workflow with mocked dependencies."""
    
    @patch('main.lfAPI.fetch_scrobbles_since')
    @patch('main.lfAPI.sync_user_country')
    @patch('main.bulk_insert_scrobbles')
    @patch('main.populate_artist_info_from_scrobbles')
    @patch('main.dump_parquet')
    def test_workflow_with_mocked_dependencies(self, 
                                             mock_dump_parquet, 
                                             mock_populate_artist_info, 
                                             mock_bulk_insert, 
                                             mock_sync_country, 
                                             mock_fetch_scrobbles):
        """Test the workflow with all external dependencies mocked."""
        # Setup mock data
        mock_username = "test_user"
        
        # Create a mock DataFrame to return from fetch_scrobbles_since
        mock_df = pd.DataFrame({
            'artist_name': ['Artist1', 'Artist2'],
            'track_title': ['Track1', 'Track2'],
            'album_title': ['Album1', 'Album2'],
            'play_time': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
        })
        mock_fetch_scrobbles.return_value = mock_df
        
        # Mock return values
        mock_sync_country.return_value = True
        
        # Create a mock progress callback
        mock_callback = MagicMock()
        
        # Call the function
        result = run_data_gathering_workflow(mock_username, mock_callback)
        
        # Assertions
        mock_fetch_scrobbles.assert_called_once_with(mock_username, None)
        mock_bulk_insert.assert_called_once()
        mock_dump_parquet.assert_called_once()
        mock_populate_artist_info.assert_called_once()
        
        # Check that the progress callback was called multiple times
        assert mock_callback.call_count > 0
        
        # Check that the function completes successfully
        assert result is None
    
    @patch('main.lfAPI.fetch_scrobbles_since')
    def test_workflow_with_empty_dataframe(self, mock_fetch_scrobbles):
        """Test the workflow when no new scrobbles are found."""
        # Setup empty DataFrame
        mock_fetch_scrobbles.return_value = pd.DataFrame()
        
        # Create a mock progress callback
        mock_callback = MagicMock()
        
        # Call the function
        result = run_data_gathering_workflow("test_user", mock_callback)
        
        # Assertions
        assert result is None
        
        # Verify that the callback was called with completion message
        mock_callback.assert_any_call("Complete", 100, "No new scrobbles to process")

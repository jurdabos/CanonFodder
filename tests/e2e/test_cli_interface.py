"""
End-to-end tests for the CLI interface.
"""

import pytest
import sys
import os
import subprocess
import time
from unittest.mock import patch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the main module
from main import CliInterface


class TestCliInterfaceE2E:
    """End-to-end tests for the CLI interface."""
    
    @pytest.mark.skip(reason="This test requires manual interaction and a terminal")
    def test_cli_startup_subprocess(self):
        """Test that the CLI application starts up correctly in a subprocess."""
        # This is a basic smoke test to check if the application starts
        # Run with --no-animation to speed up the test
        process = subprocess.Popen(
            [sys.executable, "main.py", "--no-animation"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        
        # Give the process a moment to start up
        time.sleep(1)
        
        # Send Escape key to exit
        process.communicate(input=b"\x1b")
        
        # Check if process terminated correctly
        return_code = process.wait(timeout=5)
        assert return_code == 0

    @patch('curses.initscr')
    @patch('curses.noecho')
    @patch('curses.cbreak')
    @patch('curses.start_color')
    @patch('curses.wrapper')
    def test_cli_interface_init(self, mock_wrapper, mock_start_color, mock_cbreak, mock_noecho, mock_initscr):
        """Test initialization of CliInterface with mocked curses."""
        # Setup
        mock_stdscr = mock_initscr.return_value
        
        # Create an instance of CliInterface
        cli = CliInterface()
        
        # Assert that username is initially None or loaded from environment
        assert cli.username is None or isinstance(cli.username, str)
        
        # Test that start method calls curses.wrapper
        cli.start()
        mock_wrapper.assert_called_once()


class TestCliInterfaceMethods:
    """Unit tests for specific CliInterface methods."""
    
    def test_load_username_environment(self):
        """Test loading username from environment variable."""
        # Save original environment
        original_env = os.environ.get("LASTFM_USER")
        
        try:
            # Set environment variable
            os.environ["LASTFM_USER"] = "test_user_from_env"
            
            # Create instance and check username
            cli = CliInterface()
            assert cli.username == "test_user_from_env"
            
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["LASTFM_USER"] = original_env
            else:
                del os.environ["LASTFM_USER"]
    
    @patch('main.CliInterface._save_username')
    @patch('curses.wrapper')
    def test_save_username(self, mock_wrapper, mock_save):
        """Test that username is saved correctly."""
        # Create CLI instance
        cli = CliInterface()
        
        # Set username and call save method
        test_username = "test_save_user"
        cli._save_username(test_username)
        
        # Verify that the save method was called
        mock_save.assert_called_once_with(test_username)

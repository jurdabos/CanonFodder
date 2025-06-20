"""
Unit tests for the main module's core functions.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from main import null_progress_callback, init_colors, ProgressCallback, Colors


def test_null_progress_callback():
    """Test the null_progress_callback function returns None and doesn't raise errors."""
    null_progress_callback("test_task", 50.0, "Testing message")
    assert result is None


def test_init_colors():
    """Test that init_colors runs without errors."""
    # This mainly tests that the function doesn't raise any exceptions
    try:
        init_colors()
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"init_colors raised an exception: {e}")
    
    assert success is True


class TestProgressCallback:
    """Test the ProgressCallback protocol implementation."""
    
    def test_progress_callback_protocol(self):
        """Test that a concrete class implementing ProgressCallback works as expected."""
        
        # Create a concrete implementation of ProgressCallback
        class ConcreteProgressCallback:
            def __init__(self):
                self.last_task = None
                self.last_percentage = None
                self.last_message = None
                
            def __call__(self, task, percentage, message=None):
                self.last_task = task
                self.last_percentage = percentage
                self.last_message = message
        
        # Test the implementation
        callback = ConcreteProgressCallback()
        callback("test_task", 75.5, "Progress message")
        
        assert callback.last_task == "test_task"
        assert callback.last_percentage == 75.5
        assert callback.last_message == "Progress message"


class TestColors:
    """Test the Colors class."""
    
    def test_colors_attributes(self):
        """Test that the Colors class has the expected attributes."""
        # Test a sampling of color attributes
        assert hasattr(Colors, "RESET")
        assert hasattr(Colors, "WHITE")
        assert hasattr(Colors, "TEAL")
        assert hasattr(Colors, "LIME")
        assert hasattr(Colors, "BOLD")
        assert hasattr(Colors, "UNDERLINE")
        
        # Test that the attributes contain ANSI escape sequences
        assert Colors.RESET.startswith("\033[")
        assert Colors.WHITE.startswith("\033[")

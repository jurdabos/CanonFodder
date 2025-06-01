"""
Progress tracking functionality for CanonFodder.
This module provides classes and functions for tracking and reporting progress
of various tasks in the application.
"""
from __future__ import annotations
import threading
import time
from typing import Protocol, Optional


class ProgressManager:
    """
    Manages progress tracking for tasks with callback support.
    
    This class provides methods to start tasks, update progress, and complete tasks,
    with support for callback functions to report progress to the UI.
    """
    
    def __init__(self):
        """Initialize the progress manager."""
        self.current_task = None
        self.current_progress = 0
        self.callback = None
        self._stop_event = threading.Event()
        self._progress_thread = None
    
    def start_task(self, task_name: str, initial_progress: int = 0):
        """
        Start a new task with the given name and initial progress.
        
        Parameters
        ----------
        task_name : str
            Name of the task to start
        initial_progress : int, optional
            Initial progress value (0-100), by default 0
        """
        self.current_task = task_name
        self.current_progress = initial_progress
        self._stop_event.clear()
        self._progress_thread = threading.Thread(target=self._display_progress)
        self._progress_thread.daemon = True
        self._progress_thread.start()
    
    def update_progress(self, progress: int, message: str = ""):
        """
        Update the progress of the current task.
        
        Parameters
        ----------
        progress : int
            New progress value (0-100)
        message : str, optional
            Status message, by default ""
        """
        self.current_progress = progress
        if self.callback:
            self.callback(self.current_task, progress, message)
    
    def update_subtask(self, current: int, total: int, message: str = ""):
        """
        Update progress based on subtask progress (current/total).
        
        Parameters
        ----------
        current : int
            Current item number
        total : int
            Total number of items
        message : str, optional
            Status message, by default ""
        """
        if total > 0:
            # Calculate percentage based on current/total
            percentage = min(100, int((current / total) * 100))
            self.update_progress(percentage, message)
        else:
            self.update_progress(0, message)
    
    def increment_progress(self, amount: int = 1, message: str = ""):
        """
        Increment the progress by the given amount.
        
        Parameters
        ----------
        amount : int, optional
            Amount to increment progress by, by default 1
        message : str, optional
            Status message, by default ""
        """
        self.current_progress = min(100, self.current_progress + amount)
        if self.callback:
            self.callback(self.current_task, self.current_progress, message)
    
    def complete_task(self, message: str = "Task completed"):
        """
        Mark the current task as completed.
        
        Parameters
        ----------
        message : str, optional
            Completion message, by default "Task completed"
        """
        self.update_progress(100, message)
        self._stop_event.set()
        if self._progress_thread:
            self._progress_thread.join(timeout=1.0)
    
    def _display_progress(self):
        """Internal method to periodically display progress."""
        while not self._stop_event.is_set():
            if self.callback and self.current_task:
                self.callback(self.current_task, self.current_progress, None)
            time.sleep(0.1)


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""
    
    def __call__(self, task: str, percentage: float, message: Optional[str] = None) -> None:
        """
        Report progress of a task.
        
        Parameters
        ----------
        task : str
            Name of the current task
        percentage : float
            Progress percentage (0-100)
        message : str, optional
            Optional status message
        """
        ...


def null_progress_callback(task: str, percentage: float, message: Optional[str] = None) -> None:
    """No-op progress callback for when no callback is provided."""
    pass

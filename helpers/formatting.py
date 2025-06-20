"""
Formatting utilities for CanonFodder.
This module provides functions for formatting various types of data for display.
"""
import re
from typing import Optional


def format_sql_for_display(sql: str, max_width: Optional[int] = None) -> str:
    """
    Format SQL query for display with line breaks at appropriate places.

    Parameters
    ----------
    sql : str
        SQL query to format
    max_width : int, optional
        Maximum width of each line, by default None (uses terminal width)

    Returns
    -------
    str
        Formatted SQL query with line breaks
    """
    # If no max_width provided, try to get terminal width
    if max_width is None:
        try:
            import shutil
            max_width = shutil.get_terminal_size().columns - 4  # Subtract margin
        except (ImportError, AttributeError):
            max_width = 120  # Increased default if terminal size can't be determined

    # Ensure max_width is at least 80 characters to prevent excessive truncation
    max_width = max(max_width, 120)

    # Normalize whitespace
    sql = ' '.join(sql.split())

    # Split on SQL keywords
    keywords = [
        ' SELECT ', ' FROM ', ' WHERE ', ' AND ', ' OR ', ' GROUP BY ',
        ' HAVING ', ' ORDER BY ', ' LIMIT ', ' JOIN ', ' LEFT JOIN ',
        ' RIGHT JOIN ', ' INNER JOIN ', ' OUTER JOIN ', ' ON ', ' AS '
    ]

    # Process the SQL statement
    formatted_sql = sql
    for keyword in keywords:
        # Case-insensitive replacement to ensure all keywords are caught
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        formatted_sql = pattern.sub(f"\n{keyword}", formatted_sql)

    # Split into lines and trim each line
    lines = [line.strip() for line in formatted_sql.split('\n')]

    # Further split long lines if needed
    result = []
    for line in lines:
        # If line is too long, break it into chunks
        if len(line) > max_width:
            words = line.split()
            current_line = []
            current_length = 0

            for word in words:
                # If adding this word would exceed width, start a new line
                if current_length + len(word) + 1 > max_width and current_line:
                    result.append(" ".join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1  # +1 for the space

            # Add the last line
            if current_line:
                result.append(" ".join(current_line))
        else:
            result.append(line)

    # Joining with newlines for display
    return "\n".join(result)

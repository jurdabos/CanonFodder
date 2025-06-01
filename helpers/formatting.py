"""
Formatting utilities for CanonFodder.
This module provides functions for formatting various types of data for display.
"""
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
            max_width = 80  # Default if terminal size can't be determined
    
    # Normalize whitespace
    sql = ' '.join(sql.split())
    
    # Split on SQL keywords
    keywords = [
        ' SELECT ', ' FROM ', ' WHERE ', ' AND ', ' OR ', ' GROUP BY ',
        ' HAVING ', ' ORDER BY ', ' LIMIT ', ' JOIN ', ' LEFT JOIN ',
        ' RIGHT JOIN ', ' INNER JOIN ', ' OUTER JOIN ', ' ON ', ' AS '
    ]
    
    # Replace keywords with newline + keyword
    sql_parts = []
    current_part = sql
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in current_part.lower():
            parts = []
            for part in current_part.split(keyword_lower):
                if part:
                    parts.append(part)
                parts.append(keyword)
            # Remove the last keyword that was added
            parts.pop()
            current_part = ''.join(parts)
    
    # Split the SQL into parts
    sql_parts = current_part.split()
    
    # Combine parts into lines of appropriate width
    result = []
    current_line = []
    
    for part in sql_parts:
        # If adding this part would exceed width, start a new line
        if current_line and len(" ".join(current_line + [part])) > max_width:
            result.append(" ".join(current_line))
            current_line = [part]
        else:
            current_line.append(part)
    
    # Adding the last line
    if current_line:
        result.append(" ".join(current_line))
    
    # Joining with newlines for display
    return "\n".join(result)
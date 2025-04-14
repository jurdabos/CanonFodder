from datetime import datetime
import sqlite3


def get_db(db_path="DB/artist_unification.db"):
    """
    Returns a connection to the specified SQLite database.
    If the file does not exist, it will be created automatically.
    Also ensures the 'groups_handled' table exists.
    """
    conn = sqlite3.connect(db_path)
    create_table_if_not_exists(conn)
    return conn


def create_table_if_not_exists(conn):
    """
    Creates the 'groups_handled' table if it doesn't already exist.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groups_handled (
            group_signature TEXT PRIMARY KEY,
            canonical_name TEXT,
            timestamp TEXT
        );
    """)
    conn.commit()

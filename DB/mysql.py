from .common import DB_URL, FMT_FALLBACK, make_sessionmaker
from datetime import datetime
import helper
import pandas as pd
import re
from sqlalchemy import create_engine

engine = create_engine(DB_URL, future=True, pool_recycle=3600)
SessionLocal = make_sessionmaker(engine)          # ← exported


# ----------------------------------------------------------------
# Querying MySQL DB for latest scrobble data
# ---------------------------------------------------------------
def fetch_latest_scrobble_data_from_db(engine):
    """
    Find the most recent table named like 'scrobbles_YYYYMMDD_HHMMSS'
    and load its data into a pandas DataFrame.
    Return (df, latest_table_name) if found, or (None, None) if no matching table found.
    """
    try:
        with connection.cursor() as cursor:
            # 1) Listing tables whose names start with 'scrobbles_'
            cursor.execute("SHOW TABLES LIKE 'scrobbles_%';")
            tables = cursor.fetchall()
            if not tables:
                print("No scrobbles_ tables found in the database.")
                return None, None
            # 2) Identifying the most recent table
            pattern = re.compile(r"^scrobbles_(\d{8}_\d{6})$")
            latest_table = None
            latest_stamp = None
            for (table_name,) in tables:
                match = pattern.match(table_name)
                if match:
                    dt_str = match.group(1)
                    try:
                        dt_parsed = pd.to_datetime(dt_str, format="%Y%m%d_%H%M%S")
                    except ValueError:
                        continue
                    if (latest_stamp is None) or (dt_parsed > latest_stamp):
                        latest_stamp = dt_parsed
                        latest_table = table_name
            if not latest_table:
                print("No properly named scrobbles_YYYYMMDD_HHMMSS tables found.")
                return None, None
            print(f"Latest table determined: {latest_table}")
            # 3) Loading table contents into a DataFrame
            query = f"SELECT * FROM {latest_table};"
            df = pd.read_sql(query, engine)
            if df.empty:
                print("Latest scrobble table is empty.")
            else:
                print(f"Fetched {len(df)} rows from table: {latest_table}")
            return df, latest_table
    except Exception as e:
        print(f"Error fetching latest scrobble data: {e}")
        return None, None


# --------------------------------------------------------------
# Store recent tracks in MySQL
# --------------------------------------------------------------
def store_recent_tracks_in_db(df, connection):
    """
    Creates a new MySQL table for recent tracks and inserts each row from df.
    The table columns: artist_name, album_title, track_title, play_time, country
    """
    from DB.ops import store_recent_tracks
    table_name = f"scrobbles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    artist_name VARCHAR(2550),
                    album_title VARCHAR(2550),
                    track_title VARCHAR(2550),
                    play_time DATETIME,
                    country VARCHAR(255)
                );
            """)
            # Inserting df rows
            for _, row in df.iterrows():
                ts_obj = helper.safe_parse(row["uts"] or row["Timestamp"])
                if ts_obj is None:
                    ts_literal = "NULL"
                    ts_param = None
                elif isinstance(ts_obj, datetime):
                    ts_literal = "%s"
                    ts_param = ts_obj
                else:
                    # still a string → give MySQL a format *with* seconds
                    ts_literal = f"STR_TO_DATE(%s, '{FMT_FALLBACK}')"
                    ts_param = ts_obj
                cursor.execute(
                    f"""INSERT INTO {table_name}
                        (artist_name, album_title, track_title, play_time, country)
                        VALUES (%s, %s, %s, {ts_literal}, NULL);
                    """,
                    (row["Artist"], row["Album"], row["Song"], ts_param) if ts_param is not None
                    else (row["Artist"], row["Album"], row["Song"])
                )
            connection.commit()
            print(f"Data inserted into table {table_name}.")
        return table_name
    except Exception as e:
        print(f"Error storing recent tracks in DB: {e}")
        return None

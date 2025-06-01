from DB import get_engine
from DB.ops import load_scrobble_table_from_db_to_df
from helpers import io


def test_load_scrobble_table():
    engine = get_engine()
    print(f"Testing with engine: {engine.url}")
    
    # Test the fixed function
    df, table_name = load_scrobble_table_from_db_to_df(engine)
    
    if df is None:
        print("No scrobble table found in the database.")
    else:
        print(f"Successfully loaded {len(df)} rows from {table_name} table.")
        print(f"Columns: {df.columns.tolist()}")
        
        # Test the dump_parquet function that uses load_scrobble_table_from_db_to_df
        try:
            parquet_path = io.dump_parquet()
            print(f"Successfully dumped data to parquet: {parquet_path}")
        except Exception as e:
            print(f"Error dumping to parquet: {e}")


if __name__ == "__main__":
    test_load_scrobble_table()

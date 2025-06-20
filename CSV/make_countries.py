# %%
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()
CSV_DIR = BASE_DIR
CSV_FILE = CSV_DIR / "T_ORSZAG.csv"
PROJECT_ROOT = BASE_DIR.parent
PARQUET_DIR = PROJECT_ROOT / "PQ"
PARQUET_OUT = PARQUET_DIR / "c.parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

rename_map = {
    "ALPHA2": "ISO-2",
    "ALPHA3": "ISO-3",
    "ORSZAG_NEV_ANGOL": "en_name",
    "ORSZAG": "hu_name",
}
dtype_map = {
    "ISO-2": "CHAR(2)",
    "ISO-3": "CHAR(3)",
    "en_name": "VARCHAR(128)",
    "hu_name": "VARCHAR(128)"
}

df = (
    pd.read_csv(CSV_FILE, sep="|", dtype=str)
    .rename(columns=rename_map)
    .loc[:, rename_map.values()]  # to keep only the 4
    .fillna("")
    .sort_values("en_name", ascending=True, kind="mergesort")
    .reset_index(drop=True)
)

df.to_parquet(PARQUET_OUT, engine="pyarrow", index=False)

user = os.getenv("MyDB_USER")
pwd = os.getenv("MyDB_PASSWORD")
host = os.getenv("MyDB_HOST")
port = os.getenv("MYSQL_PORT", "3306")

# %%
db = "canonfodder"
TABLE_NAME = "country_code"

uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
engine = create_engine(uri, echo=False, future=True)

with engine.begin() as conn:                 # one open transaction
    conn.exec_driver_sql(text(
        f"DROP TABLE IF EXISTS `{TABLE_NAME}`"
    ).text)
    cols_ddl = ",\n  ".join(                 # build column list
        f"`{c}` {sql}" for c, sql in dtype_map.items()
    )
    create_sql = text(
        f"""CREATE TABLE `{TABLE_NAME}` (
          {cols_ddl}
        ) CHARACTER SET = utf8mb4
          COLLATE = utf8mb4_0900_ai_ci"""
    ).text
    conn.exec_driver_sql(create_sql)

df.to_sql(TABLE_NAME, con=engine, if_exists="append", index=False, method="multi")
print("✓ Parquet and MySQL refresh finished.")

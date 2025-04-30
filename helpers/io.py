"""
Provides small I/O helpers: pick the latest csv or parquet dump, write new
parquet snapshots, register seaborn palettes, and save dataframes to Word.
"""
from datetime import datetime, UTC
from DB import engine
from DB.ops import latest_scrobble_table_to_df
from docx import Document
from glob import glob
import os
import pandas as pd
from pathlib import Path
import re
from typing import Optional
if '__file__' in globals():
    HERE = Path(__file__).resolve().parent
else:
    HERE = Path.cwd()
CSV_DIR = Path.cwd() / "CSV"
PQ_DIR = Path.cwd() / "PQ"
PQ_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  1) parquet helpers
# ──────────────────────────────────────────────────────────────
def _parquet_name(stamp: datetime | None = None) -> Path:
    """
    Builds a timestamped parquet path inside PQ_DIR
    """
    now = datetime.now(UTC)
    stamp = stamp or now
    return PQ_DIR / f"scrobbles_{stamp:%Y%m%d_%H%M%S}.parquet"


def dump_parquet(df: pd.DataFrame | None = None,
                 *,
                 stamp: datetime | None = None) -> Path:
    """
    Writes `df` or the latest scrobble table to a new parquet file
    Args:
        df: optional dataframe to dump
        stamp: overrides the timestamp in the file name
    Returns:
        path to the written parquet file
    """
    if df is None:
        from DB.ops import latest_scrobble_table_to_df
        df, _tbl = latest_scrobble_table_to_df(engine)
        if df is None:
            raise RuntimeError("No scrobble table available to dump.")
    out = _parquet_name(stamp)
    df.to_parquet(out, index=False)
    print(f"[io] parquet written → {out}")
    return out


# ──────────────────────────────────────────────────────────────
#  2) “give me the newest …” helpers
# ──────────────────────────────────────────────────────────────
def latest_csv(user: str) -> Path | None:
    """
    Returns the newest CSV file for `user` or None when none exists
    """
    files = sorted(CSV_DIR.glob(f"{user}_*.csv"), reverse=True)
    return files[0] if files else None


def latest_parquet(*, return_df: bool = False):
    """
    Returns the newest parquet file or tuple (df path) when `return_df`
    """
    files = sorted(PQ_DIR.glob("scrobbles_*.parquet"), reverse=True)
    if not files:
        return (None, None) if return_df else None
    newest = files[0]
    if return_df:
        df = pd.read_parquet(newest)
        return df, newest
    return newest


def dump_latest_table_to_parquet() -> None:
    """
    Materialises the newest DB scrobble table as a parquet file.
    """
    df_db, latest_tbl = latest_scrobble_table_to_df(engine)
    if df_db is None:
        print("No scrobble table in DB – nothing to dump.")
        return
    pq_file = PQ_DIR / f"scrobbles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df_db.to_parquet(pq_file, index=False)
    print(f"Latest scrobble table persisted → {pq_file}")


def register_custom_palette(palette_name, palettes):
    """
    Register a custom palette in Seaborn from the palette dictionary.
    """
    import seaborn as sns
    palette = next((p for p in palettes if p["paletteName"] == palette_name), None)
    if not palette:
        raise ValueError(f"Palette {palette_name} not found in the JSON file.")
    colors = [
        f"#{color['hex']}" if not color["hex"].startswith("#") else color["hex"]
        for color in sorted(palette["colors"], key=lambda x: x["position"])
    ]
    sns.set_palette(sns.color_palette(colors))
    return colors


def save_as_word_table(dataframe, file_name):
    """
    Writes `dataframe` into a Word table saved as `file_name`.
    """
    doksi = Document()
    doksi.add_heading("Categorical Feature Summary", level=1)
    table = doksi.add_table(rows=1, cols=len(dataframe.columns))
    table.style = "Table Grid"
    for idx, column in enumerate(dataframe.columns):
        table.cell(0, idx).text = column
    for _, row in dataframe.iterrows():
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)
    doksi.save(file_name)

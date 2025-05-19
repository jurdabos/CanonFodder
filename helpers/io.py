"""
Provides small I/O helpers: pick the latest csv or parquet dump, write new
parquet snapshots, register seaborn palettes, and save dataframes to Word.
"""
from collections import Counter
from datetime import datetime, UTC
from DB import engine
from DB.ops import load_scrobble_table_from_db_to_df
from docx import Document
import pandas as pd
from pathlib import Path
import re
if '__file__' in globals():
    HERE = Path(__file__).resolve().parent
else:
    HERE = Path.cwd()
PQ_DIR = Path.cwd() / "PQ"
PQ_DIR.mkdir(exist_ok=True)
OP_TOKENS = {           # space–operator–space → token
    r"\s\-\s": "_minus_",
    r"\s\+\s": "_plus_",
    r"\s\*\s": "_mul_",
    r"\s\/\s": "_div_"
}


def _parquet_name(stamp: datetime | None = None, *, constant: bool = False) -> Path:
    """
    Builds a parquet path inside PQ_DIR
    Args:
        stamp: Optional timestamp to use in the filename
        constant: If True, returns a constant filename (scrobble.parquet) instead of timestamped
    Returns:
        Path to the parquet file
    """
    if constant:
        return PQ_DIR / "scrobble.parquet"
    now = datetime.now(UTC)
    stamp = stamp or now
    return PQ_DIR / f"scrobbles_{stamp:%Y%m%d_%H%M%S}.parquet"


def dump_latest_table_to_parquet() -> None:
    """
    Materialises the newest DB scrobble table as a parquet file.
    """
    df_db, latest_tbl = load_scrobble_table_from_db_to_df(engine)
    if df_db is None:
        print("No scrobble table in DB – nothing to dump.")
        return
    pq_file = PQ_DIR / f"scrobbles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df_db.to_parquet(pq_file, index=False)
    print(f"Latest scrobble table persisted → {pq_file}")


def append_or_create_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Appends data to an existing parquet file or creates a new one if it doesn't exist
    Args:
        df: DataFrame containing the data to append
        path: Path to the parquet file
    """
    if path.exists():
        # Read existing data and concatenate with new data
        existing_df = pd.read_parquet(path)
        # Create a column that can be used to identify duplicates across both dataframes
        if "uts" in df.columns and "play_time" in df.columns:
            # If we have both, use both for best deduplication
            dedup_cols = ["artist_name", "track_title", "play_time"]
        elif "uts" in df.columns:
            # Convert uts to play_time for deduplication if needed
            df["play_time"] = pd.to_datetime(df["uts"], unit="s", utc=True)
            dedup_cols = ["artist_name", "track_title", "play_time"]
        else:
            # Fallback to whatever columns are available
            dedup_cols = ["artist_name", "track_title"]
            if "play_time" in df.columns:
                dedup_cols.append("play_time")
        # Combine and deduplicate
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=dedup_cols, keep="last")
        combined_df.to_parquet(path, index=False)
        print(f"[io] parquet updated with {len(df)} rows → {path} (total: {len(combined_df)} rows)")
    else:
        # Just write the new data
        df.to_parquet(path, index=False)
        print(f"[io] new parquet created with {len(df)} rows → {path}")


def dump_parquet(df: pd.DataFrame | None = None,
                 *,
                 stamp: datetime | None = None,
                 constant: bool = True) -> Path:
    """
    Writes scrobble table df to a .parquet file
    Args:
        df: optional dataframe to dump
        stamp: overrides the timestamp in the file name (used only if constant=False)
        constant: if True, uses a constant filename rather than a timestamped one
    Returns:
        path to the written parquet file
    """
    if df is None:
        from DB.ops import load_scrobble_table_from_db_to_df
        df, _tbl = load_scrobble_table_from_db_to_df(engine)
        if df is None:
            raise RuntimeError("No scrobble table available to dump.")
    out = _parquet_name(stamp, constant=constant)
    if constant:
        append_or_create_parquet(df, out)
    else:
        # Legacy behavior - create a new timestamped file
        df.to_parquet(out, index=False)
        print(f"[io] parquet written → {out}")
    return out


def latest_parquet(*, return_df: bool = False, use_constant: bool = True):
    """
    Returns the scrobble parquet file (either constant or newest timestamped one)
    or tuple (df + path) when `return_df`
    Args:
        return_df: If True, returns a tuple (dataframe, path), otherwise just the path
        use_constant: If True, looks for scrobble.parquet, otherwise finds the newest timestamped file
    Returns:
        Path to the parquet file or tuple (DataFrame, Path) if return_df=True
    """
    if use_constant:
        constant_file = PQ_DIR / "scrobble.parquet"
        if constant_file.exists():
            if return_df:
                df = pd.read_parquet(constant_file)
                return df, constant_file
            return constant_file
    
    # Fall back to legacy timestamped files if constant file doesn't exist or use_constant=False
    files = sorted(PQ_DIR.glob("scrobbles_*.parquet"), reverse=True)
    if not files:
        return (None, None) if return_df else None
    newest = files[0]
    if return_df:
        df = pd.read_parquet(newest)
        return df, newest
    return newest


def load_country_timeline(path: Path) -> pd.DataFrame:
    tl = (
        pd.read_parquet(path)
        .rename(columns={"country_code": "UserCountry"})
    )
    tl["start_date"] = pd.to_datetime(tl["start_date"]).dt.normalize()
    tl["end_date"] = pd.to_datetime(tl["end_date"]).dt.normalize()
    tl.sort_values("start_date", inplace=True)
    return tl


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


def sanitize(col: str, seen: Counter) -> str:
    """
    Turn 'partial_ratio - QRatio'  →  'partial_ratio_minus_QRatio'
    Guarantees each result is a valid Python identifier **and unique**.
    """
    safe = col
    for pat, tok in OP_TOKENS.items():
        safe = re.sub(pat, tok, safe)          # to encode operator
    safe = re.sub(r"\W+", "_", safe).strip("_")  # to clean leftovers
    # Guaranteeing uniqueness
    count = seen[safe]
    seen[safe] += 1
    if count:
        safe = f"{safe}_{count}"               # to append _1, _2 …
    return safe


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

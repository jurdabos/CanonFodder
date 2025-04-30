#!/usr/bin/env python3
"""
Interactive data‑profiling helper for CanonFodder
-------------------------------------------------
Run it as a standalone *script* or in a Jupyter cell with
    %run dev_profile.py               # end‑to‑end EDA using existing parquet caches
    %run dev_profile.py country       # edit the user‑country timeline first

What’s new (2025‑04‑30)
=======================
* **User‑country timeline now stores free‑text names** (no ISO restriction).
* **Artist‑country lookup first hits the local *ArtistCountry* table or
  `PQ/ac.parquet`**, then falls back to MusicBrainz if needed, and rewrites the
  cache + parquet on the fly.
* Adds a fast, vectorised join that assigns a ``UserCountry`` column to every
  scrobble by interval matching against the timeline.
* Keeps all guard‑rails (no overlaps, sensible dates) and rewrites
  ``PQ/uc.parquet`` automatically.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import select

# ---------------------------------------------------------------------------
#  CanonFodder imports (after dotenv)
# ---------------------------------------------------------------------------
load_dotenv()
from DB import SessionLocal
from DB.models import (
    ArtistCountry,
    ArtistVariantsCanonized,
    UserCountry,
    Scrobble,
)
from helpers import io
import mbAPI

# ---------------------------------------------------------------------------
#  Constants & matplotlib backend
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("TkAgg")  # before pyplot!
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[0] if "__file__" in globals() else Path.cwd()
JSON_DIR = PROJECT_ROOT / "JSON"
PQ_DIR = PROJECT_ROOT / "PQ"
UC_PARQUET = PQ_DIR / "uc.parquet"
AC_PARQUET = PQ_DIR / "ac.parquet"
PALETTES_FILE = JSON_DIR / "palettes.json"
SEPARATOR = "{"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("dev_profile")
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
mbAPI.init()

# ---------------------------------------------------------------------------
#  Display prefs
# ---------------------------------------------------------------------------
with PALETTES_FILE.open("r", encoding="utf‑8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = io.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette(custom_colors))

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
pd.options.display.float_format = "{:.2f}".format


# =============================================================================
# Helper functions – timeline & validation
# =============================================================================

def _parse_date(txt: str) -> Optional[pd.Timestamp]:
    txt = txt.strip()
    if not txt:
        return None
    try:
        return pd.Timestamp(txt).normalize()
    except ValueError as exc:
        raise ValueError(f"❌  '{d}' is not a valid date (YYYY‑MM‑DD)") from err


def _interval_ok(start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    if start is None:
        raise ValueError("Start date is required")
    if end is not None and end < start:
        raise ValueError("End date must be after start date")


def _overlaps(df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp | None) -> bool:
    cond_left = df["end_date"].isna() | (df["end_date"] >= s)
    cond_right = e is None or (df["start_date"] <= e)
    return bool(df[cond_left & cond_right].shape[0])


# ---------------------------------------------------------------------------
#  Timeline editor (free‑text country names)
# ---------------------------------------------------------------------------

def edit_country_timeline() -> pd.DataFrame:
    uc = pd.read_parquet(UC_PARQUET) if UC_PARQUET.exists() else pd.DataFrame(
        columns=["country", "start_date", "end_date"], dtype="object")
    print("\nEnter your country timeline (free‑text names). Blank to finish.\n")
    if not uc.empty:
        print(uc.to_string(index=False))
    while True:
        name = input("Country name  (blank = done): ").strip()
        if not name:
            break
        s_in = input("   Start YYYY‑MM‑DD: ")
        e_in = input("   End YYYY‑MM‑DD  (blank = ongoing): ")
        try:
            s_ts, e_ts = _parse_date(s_in), _parse_date(e_in)
            _interval_ok(s_ts, e_ts)
            if _overlaps(uc, s_ts, e_ts):
                print("⚠ Overlaps existing interval – try again\n")
                continue
            uc.loc[len(uc)] = [name, s_ts, e_ts]
            print("✔ added", name, s_ts.date(), "→", e_ts.date() if e_ts else "open‑ended")
        except ValueError as e:
            print("❌", e)
    if uc.empty:
        print("No intervals – leaving uc.parquet untouched.")
        return uc
    PQ_DIR.mkdir(parents=True, exist_ok=True)
    uc.sort_values("start_date", inplace=True)
    uc.to_parquet(UC_PARQUET, compression="zstd", index=False)
    log.info("Saved timeline → %s", UC_PARQUET.relative_to(PROJECT_ROOT))
    return uc


# =============================================================================
# Artist‑country helpers (parquet → DB → MusicBrainz)
# =============================================================================

def _load_artist_country_cache() -> pd.DataFrame:
    """Return df with columns [artist_name, country] (ISO‑2)."""
    if AC_PARQUET.exists():
        return pd.read_parquet(AC_PARQUET)
    # fall back to DB
    with SessionLocal() as s:
        rows = s.scalars(select(ArtistCountry)).all()
    if not rows:
        return pd.DataFrame(columns=["artist_name", "country"])
    df = pd.DataFrame([{"artist_name": r.artist_name, "country": r.country} for r in rows])
    df.to_parquet(AC_PARQUET, compression="zstd", index=False)
    return df


def _update_artist_country_cache(new_map: dict[str, str | None]):
    if not new_map:
        return
    # update DB
    with SessionLocal() as s:
        to_add = []
        for artist, iso in new_map.items():
            if s.scalar(select(ArtistCountry).where(ArtistCountry.artist_name == artist)):
                continue
            to_add.append(ArtistCountry(artist_name=artist, country=iso))
        if to_add:
            s.bulk_save_objects(to_add)
            s.commit()
    # update parquet
    df = _load_artist_country_cache()
    for a, c in new_map.items():
        df.loc[df.artist_name == a, "country"] = c
        if not (df.artist_name == a).any():
            df.loc[len(df)] = [a, c]
    df.to_parquet(AC_PARQUET, compression="zstd", index=False)


def artist_countries(series: pd.Series) -> pd.Series:
    cache_df = _load_artist_country_cache()
    cached = dict(zip(cache_df.artist_name, cache_df.country))
    missing = [a for a in series.unique() if a not in cached]
    fetched: dict[str, str | None] = {}
    for a in missing:
        fetched[a] = mbAPI.fetch_country(a)
    _update_artist_country_cache(fetched)
    cached.update(fetched)
    return series.map(cached)


# =============================================================================
# User‑country assignment to scrobbles (interval join)
# =============================================================================

def assign_user_country(scrobbles: pd.DataFrame, timeline: pd.DataFrame) -> pd.Series:
    if timeline.empty:
        return pd.Series([None] * len(scrobbles), index=scrobbles.index)
    timeline = timeline.sort_values("start_date").reset_index(drop=True)
    # merge_asof trick: needs key on start_date; we’ll merge then mask
    temp = scrobbles[["Datetime"]].rename(columns={"Datetime": "ts"})
    merged = pd.merge_asof(temp.sort_values("ts"), timeline, left_on="ts", right_on="start_date", direction="backward")
    # now drop rows where ts >= end_date (if end_date not null)
    mask = merged["end_date"].isna() | (merged["ts"] < merged["end_date"])
    return merged.loc[mask, "country"].reindex(scrobbles.index)


# =============================================================================
# Main routine
# =============================================================================

def main() -> None:
    # ---- load scrobbles parquet ---------------------------------------------------
    scrobbles, _ = io.latest_parquet(return_df=True)
    if scrobbles is None or scrobbles.empty:
        sys.exit("No scrobbles parquet found.")
    scrobbles.columns = ["Artist", "Album", "Song", "Datetime"]
    scrobbles.dropna(subset=["Datetime"], inplace=True)
    scrobbles = scrobbles.drop_duplicates(["Artist", "Album", "Song", "Datetime"])

    # ---- canonicalise artist names -------------------------------------------------
    with SessionLocal() as s:
        canon_rows = (
            s.query(ArtistVariantsCanonized)
            .filter(ArtistVariantsCanonized.to_link.is_(True))
            .all()
        )
    canon_map: dict[str, str] = {}
    for r in canon_rows:
        for v in re.split(rf"[{re.escape(SEPARATOR)}]", r.artist_variants):
            v = v.strip()
            if v and v != r.canonical_name:
                canon_map[v] = r.canonical_name
    if canon_map:
        scrobbles["Artist"].replace(canon_map, inplace=True)

    # ---- user‑country timeline -----------------------------------------------------
    if UC_PARQUET.exists():
        use = input("Use existing user‑country timeline? [Y]es / [E]dit / [N]ew: ").strip().lower() or "y"
        if use.startswith("e"):
            uc_df = edit_country_timeline()
        elif use.startswith("n"):
            UC_PARQUET.unlink(missing_ok=True)
            uc_df = edit_country_timeline()
        else:
            uc_df = pd.read_parquet(UC_PARQUET)
    else:
        uc_df = edit_country_timeline()

    # ---- enrich scrobbles ----------------------------------------------------------
    scrobbles["ArtistCountry"] = artist_countries(scrobbles["Artist"])
    scrobbles["UserCountry"] = assign_user_country(scrobbles, uc_df)
    scrobbles["Year"] = scrobbles["Datetime"].dt.year

    # ---- simple plots --------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    top10 = scrobbles["Artist"].value_counts().head(10)
    sns.barplot(x=top10.values, y=top10.index, palette="husl", legend=False)
    plt.title("Top 10 Artists by Scrobbles")
    plt.tight_layout()
    plt.show()

    if scrobbles["ArtistCountry"].notna().any():
        plt.figure(figsize=(12, 6))
        top_cty = scrobbles["ArtistCountry"].value_counts().head(15)
        sns.barplot(x=top_cty.index, y=top_cty.values, palette="Spectral")
        plt.title("Top 15 Artist Countries")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    if scrobbles["UserCountry"].notna().any():
        plt.figure(figsize=(12, 6))
        sns.countplot(data=scrobbles, x="Year", hue="UserCountry")
        plt.title("Plays per Year by User Country")
        plt.tight_layout()
        plt.show()


# =============================================================================
if __name__ == "__main__":
    main()

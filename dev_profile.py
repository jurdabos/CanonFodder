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
# %%
from __future__ import annotations

from dotenv import load_dotenv
import argparse
load_dotenv()
from DB import SessionLocal
from DB.models import (
    ArtistCountry,
    ArtistVariantsCanonized,
    UserCountry,
    Scrobble,
)
from datetime import date, datetime

from helpers import io
import json
import logging

log = logging.getLogger(__name__)

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mbAPI

mbAPI.init()
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
import os

os.environ["MPLBACKEND"] = "TkAgg"
import pandas as pd
from pathlib import Path
import re
import seaborn as sns
from sqlalchemy import select, text
import sys
from typing import Optional, Tuple

# %%
# Constants & basic setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("dev_profile")
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
mbAPI.init()
matplotlib.use("TkAgg")
os.environ["MPLBACKEND"] = "TkAgg"
log.addFilter(lambda rec: not rec.name.startswith("musicbrainzngs.mbxml"))
PROJECT_ROOT = Path(__file__).resolve().parents[0] if "__file__" in globals() else Path.cwd()
JSON_DIR = PROJECT_ROOT / "JSON"
PQ_DIR = PROJECT_ROOT / "PQ"
UC_PARQUET = PQ_DIR / "uc.parquet"
AC_PARQUET = PQ_DIR / "ac.parquet"
PALETTES_FILE = JSON_DIR / "palettes.json"
SEPARATOR = "{"
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.options.display.float_format = "{: .2f}".format
with PALETTES_FILE.open("r", encoding="utf‑8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = io.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette(custom_colors))


# %%
# -------------------------------------------------------------------------------------
#   Helper functions – timeline parsing / validation
# -------------------------------------------------------------------------------------

def _parse_date(d: str) -> Optional[pd.Timestamp]:
    d = d.strip()
    if not d:
        return None
    try:
        return pd.Timestamp(d).normalize()
    except ValueError as err:
        raise ValueError(f"❌  '{d}' is not a valid date (YYYY‑MM‑DD)") from err


def _interval_ok(start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    if start is None:
        raise ValueError("Start date is required")
    if end is not None and end < start:
        raise ValueError("End date must be after start date")


def _overlaps(df: pd.DataFrame, sta: pd.Timestamp, e: pd.Timestamp | None) -> bool:
    cond_left = df["end_date"].isna() | (df["end_date"] >= sta)
    cond_right = e is None or (df["start_date"] <= e)
    return bool(df[cond_left & cond_right].shape[0])


# %%
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


# %%
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


# %%
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


# %%
# -------------------------------------------------------------------------------------
#   STEP 1 – load scrobbles parquet & deduplicate
# -------------------------------------------------------------------------------------
print("=" * 90)
print("Welcome to the CanonFodder data profiling workflow!")
print("We'll load your scrobble data, apply any previously saved artist name unifications,")
print("then explore on forward.")
print("=" * 90, "\n")

data, latest_filename = io.latest_parquet(return_df=True)
if data is None or data.empty:
    sys.exit("🚫  No scrobble data found – aborting EDA.")

data.columns = ["Artist", "Album", "Song", "Datetime"]
data.dropna(subset=["Datetime"], inplace=True)

before_count = len(data)
data = data.drop_duplicates(["Artist", "Album", "Song", "Datetime"])
log.info("Deduplicated %d rows → %d remain", before_count - len(data), len(data))

# %%
# -------------------------------------------------------------------------------------
#   STEP 2 – apply artist canonicalisation from DB
# -------------------------------------------------------------------------------------
log.info("Fetching already canonised artist-name variants…")
with SessionLocal() as sess:
    canon_rows = (
        sess.query(ArtistVariantsCanonized)
        .filter(ArtistVariantsCanonized.to_link.is_(True))
        .all()
    )
SEPARATOR = "{"


def _split_variants(raw: str) -> list[str]:
    """
    Split the artist_variants field into its individual names.
    • Primary separator is “{”   (the new, unambiguous choice)
    • Strips whitespace and ignores empty items.
    """
    return [v.strip()
            for v in re.split(rf"[{re.escape(SEPARATOR)}]", raw)
            if v.strip()]


variant_to_canon: dict[str, str] = {}
for row in canon_rows:
    # `artist_variants` keeps the variants in one string, {-separated (a{b{c).
    variants: list[str] = [v.strip() for v in row.artist_variants.split("{") if v.strip()]
    for variant in _split_variants(row.artist_variants):
        if variant and variant != row.canonical_name:
            variant_to_canon[variant] = row.canonical_name
if variant_to_canon:
    data["Artist"] = data["Artist"].replace(variant_to_canon)

# %%
# -------------------------------------------------------------------------------------
#   STEP 3 – basic EDA (top artists)
# -------------------------------------------------------------------------------------
artist_counts = data["Artist"].value_counts()
top_artists = artist_counts.head(10)
print(top_artists)
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors_10 = io.register_custom_palette("colorpalette_10", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors_10))
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_artists.values,
    y=top_artists.index,
    palette=custom_colors_10[: len(top_artists)],
    hue=top_artists.index,
    legend=False,
)
plt.title("Top 10 Artists by Scrobbles", fontsize=16)
plt.xlabel("Scrobbles", fontsize=14)
plt.ylabel("Artist", fontsize=14)
plt.tight_layout()
plt.show()
print("Birds-eye stats for artist counts")
print(artist_counts.describe())
# Converting artist_counts to DataFrame
artist_counts_df = artist_counts.reset_index()
artist_counts_df.columns = ["Artist", "Count"]
count_threshold = 3
print(f"And what if we exclude artists with a play count below {count_threshold}?")
# Shrinking of the data set can be easily undone here by setting count_threshold to 1 in future flows
fltrd_artcount = artist_counts_df[artist_counts_df["Count"] >= count_threshold]
print(fltrd_artcount.describe())

# %%
# -------------------------------------------------------------------------------------
#   STEP 4 – user‑country timeline (fast‑lane vs CLI)
# -------------------------------------------------------------------------------------
if UC_PARQUET.exists():
    print(f"Fast‑lane timeline for historical user country locations found at {UC_PARQUET.relative_to(PROJECT_ROOT)}.")
    choice = input("[U]se as‑is  |  [E]dit   |  [N]ew (overwrite)  ?  [U/e/n]: ").strip().lower() or "u"
    if choice.startswith("e"):
        uc_df = edit_country_timeline()
    elif choice.startswith("n"):
        UC_PARQUET.unlink(missing_ok=True)
        uc_df = edit_country_timeline()
    else:
        uc_df = pd.read_parquet(UC_PARQUET)
else:
    print("No user‑country timeline found – let's create one now.")
    uc_df = edit_country_timeline()


# %%
# -------------------------------------------------------------------------------------
#   STEP 5 – enrich with country via MusicBrainz & chart
# -------------------------------------------------------------------------------------
def _country_for_series(series: pd.Series) -> pd.Series:
    """
    Vectorised ISO-country lookup with on-disk cache.
    """
    iso = {artist: mbAPI.fetch_country(artist) for artist in series.unique()}
    return series.map(iso)


data["Country"] = _country_for_series(data["Artist"])
country_count = data.Country.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={"Country": "count"})
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=country_count.index,
    y="count",
    data=country_count,
    palette="Spectral",
    hue=country_count.index,
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=10,
        color="black",
    )
ax.set_xticks(range(len(country_count)))
ax.set_xticklabels(country_count.index, rotation=45, ha="right", fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_title("Top 15 Countries", fontsize=16)
ax.set_xlabel("Country", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()
plt.show()
plt.close()

# %%
# -------------------------------------------------------------------------------------
#   STEP 6 – user-country analytics
# -------------------------------------------------------------------------------------

# %%
# -------------------------------------------------------------------------------------
#   STEP 7 – temporal enrichment
# -------------------------------------------------------------------------------------
data["Year"] = data["Datetime"].dt.year
data["Month"] = data["Datetime"].dt.month
data["Day"] = data["Datetime"].dt.day


# %%
# --- entry point ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    pee = argparse.ArgumentParser(description="CanonFodder dev profiling helper")
    sub = pee.add_subparsers(dest="cmd", help="Sub‑commands")
    sub.add_parser("country", help="Edit user‑country timeline interactively")
    return pee.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

# %%
# DB-based user country lookup for integrated UC-analytics workflow for the future
uc_df = pd.read_parquet(Path(__file__).resolve().parent / "PQ/uc.parquet")
with SessionLocal() as s:
    current_country = (
        s.scalars(select(UserCountry).where(UserCountry.end_date.is_(None))).first()
    )
    if current_country:
        tracks = s.scalars(
            select(Scrobble).where(
                Scrobble.play_time >= current_country.start_date,
                (current_country.end_date.is_(None)) | (Scrobble.play_time < current_country.end_date),
            )
        ).all()

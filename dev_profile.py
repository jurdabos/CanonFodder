"""
Interactive data‑profiling helper for CanonFodder
-------------------------------------------------
Run it as a Jupyter-style cells
-------------------------------------------------
Per 2025‑04‑30
-------------------------------------------------
* **User‑country timeline stores free‑text names** (no ISO restriction).
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
plt.ion()
import mbAPI

mbAPI.init()
import musicbrainzngs
musicbrainzngs.logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
import os

os.environ["MPLBACKEND"] = "TkAgg"
import pandas as pd
from pathlib import Path
import re
import seaborn as sns
from sqlalchemy import select, text, func
import sys
from typing import Optional, Tuple

# %%
# Constants & basic setup
logging.basicConfig(
    level=logging.WARNING,
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
AC_COLS = ["artist_name", "country", "mbid", "disambiguation_comment"]
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
#  Timeline editor
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
def _df_from_db() -> pd.DataFrame:
    """Pull the entire artistcountry table into a DataFrame."""
    with SessionLocal() as sessio:
        rows = sessio.scalars(select(ArtistCountry)).all()
    if not rows:
        return pd.DataFrame(columns=AC_COLS)
    return pd.DataFrame(
        [
            {
                "artist_name": r.artist_name,
                "country": r.country,
                "mbid": r.mbid,
                "disambiguation_comment": r.disambiguation_comment,
            }
            for r in rows
        ],
        columns=AC_COLS
    )


def _load_ac_cache() -> pd.DataFrame:
    """
    Return the artist-country cache as DataFrame.
    • If the Parquet is newer than the last DB update → read Parquet
    • otherwise
        – pull from DB
        – overwrite Parquet
    """
    if AC_PARQUET.exists():
        pq_mtime = AC_PARQUET.stat().st_mtime
        with SessionLocal() as session:
            db_mtime = session.scalar(
                select(func.max(ArtistCountry.id))  # monotonic surrogate pk
            ) or 0
        if pq_mtime > db_mtime:
            return pd.read_parquet(AC_PARQUET)
    dataf = _df_from_db()
    dataf.to_parquet(AC_PARQUET, index=False, compression="zstd")
    return dataf


def _upsert_artist_country(new_rows: list[dict]) -> None:
    """Insert *or* enrich existing rows in a backend-agnostic way."""
    if not new_rows:
        return
    with SessionLocal() as se:
        for r in new_rows:
            obj = (
                se.query(ArtistCountry)
                .filter_by(artist_name=r["artist_name"])
                .one_or_none()
            )
            if obj:
                if not obj.mbid and r["mbid"]:
                    obj.mbid = r["mbid"]
                if not obj.disambiguation_comment and r["disambiguation_comment"]:
                    obj.disambiguation_comment = r["disambiguation_comment"]
                if not obj.country and r["country"]:
                    obj.country = r["country"]
            else:
                se.add(ArtistCountry(**r))
        se.commit()


# Syncing Parquet with DB
df = _df_from_db()
df.to_parquet(AC_PARQUET, index=False, compression="zstd")


def artist_countries(series: pd.Series) -> pd.Series:
    """
    Vectorised ISO-country lookup with on-disk/DB cache;
    back-fills MBID + disambiguation when available.
    """
    cache_df = _load_ac_cache()
    cached = cache_df.set_index("artist_name").to_dict("index")
    missing = [a for a in series.unique() if a not in cached]
    new_rows = []
    for artist in missing:
        mb_res = mbAPI.search_artist(artist, limit=1)
        mb_row = mb_res[0] if mb_res else {}
        new_rows.append(
            dict(
                artist_name=artist,
                country=mb_row.get("country"),
                mbid=mb_row.get("id"),
                disambiguation_comment=mb_row.get("disambiguation"),
            )
        )
    _upsert_artist_country(new_rows)
    # Refreshing cache after insert
    cache_df = _load_ac_cache()
    cached = cache_df.set_index("artist_name").country.to_dict()
    return series.map(cached)


def top_n_artists_by_country(adatkeret, country, n=24):
    """
    Return the N most-played artists for one country,
    sorted by descending play-count.
    """
    (adatkeret
     .loc[adatkeret["Country"] == country, "Artist"]
     .value_counts()
     .head(n)
     )


# %%
# =============================================================================
# User‑country assignment to scrobbles (interval join)
# =============================================================================
def load_country_timeline(path: Path) -> pd.DataFrame:
    tl = (
        pd.read_parquet(path)
        .rename(columns={"country_name": "UserCountry"})
    )
    tl["start_date"] = pd.to_datetime(tl["start_date"]).dt.normalize()
    tl["end_date"] = pd.to_datetime(tl["end_date"]).dt.normalize()
    tl.sort_values("start_date", inplace=True)
    return tl


def assign_user_country(datafr, timeline):
    temp = datafr.copy()
    temp["day"] = pd.to_datetime(temp["Datetime"], unit="s").dt.normalize()
    out = pd.merge_asof(
        temp.sort_values("day"),
        timeline[["start_date", "UserCountry"]],
        left_on="day",
        right_on="start_date",
        direction="backward",
    )
    return out["UserCountry"]


# %%
# -------------------------------------------------------------------------------------
#   STEP 1 – load scrobbles parquet & deduplicate
# -------------------------------------------------------------------------------------
print("=" * 90)
print("Welcome to the CanonFodder data profiling workflow!")
print("We'll load your scrobble data, apply any previously saved artist name unifications,")
print("then explore on forward.")
print("=" * 90)
data, latest_filename = io.latest_parquet(return_df=True)
if data is None or data.empty:
    sys.exit("🚫  No scrobble data found – aborting EDA.")
data.columns = ["Artist", "Album", "Song", "Datetime"]
data.dropna(subset=["Datetime"], inplace=True)
data = data.drop_duplicates(["Artist", "Album", "Song", "Datetime"])
log.info("After dedup, %d rows remain.", len(data))

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
    variants: list[str] = [v.strip() for v in row.artist_variants_text.split("{") if v.strip()]
    for variant in _split_variants(row.artist_variants_text):
        if variant and variant != row.canonical_name:
            variant_to_canon[variant] = row.canonical_name
if variant_to_canon:
    data["Artist"] = data["Artist"].replace(variant_to_canon)

# %%
# -------------------------------------------------------------------------------------
#   STEP 3 – EDA
# -------------------------------------------------------------------------------------
artist_counts = data["Artist"].value_counts()
top_artists = artist_counts.head(10)
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
print("Birds-eye stats for artist counts")
print(artist_counts.describe())


# %%
# -------------------------------------------------------------------------------------
#   STEP 4 – user‑country timeline (fast‑lane vs CLI)
# -------------------------------------------------------------------------------------
if UC_PARQUET.exists():
    use = input("Use existing user‑country timeline? [Y]es / [E]dit / [N]ew: ").strip().lower() or "y"
    if use.startswith("e"):
        uc_df = edit_country_timeline()
    elif use.startswith("n"):
        UC_PARQUET.unlink(missing_ok=True)
        uc_df = edit_country_timeline()
    else:
        uc_df = pd.read_parquet(UC_PARQUET)

# %%
# -------------------------------------------------------------------------------------
#   STEP 5 – enrich with country via MusicBrainz & chart
# -------------------------------------------------------------------------------------
ac_cache = (
    pd.read_parquet(AC_PARQUET)
    .set_index("artist_name")["country"]  # Series {artist → country}
    .to_dict()
)


def _country_for_series(series: pd.Series,
                        cache: dict[str, str | None]) -> pd.Series:
    """
    Vectorised ISO-country lookup:
      1. use cached country if present
      2. otherwise query MusicBrainz once and extend the cache
    """
    missing = [a for a in series.unique() if a not in cache]
    if missing:
        mb_results = {
            artist: mbAPI.fetch_country(artist)
            for artist in missing
        }
        cache.update(mb_results)
        if mb_results:
            (pd.DataFrame
             .from_dict(mb_results, orient="index", columns=["country"])
             .assign(artist_name=lambda d: d.index)
             .to_parquet(AC_PARQUET, compression="zstd",
                         append=True, index=False))
    return series.map(cache)


data["ArtistCountry"] = _country_for_series(data["Artist"], ac_cache)
country_count = data.ArtistCountry.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={"ArtistCountry": "count"})
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
ax.set_xlabel("ArtistCountry", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()

# %%
# -------------------------------------------------------------------------------------
#   STEP 6 – user-country analytics
# -------------------------------------------------------------------------------------
uc_df = load_country_timeline(UC_PARQUET)
data["UserCountry"] = assign_user_country(data, uc_df)
user_country_count = data.UserCountry.value_counts().sort_values(ascending=False).to_frame()[:10]
user_country_count = user_country_count.rename(columns={"UserCountry": "count"})
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=user_country_count.index,
    y="count",
    data=user_country_count,
    palette="Spectral",
    hue=user_country_count.index,
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
ax.set_xticks(range(len(user_country_count)))
ax.set_xticklabels(user_country_count.index, rotation=45, ha="right", fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_title("User countries per scrobble count", fontsize=16)
ax.set_xlabel("UserCountry", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# -------------------------------------------------------------------------------------
#   STEP 7 – temporal enrichment
# -------------------------------------------------------------------------------------
adat = data.copy()
s = pd.to_datetime(adat["Datetime"], unit="s", utc=True)
s = s.dt.tz_convert(None)
data["Datetime"] = s
data.describe()
data["Year"] = data["Datetime"].dt.year
data["Month"] = data["Datetime"].dt.month
data["Day"] = data["Datetime"].dt.day
# Total scrobbles per year
data.groupby("Year")["Song"].count().plot(kind="bar", figsize=(10, 4), rot=0,
                                          title="Total scrobbles per year")
# Average per month
data.groupby("Month")["Song"].count().div(data["Year"].nunique()).plot(kind="bar", title="Average monthly scrobbles")

# Calendar heatmap for single year
y = 2024
pivot = (data[data["Year"] == y]
         .assign(DoW=lambda x: x["Datetime"].dt.dayofweek,
                 Week=lambda x: x["Datetime"].dt.isocalendar().week)
         .pivot_table(index="Week", columns="DoW",
                      values="Song", aggfunc="count", fill_value=0))
sns.heatmap(pivot, cmap="viridis")
plt.title(f"Scrobbles during {y}")


# Top artists per year
def yearly_top(adatkupac, n=2):
    return (adatkupac.groupby(["Year", "Artist"])["Song"].count()
            .groupby(level=0, group_keys=False)
            .nlargest(n)
            .reset_index(name="Plays"))


yearly = yearly_top(data)


# Day-of-week pattern
data["Datetime"].dt.day_name().value_counts().reindex([
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"]).plot(kind="bar", rot=30, title="Scrobbles by day of week")


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

"""
Interactive data‑profiling helper for CanonFodder
-------------------------------------------------
Run it as a Jupyter-style cells
-------------------------------------------------
Per 2025‑04‑30
-------------------------------------------------
* **User‑country timeline stores ISO-2 country codes.
* **Artist‑country lookup first hits the local *ArtistInfo* table or
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
    ArtistInfo,
    ArtistVariantsCanonized,
)
from datetime import date, datetime
from helpers import cli
from helpers import io
from helpers import stats
import json
import logging
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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import seaborn as sns
from sqlalchemy import select
import sys

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
# =============================================================================
# Artist‑country helpers (parquet → DB → MusicBrainz)
# =============================================================================
def _append_to_parquet(path: Path, adatk: pd.DataFrame) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        adatk = (pd.concat([old, adatk])
                 .drop_duplicates(subset="artist_name", keep="last"))
    adatk.to_parquet(path, compression="zstd", index=False)


def _country_for_series(series: pd.Series, cache: dict[str, str | None]) -> pd.Series:
    missing = [a for a in series.unique() if a not in cache]
    if missing:
        mb_results = {a: mbAPI.fetch_country(a) for a in missing}
        cache.update(mb_results)
        if mb_results:
            new_df = (pd.DataFrame
                      .from_dict(mb_results, orient="index", columns=["country"])
                      .assign(artist_name=lambda d: d.index))
            _append_to_parquet(AC_PARQUET, new_df)
    return series.map(cache)


def _df_from_db() -> pd.DataFrame:
    """Pull the entire ArtistInfo table into a DataFrame."""
    with SessionLocal() as sessio:
        rows = sessio.scalars(select(ArtistInfo)).all()
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


# Syncing Parquet with DB
df = _df_from_db()
df.to_parquet(AC_PARQUET, index=False, compression="zstd")
# io.dump_parquet()
# THE ABOVE LINE HAS BEEN COMMENTED OUT BECAUSE FIRST FETCH IS GOOD, SUBSEQUENT APPENDS ARE INCORRECT. FIX LATER!

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
data.columns = ["Artist", "Album", "Datetime", "Song", "MBID"]
data.dropna(subset=["Datetime"], inplace=True)
data = data.drop_duplicates(["Artist", "Album", "Song", "Datetime"])
log.info("After dedup, %d rows remain.", len(data))

# %%
# -------------------------------------------------------------------------------------
#   STEP 2 – EDA - Why canonization of artist name variants matters?
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
plt.show()

# %%
# Canonization rationale snapshot
top3 = artist_counts.head(3).index
bohren_re = r'Bohren'
autechre_re = r'Autechre'
mask = (
        data['Artist'].isin(top3)
        | data['Artist'].str.contains(bohren_re, case=False, na=False)
        | data['Artist'].str.contains(autechre_re, case=False, na=False)
)
undesired = ['lustmord', 'sikora', 'sophie', 'orbital']
bad_pat = '|'.join(undesired)
mask &= ~data['Artist'].str.contains(bad_pat, case=False, na=False)
filtdata = data.loc[mask].copy()
bohren_canon = 'Bohren & der Club of Gore'
bohren_variants = {
    'Bohren & der Club of Gore': bohren_canon,
    'Bohren und der Club of Gore': bohren_canon,
}
filtdata['Artist_canon'] = filtdata['Artist'].replace(bohren_variants)
highlight = '#0D3C45'
default = '#ABB7C4'
highlight_artists = {
    'Autechre',
    'Bohren & der Club of Gore',
    'Bohren und der Club of Gore',
    'Bohren (canonized)',
}


def colour_for(artist: str) -> str:
    return highlight if artist in highlight_artists else default


counts1 = (filtdata
           .groupby('Artist')
           .size()
           .reset_index(name='Scrobbles')
           .sort_values('Scrobbles', ascending=False))
text_colours = ['white' if colour_for(a) == highlight else 'black'
                for a in counts1['Artist']]
trace_left = go.Bar(
    x=counts1['Scrobbles'],
    y=counts1['Artist'],
    orientation='h',
    marker_color=[colour_for(a) for a in counts1['Artist']],
    text=counts1['Scrobbles'],
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color=text_colours),
    hovertemplate='%{y}: %{x}<extra></extra>'
)
wanted = [bohren_canon, 'Autechre', 'Radiohead', 'Secret Chiefs 3']
counts2 = (filtdata[filtdata['Artist_canon'].isin(wanted)]
           .groupby('Artist_canon')
           .size()
           .reset_index(name='Scrobbles')
           .replace({bohren_canon: 'Bohren (canonized)'}))
bohren_parts = (filtdata
                [filtdata['Artist'].isin(['Bohren & der Club of Gore',
                                          'Bohren und der Club of Gore'])]
                .groupby('Artist')
                .size())
autechre_cnt = counts2.loc[counts2['Artist_canon'] == 'Autechre', 'Scrobbles'].iat[0]
trace_bohren_amp = go.Bar(
    x=[bohren_parts['Bohren & der Club of Gore']],
    y=['Bohren (canonized)'],
    orientation='h',
    marker_color=highlight,
    text=[bohren_parts['Bohren & der Club of Gore']],
    textposition='inside',
    insidetextanchor='middle',
    hovertemplate='Bohren & der Club of Gore: %{x}<extra></extra>'
)
trace_bohren_und = go.Bar(
    x=[bohren_parts['Bohren und der Club of Gore']],
    y=['Bohren (canonized)'],
    orientation='h',
    marker_color=highlight,
    text=[bohren_parts['Bohren und der Club of Gore']],
    textposition='inside',
    insidetextanchor='middle',
    hovertemplate='Bohren und der Club of Gore: %{x}<extra></extra>'
)


def single_bar(artist):
    cnt = counts2.loc[counts2['Artist_canon'] == artist, 'Scrobbles'].iat[0]
    return go.Bar(
        x=[cnt],
        y=[artist],
        orientation='h',
        marker_color=colour_for(artist),
        text=[cnt],
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate=f'{artist}: %{{x}}<extra></extra>'
    )


trace_autechre = single_bar('Autechre')
trace_radiohead = single_bar('Radiohead')
trace_sc3 = single_bar('Secret Chiefs 3')
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Top artists before canonisation of Bohren variants",
                    "Top artists after canonisation of Bohren variants"),
    shared_xaxes=True,
    vertical_spacing=0.12
)
fig.add_trace(trace_left, row=1, col=1)
fig.add_traces([trace_bohren_amp,
                trace_bohren_und,
                trace_autechre,
                trace_radiohead,
                trace_sc3], rows=[2] * 5, cols=[1] * 5)
fig.update_xaxes(showticklabels=False, ticks="", row=1, col=1)
fig.update_xaxes(showticklabels=False, ticks="", row=2, col=1)
fig.update_yaxes(autorange='reversed', row=1, col=1)
fig.update_yaxes(autorange='reversed', row=2, col=1)
fig.update_layout(
    barmode='stack',
    showlegend=False,
    height=700,
    width=700,
    margin=dict(l=90, r=40, t=80, b=40)
)
bohren_total = bohren_parts.sum()
fig.add_annotation(
    x=bohren_total, y='Bohren (canonized)',
    xref='x2', yref='y2',
    text=str(bohren_total),
    showarrow=False,
    font=dict(color='black', size=12),
    xshift=19
)
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)
fig.show()
fig.write_image("scrobbles.png", width=1000, height=800, scale=2)

# %%
# -------------------------------------------------------------------------------------
#   STEP 3 – apply artist canonicalisation from DB
# -------------------------------------------------------------------------------------
log.info("Fetching already canonised artist-name variants…")
with SessionLocal() as sess:
    canon_rows = (
        sess.query(ArtistVariantsCanonized)
        .filter(ArtistVariantsCanonized.to_link.is_(True))
        .all()
    )


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
#   STEP 4 – user‑country timeline (fast‑lane vs CLI)
# -------------------------------------------------------------------------------------
if UC_PARQUET.exists():
    use = cli.choose_timeline()
    if use == "e":
        uc_df = cli.edit_country_timeline()
    elif use == "n":
        UC_PARQUET.unlink(missing_ok=True)
        uc_df = cli.edit_country_timeline()
    else:  # "y"
        uc_df = pd.read_parquet(UC_PARQUET)
else:
    uc_df = cli.edit_country_timeline()

# %%
# -------------------------------------------------------------------------------------
#   STEP 5 – enrich with country via MusicBrainz & chart
# -------------------------------------------------------------------------------------
ac_cache = (
    pd.read_parquet(AC_PARQUET)
    .set_index("artist_name")["country"]
    .to_dict()
)
data["ArtistInfo"] = _country_for_series(data["Artist"], ac_cache)
country_count = data.ArtistInfo.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={"ArtistInfo": "count"})
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
ax.set_xlabel("ArtistInfo", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()

# %%
# -------------------------------------------------------------------------------------
#   STEP 6 – user-country analytics
# -------------------------------------------------------------------------------------
uc_df = io.load_country_timeline(UC_PARQUET)
data["UserCountry"] = stats.assign_user_country(data, uc_df)
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

"""
corefunc.dataprofiler
=====================
Utility functions to incorporate profiling tasks into main application.
All public functions:
* quick_viz
* run_profiling
"""
from __future__ import annotations
from collections import namedtuple
import logging
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

LOGGER = logging.getLogger(__name__)


ProfileResult = namedtuple(
    "ProfileResult",
    "df artist_counts sim names name_to_idx"
)
"""Container bundling results returned by :func:`run_profiling`."""


# corefunc/dataprofiler.py
def quick_viz(profile: ProfileResult, n: int = 10) -> None:
    """
    Displays a bar plot of the top n artists.
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    top = profile.artist_counts.head(n)
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top.values, y=top.index, palette="rocket_r")
    plt.title(f"Top {n} Artists by Scrobbles")
    plt.xlabel("Scrobbles")
    plt.tight_layout()
    plt.show()


def run_profiling(data: pd.DataFrame) -> ProfileResult:
    """
    Cleans raw last.fm scrobble data and computes name-level similarity.
    Performs duplicate-dropping, coercion to UTC.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing last.fm scrobbles

    Returns
    -------
    ProfileResult
        A ProfileResult object containing the mutated dataframe,
        the artist counts, the similarity matrix, the names,
        and the names-to-index mapping.
    """
    before = len(data)
    data = data.copy()
    if "uts" in data.columns:
        uts_dt = pd.to_datetime(
            data["uts"].astype("int64"),
            unit="s",
            utc=True,
            errors="coerce",
        )
        data = data.assign(uts=uts_dt)
    data.columns = ["Artist", "Album", "Song", "Datetime"]
    data = data.drop_duplicates(subset=["Artist", "Album", "Song", "Datetime"])
    LOGGER.info("Dropped %d duplicate rows", before - len(data))
    data = data.dropna(subset=["Datetime"])
    # -- basic counts --------------------------------------------------
    artist_counts = data["Artist"].value_counts()
    # -- similarity matrix ---------------------------
    names = list(dict.fromkeys(artist_counts.index.str.casefold().str.strip()))
    name_to_idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    sim = np.zeros((n, n), dtype=float)
    for i, a in enumerate(names):
        for j in range(i, n):
            sim[i, j] = sim[j, i] = fuzz.token_sort_ratio(a, names[j]) / 100
    return ProfileResult(data, artist_counts, sim, names, name_to_idx)


def generate_markdown_report(profile: ProfileResult) -> str:
    """
    Generates a Markdown report from a ProfileResult.

    Parameters
    ----------
    profile : ProfileResult
        The profile result to generate a report for

    Returns
    -------
    str
        The Markdown report
    """
    # Get the top 20 artists
    top_artists = profile.artist_counts.head(20)

    # Generate Markdown
    md = f"""# Data Profiling Report

## Overview
- Total scrobbles: {len(profile.df)}
- Unique artists: {len(profile.artist_counts)}

## Top 20 Artists
| Artist | Scrobbles |
|--------|-----------|
"""

    for artist, count in top_artists.items():
        md += f"| {artist} | {count} |\n"

    # Add similarity information
    md += f"""
## Artist Name Similarity

The profiling identified {len(profile.names)} unique artist names.
A similarity matrix was computed using fuzzy string matching to identify potential duplicates.

### Similarity Statistics
- Highest similarity between different artists: {profile.sim.max():.2f}
- Average similarity between artists: {profile.sim.mean():.2f}
"""

    return md


def generate_html_report(profile: ProfileResult, output_file=None) -> str:
    """
    Generates an HTML report from a ProfileResult using Showdown.

    Parameters
    ----------
    profile : ProfileResult
        The profile result to generate a report for
    output_file : str or Path, optional
        If provided, the HTML will be written to this file

    Returns
    -------
    str
        The HTML report
    """
    try:
        from helpers.markdown import render_markdown

        # Generate Markdown report
        md_report = generate_markdown_report(profile)

        # Render to HTML
        html_report = render_markdown(md_report, output_file)

        return html_report
    except ImportError:
        # If Showdown is not available, return the Markdown report
        return generate_markdown_report(profile)

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
import os
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
    Args:
        data: the pd df containing last.fm scrobbles
    Returns:
        A ProfileResult object containing the mutated dataframe,
        the artist counts, the similarity matrix, the names,
        and the names-to-index mapping.
    Example:
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

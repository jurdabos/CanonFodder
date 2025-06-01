"""
corefunc.canonizer
==================

Utility functions that normalise artist-name variants and compute a
minimal ε for DBSCAN clustering, ensuring user-defined ‘must-merge’
sub-groups are honoured.

All public functions:

* apply_previous
* minimal_epsilon
"""

from __future__ import annotations
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized
import logging
log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import re
from sklearn.cluster import DBSCAN

SEPARATOR = "{"


def _split(raw: str) -> list[str]:
    """
    Splits a raw artist-variant string into individual names.
    The string is split on the canonical separator ``SEPARATOR`` or pipe
    characters and trimmed. Empty fragments are ignored.
    Args:
        raw: Concatenated variant string, e.g. "Prince{|}‪プリンス‬".
    Returns:
        A list of cleaned variants, with order preserved.
    Example:
        >>> _split("a {|} b|c")
        ['a', 'b', 'c']
    """
    return [v.strip() for v in re.split(rf"[{re.escape(SEPARATOR)}|]", raw) if v.strip()]


def apply_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies previously canonised variants to a df in-place.
    Args:
        df: the dataframe passed in to modify
    Returns:
        df with mutated df["Artist"]
    I/O contract:
        Expects a canonical ‘Artist’ column.
    """
    with SessionLocal() as s:
        rows = (s.query(ArtistVariantsCanonized)
                .filter(ArtistVariantsCanonized.to_link.is_(True))
                .all())
    mapping = {v: r.canonical_name
               for r in rows
               for v in _split(r.artist_variants_text)
               if v and v != r.canonical_name}
    if mapping:
        df["Artist"] = df["Artist"].replace(mapping)
        log.info("Applied on %d variants covering %d canonical names",
                 len(mapping), len(set(mapping.values())))
    return df


def minimal_epsilon(sim: np.ndarray,
                    name_to_idx: dict[str, int],
                    must_merge: list[list[str]],
                    step: float = 0.01) -> float | None:
    """
    Scans ε until every present subgroup in *must_merge* ends up in the same
    DBSCAN cluster. Sub-groups that are missing one or more members in the
    current data set are ignored (a warning is logged); if *all* sub-groups are
    missing we still raise.
    Args:
        sim: square similarity matrix [n × n] with values in [0, 1]
        name_to_idx: mapping from lower-cased artist name to row index
        must_merge: list of lists, each inner list is a mutually required group
        step: step size for the ε scan (default 0.01)
    Raises:
        ValueError: If none of the required sub-groups are present.
    """
    groups_ok = []
    groups_missing = []
    for g in must_merge:
        if all(name in name_to_idx for name in g):
            groups_ok.append(g)
        else:
            groups_missing.append(g)
    if not groups_ok:  # nothing to test
        raise ValueError(
            "none of the must_merge sub-groups exist in this data set; "
            f"missing samples: {groups_missing}"
        )
    if groups_missing:
        import warnings
        warnings.warn(f"Ignoring incomplete sub-groups: {groups_missing}")
    for eps in np.arange(step, 1.0 + 1e-9, step):
        labels = DBSCAN(eps=eps,
                        min_samples=2,
                        metric="precomputed").fit_predict(1 - sim)
        for group in groups_ok:
            idx = [name_to_idx[n] for n in group]
            lbls = {labels[i] for i in idx}
            if len(lbls) != 1 or -1 in lbls:
                break
        else:
            return round(float(eps), 4)
    return None

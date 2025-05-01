"""
Provides interactive command-line helpers for data cleaning and user prompts.
"""
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime, UTC
from DB import get_session as _get_session

from DB.models import ArtistVariantsCanonized, ArtistCountry
from DB import SessionLocal

import os
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
import questionary
from sqlalchemy import select, insert, update

SEPARATOR = "{"


def _apply_canonical(canonical: str,
                     variants: list[str],
                     data: pd.DataFrame,
                     artcounts: pd.DataFrame) -> None:
    """Replace every variant in *data* with *canonical* and refresh counts in-place."""
    data["Artist"] = data["Artist"].replace(dict.fromkeys(variants, canonical))
    artcounts.loc[artcounts["Artist"].isin(variants), "Artist"] = canonical


def _remember_artist_variant(signature: str, canonical: str, link_flag: bool, sess):
    """Universal UPSERT for artist_variants_canonized."""
    existing = sess.execute(
        select(ArtistVariantsCanonized)
        .where(ArtistVariantsCanonized.artist_variants == signature)
    ).scalar_one_or_none()

    if existing:
        sess.execute(
            update(ArtistVariantsCanonized)
            .where(ArtistVariantsCanonized.artist_variants == signature)
            .values(canonical_name=canonical, to_link=link_flag)
        )
    else:
        sess.execute(
            insert(ArtistVariantsCanonized)
            .values(artist_variants=signature, canonical_name=canonical, to_link=link_flag)
        )


def _split_variants(sig: str) -> list[str]:
    return [v.strip() for v in sig.split(SEPARATOR) if v.strip()]


def ask(question: str,
        default: str | None = None) -> str:
    """
    Prompts until the user enters text or accepts `default`
    Args:
        question: text shown to the user
        default: value returned when the user presses Enter
    Returns:
        user response as str
    """
    while True:
        prompt = f"{question.strip()} "
        if default is not None:
            prompt += f"[{default}] "
        print(prompt, end='', flush=True)
        reply = input().strip()
        if reply:
            return reply
        if default is not None:
            return default
        print("Please enter a value.")


def choose_lastfm_user() -> str:
    """
    Asks once for the Last.fm user.
    ─ behaviour ─
    • If LASTFM_USER is set in .env / environment → offer it as the default
    • Empty input while no default is known → keep asking
    """
    default = os.getenv("LASTFM_USER", "").strip() or None
    while True:
        tail = f" [{default}]" if default else ""
        reply = input(f"If you are querying data for last.fm user {tail}, press enter"
                      " Otherwise, type username here: › ").strip()
        if reply:
            return reply
        if default:
            return default
        print("Please type a user name (or set LASTFM_USER in your .env).")


def unify_artist_names_cli(
        data: pd.DataFrame,
        fltrd_artcount: pd.DataFrame,
        similar_artist_groups: list[list[str]],
):
    """
    Interactive resolver for artist-name duplicates, updating canonization and optional comments.
    """
    with _get_session() as sess:
        with sess.begin():
            for group in similar_artist_groups:
                group = list(group)
                if len(group) <= 1:
                    continue
                signature = SEPARATOR.join(sorted(group))
                prev_row = sess.execute(
                    select(ArtistVariantsCanonized)
                    .where(ArtistVariantsCanonized.artist_variants == signature)
                ).scalar_one_or_none()
                if prev_row:
                    if not prev_row.to_link:
                        print(f"\nUser previously SKIPPED {group}.")
                        continue
                    print(f"\nUser previously unified {group} → '{prev_row.canonical_name}'. Applying again.")
                    _apply_canonical(prev_row.canonical_name, group, data, fltrd_artcount)
                    continue
                print("\n---")
                print("These artist names appear to be duplicates:")
                print("\n".join(f"  - {v}" for v in group))
                choice = questionary.select(
                    "Which name should become canonical?",
                    choices=group + ["Custom name…", "Skip this group"]
                ).ask()
                if not choice or choice.startswith("Skip"):
                    _remember_artist_variant(signature, "__SKIP__", link_flag=False, sess=sess)
                    continue
                canonical = (
                    questionary.text("Enter the custom canonical name:").ask().strip()
                    if choice.startswith("Custom") else choice
                )
                if not canonical:
                    _remember_artist_variant(signature, "__SKIP__", link_flag=False, sess=sess)
                    continue
                comment = questionary.text(
                    "Optional comment/disambiguation (hit ↵ to skip):"
                ).ask().strip() or None
                _apply_canonical(canonical, group, data, fltrd_artcount)
                _remember_artist_variant(signature, canonical, link_flag=True, sess=sess)
    refreshed = (
        data["Artist"]
        .value_counts()
        .rename_axis("Artist")
        .reset_index(name="Count")
    )
    return data, refreshed


def verify_commas(csv_path: str | Path) -> None:
    """
    Checks whether *commas inside values* survived the 3rd-party CSV export.
    The function inspects several representative strings in every affected
    column (Artist / Album / Song).  For each “with-comma” spelling it prints
    how many rows match it and how many rows match the “comma-stripped”
    version.
    Parameters
    ----------
    csv_path : str | pathlib.Path
        Path to the CSV file produced by the web export.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # ------------------------------------------------------------------
    # 1)  What should we look for?
    #     column name  canonical-with-comma              sans-comma variant
    # ------------------------------------------------------------------
    probes: list[tuple[str | int, str, str]] = [
        # 1st col (= col 0) should always be the artist
        (0, "Volcano, I'm Still Excited!!", "Volcano I'm Still Excited!!"),
        (0, "Emerson, Lake & Palmer", "Emerson Lake & Palmer"),
        # 2nd col (= 1) is album, 3rd (= 2) is track in that export
        (1, "Grey Tickles, Black Pressure", "Grey Tickles Black Pressure"),
        (1, "Ágy,  asztal, TV", "Ágy  asztal TV"),
        (2, "Video fiú, video lány", "Video fiú video lány"),
        (2, "Nyálas, nyers angyalok", "Nyálas nyers angyalok"),
        (2, "I Have the Moon, You Have the Internet",
         "I Have the Moon You Have the Internet"),
    ]
    banner = "\n── Checking whether CSV export kept the internal commas ──"
    print(banner)
    for col, with_comma, no_comma in probes:
        series = df.iloc[:, col] if isinstance(col, int) else (
            df[col] if col in df.columns else None
        )
        if series is None:
            print(f"[warn] column \"{col!r}\" is not found in CSV - skipped")
            continue
        kept = int(series.eq(with_comma).sum())
        lost = int(series.eq(no_comma).sum())
        print(f"\n» column {col:5}  "
              f"→ '{with_comma}' OR '{no_comma}?'\n"
              f"    rows *with* comma   : {kept}\n"
              f"    rows sans comma     : {lost}")
    print("\n──────────────────────────────────────────────────────────\n")


def yes_no(question: str, *, default: str = "n") -> bool:
    """
    Returns true when the user answers yes
    """
    return ask(question, default).lower().startswith("y")

"""
Provides interactive command-line helpers for data cleaning and user prompts.
"""
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, UTC
from DB.models import ArtistVariantsCanonized
import os
import pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
import questionary
from sqlalchemy import select


def _apply_canonical(canon, variants, data, counts):
    """
    Replaces every occurrence of each variant with `canon` in two dataframes
    """
    for old in variants:
        if old != canon:
            data.loc[data["Artist"] == old, "Artist"] = canon
            counts.loc[counts["Artist"] == old, "Artist"] = canon


def _remember(sig: str, name: str, sess):
    """
    Persists a user decision for an artist-variant group so it is reused
    """
    now = datetime.now(UTC)
    sess.execute(
        ArtistVariantsCanonized.insert().values(
            group_signature=sig,
            canonical_name=name,
            timestamp=now
        )
    )


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
        data,
        fltrd_artcount,
        similar_artist_groups,
):
    """
    Guides the user through resolving duplicate artist names
    Args:
        data: dataframe with an Artist column
        fltrd_artcount: dataframe with current artist counts
        similar_artist_groups: iterable of duplicate-candidate groups
    Returns:
        tuple (dataframe with updated names, dataframe with refreshed counts)
    """
    from DB import get_engine as _get_engine, get_session as _get_session
    with _get_session().begin() as sess:
        for group in similar_artist_groups:
            group = list(group)
            if len(group) <= 1:
                continue
            sig = "|".join(sorted(group))
            existing = sess.scalar(
                select(ArtistVariantsCanonized.canonical_name)
                .where(ArtistVariantsCanonized.group_signature == sig)
            )
            # -----------------------------------------------------------------
            if existing is not None:  # handled earlier
                if existing == "__SKIP__":
                    print(f"\nUser previously SKIPPED group {group}.")
                    continue
                print(f"\nUser previously unified {group} to "
                      f"'{existing}'. Applying again.")
                _apply_canonical(existing, group, data, fltrd_artcount)
                continue
            # ----------------------------- ask the user ----------------------
            print("\n---")
            print(f"These artist names appear to be duplicates:\n{group}")
            choice = questionary.select(
                "Which name would you like to keep for all occurrences?",
                choices=group + ["Custom", "Skip"]
            ).ask()
            if not choice or choice == "Skip":
                _remember(sig, "__SKIP__", sess)
                continue
            canonical = (questionary.text("Enter a custom canonical name:").ask()
                         if choice == "Custom" else choice)
            if not canonical:
                _remember(sig, "__SKIP__", sess)
                continue
            _apply_canonical(canonical, group, data, fltrd_artcount)
            _remember(sig, canonical, sess)
    # ---------- re-aggregate counts ------------------------------------------
    out = data["Artist"].value_counts().reset_index(names=["Artist", "Count"])
    return data, out


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

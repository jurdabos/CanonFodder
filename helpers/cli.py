"""
Provides interactive command-line helpers for data cleaning and user prompts.
"""

from __future__ import annotations
import hashlib
import logging
import os
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional
import click
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from .io import AVC_PQ, UC_PQ, read_parquet, append_to_parquet, dump_parquet  # noqa: E402

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
SEPARATOR = "{"


def _apply_canonical(canonical: str, variants: list[str], data: pd.DataFrame, artcounts: pd.DataFrame) -> None:
    """Replace every variant in *data* with *canonical* and refresh counts in-place."""
    data["Artist"] = data["Artist"].replace(dict.fromkeys(variants, canonical))
    artcounts.loc[artcounts["Artist"].isin(variants), "Artist"] = canonical


def _interval_ok(start: pd.Timestamp | None, end: pd.Timestamp | None) -> None:
    if start is None:
        raise ValueError("Start date is required")
    if end is not None and end < start:
        raise ValueError("End date must be after start date")


def _overlaps(keret: pd.DataFrame, sta: pd.Timestamp, e: pd.Timestamp | None) -> bool:
    cond_left = keret["end_date"].isna() | (keret["end_date"] >= sta)
    cond_right = e is None or (keret["start_date"] <= e)
    return bool(keret[cond_left & cond_right].shape[0])


def _parse_date(d: str) -> Optional[pd.Timestamp]:
    d = d.strip()
    if not d:
        return None
    try:
        return pd.Timestamp(d).normalize()
    except ValueError as err:
        raise ValueError(f"❌  '{d}' is not a valid date (YYYY‑MM‑DD)") from err


def _remember_artist_variant(signature: str, canonical: str, link_flag: bool, comment: str | None) -> None:
    """Upserts a variant decision into avc.parquet."""
    signature_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    row = pd.DataFrame(
        [
            {
                "artist_variants_hash": signature_hash,
                "artist_variants_text": signature,
                "canonical_name": canonical,
                "to_link": link_flag,
                "comment": comment or "",
                "stamp": datetime.now(UTC).isoformat(),
            }
        ]
    )
    append_to_parquet(row, AVC_PQ, dedup_cols=["artist_variants_hash"])


def _split_variants(sig: str) -> list[str]:
    return [v.strip() for v in sig.split(SEPARATOR) if v.strip()]


def ask(question: str, default: str | None = None) -> str:
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
        print(prompt, end="", flush=True)
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
        reply = input(
            f"If you are querying data for last.fm user {tail}, press enter Otherwise, type username here: › "
        ).strip()
        if reply:
            return reply
        if default:
            return default
        print("Please type a user name (or set LASTFM_USER in your .env).")


def choose_timeline(default: str = "Y") -> str:
    """
    Return 'y', 'e', or 'n'.

    • If running in a true TTY → prompt the user.
    • If stdin is not a TTY (PyCharm SciView, Jupyter, CI) → return *default*.
    """

    def _prompt() -> str:
        answ = input("Use existing user-country timeline? [Y]es/[E]dit/[N]ew: ").strip() or default
        return answ[0].lower()

    # PyCharm's console / notebooks: no interactive stdin
    if not sys.stdin.isatty() or os.getenv("PYCHARM_HOSTED"):
        print(f"(no TTY – assuming '{default}')")
        return default[0].lower()
    while True:
        try:
            ans = _prompt()
            if ans in {"y", "e", "n"}:
                return ans
            print("Please enter Y, E, or N.")
        except (EOFError, KeyboardInterrupt):
            print()  # new line
            sys.exit("aborted by user")


# ---------------------------------------------------------------------------
#  Timeline editor
# ---------------------------------------------------------------------------
def edit_country_timeline() -> pd.DataFrame:
    """Interactively edits the user-country timeline stored in uc.parquet."""
    uc = read_parquet(UC_PQ)
    if uc is None or uc.empty:
        uc = pd.DataFrame(columns=["country_code", "start_date", "end_date"], dtype="object")
    click.echo("\nEnter your country timeline. Blank to finish.\n")
    if not uc.empty:
        click.echo(uc.to_string(index=False))
    while True:
        name = click.prompt("Country code (blank = done)", default="", show_default=False)
        if not name:
            break
        s_in = click.prompt("   Start YYYY-MM-DD")
        e_in = click.prompt("   End YYYY-MM-DD (blank = ongoing)", default="", show_default=False)
        try:
            s_ts, e_ts = _parse_date(s_in), _parse_date(e_in)
            _interval_ok(s_ts, e_ts)
            if _overlaps(uc, s_ts, e_ts):
                click.echo("Overlaps existing interval - try again\n")
                continue
            uc.loc[len(uc)] = [name, s_ts, e_ts]
            click.echo(f"Added {name} {s_ts.date()} -> {e_ts.date() if e_ts else 'open-ended'}")
        except ValueError as exc:
            click.echo(f"Error: {exc}")
    if uc.empty:
        click.echo("No intervals - leaving uc.parquet untouched.")
        return uc
    uc.sort_values("start_date", inplace=True)
    dump_parquet(uc, UC_PQ)
    log.info("Saved timeline -> %s", UC_PQ)
    return uc


def make_signature(variants: list[str]) -> str:
    """Canonical, DB-compatible signature string."""
    return SEPARATOR.join(sorted(v.strip() for v in variants if v.strip()))


def make_signature_hash(signature: str) -> str:
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()


def unify_artist_names_cli(
    data: pd.DataFrame,
    fltrd_artcount: pd.DataFrame,
    similar_artist_groups: list[list[str]],
):
    """
    Interactively resolves artist-name duplicates using Click prompts.

    Commits each decision immediately to avc.parquet.
    """
    avc_df = read_parquet(AVC_PQ)
    groups_to_review = similar_artist_groups.copy()
    while groups_to_review:
        group = groups_to_review.pop(0)
        if len(group) <= 1:
            continue
        signature = make_signature(group)
        # Checking previous decisions in avc.parquet
        if avc_df is not None and not avc_df.empty:
            prev = avc_df.loc[avc_df["artist_variants_text"] == signature]
            if not prev.empty:
                row = prev.iloc[-1]
                if not row["to_link"]:
                    continue  # to skip previously skipped group
                _apply_canonical(row["canonical_name"], group, data, fltrd_artcount)
                continue
        click.echo("\n---")
        click.echo("These artist names appear to be duplicates:")
        for v in group:
            click.echo(f"  - {v}")
        choices = {str(i + 1): name for i, name in enumerate(group)}
        choices["c"] = "Custom name"
        choices["s"] = "Skip this group"
        click.echo("Choose canonical name:")
        for key, label in choices.items():
            click.echo(f"  [{key}] {label}")
        choice = click.prompt("Selection", type=str, default="s")
        if choice == "s":
            comment = click.prompt("Optional comment (Enter to skip)", default="", show_default=False) or None
            _remember_artist_variant(signature, "__SKIP__", False, comment)
            continue
        if choice == "c":
            canonical = click.prompt("Enter the custom canonical name").strip()
        elif choice in choices:
            canonical = choices[choice]
        else:
            click.echo("Invalid choice, skipping.")
            continue
        if not canonical:
            click.echo("No canonical name provided, skipping.")
            _remember_artist_variant(signature, "__SKIP__", False, "Skipped: no name")
            continue
        comment = click.prompt("Optional comment (Enter to skip)", default="", show_default=False) or None
        _apply_canonical(canonical, group, data, fltrd_artcount)
        _remember_artist_variant(signature, canonical, True, comment)
        # Reloading avc_df after each write to keep it fresh
        avc_df = read_parquet(AVC_PQ)
    refreshed = data["Artist"].value_counts().rename_axis("Artist").reset_index(name="Count")
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
        (2, "I Have the Moon, You Have the Internet", "I Have the Moon You Have the Internet"),
    ]
    banner = "\n── Checking whether CSV export kept the internal commas ──"
    print(banner)
    for col, with_comma, no_comma in probes:
        series = df.iloc[:, col] if isinstance(col, int) else (df[col] if col in df.columns else None)
        if series is None:
            print(f'[warn] column "{col!r}" is not found in CSV - skipped')
            continue
        kept = int(series.eq(with_comma).sum())
        lost = int(series.eq(no_comma).sum())
        print(
            f"\n» column {col:5}  "
            f"→ '{with_comma}' OR '{no_comma}?'\n"
            f"    rows *with* comma   : {kept}\n"
            f"    rows sans comma     : {lost}"
        )
    print("\n──────────────────────────────────────────────────────────\n")


def yes_no(question: str, *, default: str = "n") -> bool:
    """
    Returns true when the user answers yes
    """
    return ask(question, default).lower().startswith("y")

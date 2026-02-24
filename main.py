"""
Click-based entry point for c9r (formerly CanonFodder).

Provides composable CLI command groups: ingest, enrich, canonise, review,
train, serve, dashboard, purge, flow.
"""
from __future__ import annotations
import logging
import math
import os
import signal
import sys
from pathlib import Path
import click
from dotenv import load_dotenv
load_dotenv()
log = logging.getLogger("c9r")

# ── Source selection helpers ─────────────────────────────────────────────────────
_SOURCE_CHOICES = click.Choice(["lastfm", "listenbrainz", "lb"], case_sensitive=False)


def _normalise_source(source: str) -> str:
    """Normalises the 'lb' alias to 'listenbrainz'."""
    return "listenbrainz" if source.lower() == "lb" else source.lower()


def _resolve_user(source: str, user: str | None) -> str:
    """Resolves username from --user flag or source-appropriate env var."""
    if user:
        return user
    envvar = "LASTFM_USER" if source == "lastfm" else "LB_USER"
    val = os.environ.get(envvar)
    if not val:
        raise click.UsageError(f"--user is required (or set {envvar} in .env)")
    return val


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """c9r — scrobble ingestion, enrichment, and canonisation toolkit."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ── ingest ─────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--user", "-u", default=None, help="Username (env: LASTFM_USER or LB_USER).")
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE).")
@click.option("--full", is_flag=True, help="Fetch full history instead of incremental.")
def ingest(user: str | None, source: str, full: bool) -> None:
    """Fetches scrobbles and appends to scrobble.parquet."""
    source = _normalise_source(source)
    user = _resolve_user(source, user)
    from helpers.io import ingest_scrobbles, latest_scrobble_ts
    since = None if full else latest_scrobble_ts()
    click.echo(f"Fetching scrobbles for {user} from {source}" + (f" since uts={since}" if since else " (full)") + " …")
    if source == "lastfm":
        from HTTP.lfAPI import fetch_scrobbles_since
    else:
        from HTTP.lblink import fetch_scrobbles_since
    df = fetch_scrobbles_since(user, since=since)
    if df.empty:
        click.echo("No new scrobbles.")
        return
    n = ingest_scrobbles(df)
    click.echo(f"Ingested {n} scrobbles.")


# ── enrich ─────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--user", "-u", default=None, help="Username (env: LASTFM_USER or LB_USER).")
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE).")
@click.option("--mbids/--no-mbids", default=True, help="Enrich missing artist MBIDs via Last.fm.")
@click.option("--country/--no-country", default=True, help="Sync user country from Last.fm.")
def enrich(user: str | None, source: str, mbids: bool, country: bool) -> None:
    """Enriches scrobble data with MBIDs and MusicBrainz metadata."""
    source = _normalise_source(source)
    user = _resolve_user(source, user)
    if source == "listenbrainz":
        if mbids:
            click.echo("Skipping MBID enrichment — ListenBrainz already provides MBIDs.")
        if country:
            click.echo("Skipping country sync — not available for ListenBrainz.")
        return
    from HTTP.lfAPI import enrich_artist_mbids, sync_user_country
    if mbids:
        click.echo(f"Enriching artist MBIDs for {user} …")
        result = enrich_artist_mbids(user)
        click.echo(f"{result['status']}: {result['message']}")
    if country:
        try:
            changed = sync_user_country(user, ask=False)
            click.echo("Country updated." if changed else "Country already up-to-date.")
        except RuntimeError as exc:
            click.echo(f"Country sync skipped: {exc}")


# ── canonise ───────────────────────────────────────────────────────────────
@cli.command()
def canonise() -> None:
    """Runs the artist-name canonisation pipeline (fuzzy matching + ML)."""
    click.echo("Canonise command — will be wired in Phase 6/7.")


# ── review ─────────────────────────────────────────────────────────────────
@cli.command()
def review() -> None:
    """Interactively reviews artist-name variant groups."""
    click.echo("Review command — will be wired in Phase 6.")


# ── train ──────────────────────────────────────────────────────────────────
@cli.command()
def train() -> None:
    """Trains the XGBoost canonisation model."""
    from corefunc.canon import train_model
    click.echo("Training XGBoost model …")
    train_model()
    click.echo("Done.")


# ── serve ──────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=8000, type=int, help="Port to listen on.")
def serve(host: str, port: int) -> None:
    """Starts the FastAPI model-serving endpoint."""
    import uvicorn
    click.echo(f"Starting model server on {host}:{port} …")
    uvicorn.run("corefunc.model_server:app", host=host, port=port, workers=1)


# ── dashboard ───────────────────────────────────────────────────────────────
@cli.command()
@click.option("--top", "-n", default=10, type=int, help="Number of top artists to show.")
def dashboard(top: int) -> None:
    """Prints a quick text dashboard of scrobble statistics."""
    from helpers.query import scrobble_count, unique_artists, top_artists
    total = scrobble_count()
    artists = unique_artists()
    click.echo(f"Scrobbles: {total:,}   Unique artists: {artists:,}")
    df = top_artists(top)
    if not df.empty:
        click.echo(f"\nTop {top} artists:")
        for _, row in df.iterrows():
            click.echo(f"  {row['play_count']:>6,}  {row['artist_name']}")


# ── purge ──────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--all", "purge_all", is_flag=True, help="Purge all Parquet files.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt (only with --all).")
def purge(purge_all: bool, yes: bool) -> None:
    """Removes Parquet data files.

    Without --all, presents each file interactively for selection.
    """
    from helpers.io import PQ_DIR
    targets = sorted(PQ_DIR.glob("*.parquet"))
    if not targets:
        click.echo("No Parquet files found.")
        return
    if purge_all:
        if not yes:
            click.confirm("This will delete all Parquet data files. Continue?", abort=True)
        for p in targets:
            p.unlink()
            click.echo(f"Deleted {p.name}")
    else:
        deleted = 0
        for p in targets:
            if click.confirm(f"Delete {p.name}?"):
                p.unlink()
                click.echo(f"  Deleted {p.name}")
                deleted += 1
            else:
                click.echo(f"  Skipped {p.name}")
        click.echo(f"\nPurged {deleted} of {len(targets)} file(s).")


# ── fix-encoding ─────────────────────────────────────────────────────────────────
@cli.command("fix-encoding")
def fix_encoding_cmd() -> None:
    """Repairs encoding-corrupted strings in scrobble.parquet."""
    from corefunc.data_cleaning import fix_encoding
    click.echo("Scanning for encoding issues …")
    fixed, total = fix_encoding()
    if fixed:
        click.echo(f"Repaired {fixed} rows out of {total:,} total.")
    else:
        click.echo("No encoding issues found.")


# ── qa ─────────────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def qa(ctx: click.Context) -> None:
    """Runs or queries post-ingestion quality checks."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@qa.command("scrobble")
@click.option("--hours", "-h", default=None, type=int, help="Only check scrobbles from the last N hours.")
@click.option("--source", "-s", type=_SOURCE_CHOICES, default=None, envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE).")
def qa_scrobble(hours: int | None, source: str | None) -> None:
    """Runs QA checks on scrobble.parquet."""
    from corefunc.qa import qa_lb_ingest
    source = _normalise_source(source) if source else None
    click.echo("Running scrobble QA checks …")
    report = qa_lb_ingest(last_n_hours=hours, source=source)
    if report.get("status") == "skipped":
        click.echo(f"Skipped: {report['reason']}")
        return
    # Printing summary
    passed = report["passed"]
    click.echo(f"\nRows checked: {report['row_count']:,}")
    click.echo(f"Overall: {'PASS' if passed else 'FAIL'}")
    # Schema
    sch = report["schema"]
    if not sch["pass"]:
        click.echo(f"  Schema FAIL — missing: {sch['missing']}, unexpected: {sch['unexpected']}")
    # Nulls
    for col, stats in report["nulls"].items():
        if stats["null_pct"] > 0 or stats["empty_pct"] > 0:
            click.echo(f"  {col}: {stats['null_pct']}% null, {stats['empty_pct']}% empty")
        else:
            click.echo(f"  {col}: clean")
    # Timestamps
    ts = report["timestamps"]
    if not ts["pass"]:
        for issue in ts["issues"]:
            click.echo(f"  Timestamp: {issue}")
    # Duplicates
    dup = report["duplicates"]
    click.echo(f"  Duplicates: {dup['duplicate_count']:,} ({dup['duplicate_pct']}%)")
    if not dup["pass"]:
        click.echo("  ⚠ Duplicate rate exceeds 5% threshold")
    # MBIDs
    mb = report["mbids"]
    click.echo(f"  MBID fill: {mb.get('fill_rate', 0)}%, valid: {mb.get('valid_rate', 0)}%")
    # Encoding
    enc = report["encoding"]
    if not enc["pass"]:
        click.echo(f"  Encoding: {enc['bad_char_rows']} rows with bad characters")
    # Reconciliation
    rec = report["reconciliation"]
    if rec.get("fetched") is not None:
        click.echo(f"  Reconciliation: fetched={rec['fetched']:,}, stored={rec['stored']:,}, diff={rec.get('diff', 0):,}")
    click.echo("\nReport appended to PQ/qa_report.parquet")


@qa.command("a_i")
def qa_artist_info_cmd() -> None:
    """Runs QA checks on artist_info.parquet."""
    from corefunc.qa import qa_artist_info
    click.echo("Running artist_info QA checks …")
    report = qa_artist_info()
    if report.get("status") == "skipped":
        click.echo(f"Skipped: {report['reason']}")
        return
    passed = report["passed"]
    click.echo(f"\nRows checked: {report['row_count']:,}")
    click.echo(f"Overall: {'PASS' if passed else 'FAIL'}")
    sch = report["schema"]
    if not sch["pass"]:
        click.echo(f"  Schema FAIL — missing: {sch['missing']}, unexpected: {sch['unexpected']}")
    for col, stats in report["nulls"].items():
        if stats["null_pct"] > 0 or stats["empty_pct"] > 0:
            click.echo(f"  {col}: {stats['null_pct']}% null, {stats['empty_pct']}% empty")
        else:
            click.echo(f"  {col}: clean")
    dup = report["duplicates"]
    click.echo(f"  Duplicates: {dup['duplicate_count']:,} ({dup['duplicate_pct']}%)")
    if not dup["pass"]:
        click.echo("  ⚠ Duplicate rate exceeds 5% threshold")
    mb = report["mbids"]
    click.echo(f"  MBID fill: {mb.get('fill_rate', 0)}%, valid: {mb.get('valid_rate', 0)}%")
    enc = report["encoding"]
    if not enc["pass"]:
        click.echo(f"  Encoding: {enc['bad_char_rows']} rows with bad characters")
    click.echo("\nReport appended to PQ/qa_report.parquet")


@qa.command("avc")
def qa_avc_cmd() -> None:
    """Runs QA checks on avc.parquet."""
    from corefunc.qa import qa_avc
    click.echo("Running avc QA checks …")
    report = qa_avc()
    if report.get("status") == "skipped":
        click.echo(f"Skipped: {report['reason']}")
        return
    passed = report["passed"]
    click.echo(f"\nRows checked: {report['row_count']:,}")
    click.echo(f"Overall: {'PASS' if passed else 'FAIL'}")
    sch = report["schema"]
    if not sch["pass"]:
        click.echo(f"  Schema FAIL — missing: {sch['missing']}, unexpected: {sch['unexpected']}")
    for col, stats in report["nulls"].items():
        if stats["null_pct"] > 0 or stats["empty_pct"] > 0:
            click.echo(f"  {col}: {stats['null_pct']}% null, {stats['empty_pct']}% empty")
        else:
            click.echo(f"  {col}: clean")
    dup = report["duplicates"]
    click.echo(f"  Duplicates: {dup['duplicate_count']:,} ({dup['duplicate_pct']}%)")
    ts = report["timestamps"]
    if not ts["pass"]:
        for issue in ts["issues"]:
            click.echo(f"  Timestamp: {issue}")
    enc = report["encoding"]
    if not enc["pass"]:
        click.echo(f"  Encoding: {enc['bad_char_rows']} rows with bad characters")
    click.echo("\nReport appended to PQ/qa_report.parquet")


@qa.command("uc")
def qa_uc_cmd() -> None:
    """Shows summary stats for uc.parquet (user-country history)."""
    from corefunc.qa import qa_uc
    click.echo("Running uc summary …")
    report = qa_uc()
    if report.get("status") == "skipped":
        click.echo(f"Skipped: {report['reason']}")
        return
    click.echo(f"\n  Entries: {report['row_count']:,}")
    click.echo(f"  Unique countries: {report['unique_countries']}")
    click.echo("\nReport appended to PQ/qa_report.parquet")


def _format_qa_src(row) -> str:
    """Builds the src= display value from source and target columns."""
    import pandas as pd
    parts: list[str] = []
    for key in ("source", "target"):
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if s and s not in ("None", "nan", "<NA>"):
            parts.append(s)
    return "/".join(parts)


@qa.command()
@click.option("--last", "last_n", default=5, type=int, help="Number of recent reports to show.")
@click.option("--all", "show_all", is_flag=True, help="Show all reports (overrides --last).")
@click.option("--fail-only", is_flag=True, help="Only show failed reports.")
def show(last_n: int, show_all: bool, fail_only: bool) -> None:
    """Displays past QA reports from qa_report.parquet."""
    from helpers.query import qa_reports
    limit = None if show_all else last_n
    df = qa_reports(last_n=limit, fail_only=fail_only)
    if df.empty:
        click.echo("No QA reports found.")
        return
    click.echo(f"{'All' if show_all else f'Last {last_n}'} QA reports"
               f"{' (failures only)' if fail_only else ''}:\n")
    for _, row in df.iterrows():
        status = "PASS" if row["passed"] else "FAIL"
        ts = str(row["timestamp"])[:19]
        src = _format_qa_src(row)
        src_str = f"  src={src}" if src else ""
        target = str(row.get("target", ""))
        _hr = row.get("hash_fill_rate", 0)
        hash_rate = 0 if (_hr is None or (isinstance(_hr, float) and math.isnan(_hr))) else _hr
        _mr = row.get("mbid_fill_rate", 0)
        mbid_rate = 0 if (_mr is None or (isinstance(_mr, float) and math.isnan(_mr))) else _mr
        if target == "user_country":
            _uc = row.get("unique_countries")
            countries = int(_uc) if _uc is not None and not (isinstance(_uc, float) and math.isnan(_uc)) else "?"
            click.echo(f"  {ts}  {status}{src_str}"
                       f"  rows={int(row['row_count']):,}"
                       f"  countries={countries}")
        else:
            if target == "artist_variants_canonized":
                fill_str = f"hash_fill={hash_rate}%"
            else:
                fill_str = f"mbid_fill={mbid_rate}%"
            click.echo(f"  {ts}  {status}{src_str}"
                       f"  rows={int(row['row_count']):,}"
                       f"  dupes={row['duplicate_pct']}%"
                       f"  {fill_str}"
                       f"  bad_chars={int(row['bad_char_rows'])}")


# ── flow ───────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE).")
@click.option("--full", is_flag=True, help="Fetch full history instead of incremental.")
def flow(source: str, full: bool) -> None:
    """Runs the full Prefect orchestration flow."""
    source = _normalise_source(source)
    from flows.cf_ingest import weekly_ingest_flow
    click.echo("Starting Prefect flow …")
    result = weekly_ingest_flow(full=full, source=source)
    click.echo(f"Done — {result['new_scrobbles']} scrobbles, {result['enriched_artists']} enriched.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    cli()

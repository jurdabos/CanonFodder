"""
Click-based entry point for c9r (formerly CanonFodder).

Provides composable CLI command groups: ingest, enrich, canonise, review,
train, serve, dashboard, purge, flow.
"""
from __future__ import annotations
import logging
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
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", help="Data source.")
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
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", help="Data source.")
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
@click.confirmation_option(prompt="This will delete data. Continue?")
def purge(purge_all: bool) -> None:
    """Removes Parquet data files."""
    from helpers.io import PQ_DIR
    targets = list(PQ_DIR.glob("*.parquet")) if purge_all else []
    if not targets:
        click.echo("Nothing to purge (use --all).")
        return
    for p in targets:
        p.unlink()
        click.echo(f"Deleted {p.name}")


# ── qa ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--hours", "-h", default=None, type=int, help="Only check scrobbles from the last N hours.")
def qa(hours: int | None) -> None:
    """Runs post-ingestion quality checks on scrobble.parquet."""
    from corefunc.qa import qa_lb_ingest, ALBUM_NULL_THRESHOLD
    click.echo("Running QA checks …")
    report = qa_lb_ingest(last_n_hours=hours)
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


# ── flow ───────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", help="Data source.")
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

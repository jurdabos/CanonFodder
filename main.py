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
    """Fetches scrobbles and appends to scrobble.parquet"""
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


# ── enrich ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--mbapi", is_flag=True, help="Use remote MusicBrainz API instead of local mirror.")
@click.option("--lastfmapi", is_flag=True, help="Use Last.fm API for MBIDs + remote MB API for metadata.")
@click.option("--country", is_flag=True, default=False, help="Sync user country to uc.parquet (requires --user).")
@click.option("--rebuild", is_flag=True, help="Rebuild artist_info.parquet from scratch.")
@click.option("--user", "-u", default=None, help="Username (only for --country; env: LASTFM_USER or LB_USER).")
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (only for --country; env: C9R_SOURCE).")
def enrich(mbapi: bool, lastfmapi: bool, country: bool, rebuild: bool, user: str | None, source: str) -> None:
    """Enrich scrobble & artist_info with MBIDs and metadata

    Default: local MusicBrainz mirror.  Use --mbapi or --lastfmapi for remote.
    """
    if mbapi and lastfmapi:
        raise click.UsageError("--mbapi and --lastfmapi are mutually exclusive.")
    backend = "mbapi" if mbapi else "lastfmapi" if lastfmapi else "local"
    label = {"local": "local MB mirror", "mbapi": "remote MB API", "lastfmapi": "Last.fm + remote MB API"}[backend]
    click.echo(f"Enriching via {label} …")
    from corefunc.enrich import enrich_all
    try:
        result = enrich_all(backend=backend, rebuild=rebuild)
        click.echo(
            f"Done — {result['artist_info_rows']:,} artist_info rows, "
            f"{result['mbids_backfilled']:,} MBIDs backfilled into scrobble.parquet."
        )
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        return
    if country:
        source = _normalise_source(source)
        if source == "listenbrainz":
            click.echo("Skipping country sync — not available for ListenBrainz.")
        else:
            user = _resolve_user(source, user)
            from HTTP.lfAPI import sync_user_country
            try:
                changed = sync_user_country(user, ask=False)
                click.echo("Country updated." if changed else "Country already up-to-date.")
            except RuntimeError as exc:
                click.echo(f"Country sync skipped: {exc}")


# ── canonise ───────────────────────────────────────────────────────────────
@cli.command()
def canonise() -> None:
    """Run artist name canonisation w/ ML bump"""
    click.echo("Canonise command — will be wired in Phase 6/7.")


# ── review ─────────────────────────────────────────────────────────────────
@cli.command()
def review() -> None:
    """Review artist-name variant groups interactively"""
    click.echo("Review command — will be wired in Phase 6.")


# ── train ──────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--run-name", default=None, help="MLflow run name for this training session.")
def train(run_name: str | None) -> None:
    """Train a canonisation model"""
    from corefunc.canon import train_model
    click.echo("Training XGBoost model …")
    train_model(run_name=run_name)
    click.echo("Done.")


# ── mlflow-ui ──────────────────────────────────────────────────────────────────
@cli.command("mlflow-ui")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=5000, type=int, help="Port to listen on.")
def mlflow_ui(host: str, port: int) -> None:
    """Launches the MLflow tracking UI for experiment comparison"""
    from helpers.experiment import TRACKING_URI
    click.echo(f"Starting MLflow UI at http://{host}:{port}  (store: {TRACKING_URI})")
    os.execvp("mlflow", ["mlflow", "ui", "--backend-store-uri", TRACKING_URI, "--host", host, "--port", str(port)])


# ── serve ──────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=8000, type=int, help="Port to listen on.")
def serve(host: str, port: int) -> None:
    """Start the ASGI model-serving endpoint"""
    import uvicorn
    click.echo(f"Starting model server on {host}:{port} …")
    uvicorn.run("corefunc.model_server:app", host=host, port=port, workers=1)


# ── dashboard ───────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def dashboard(ctx: click.Context) -> None:
    """Dash the board for scrobble data"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@dashboard.command("artist")
@click.option("--top", "-n", default=10, type=int, help="Number of top artists to show.")
def dashboard_artist(top: int) -> None:
    """Shows top artists by play count."""
    from helpers.query import scrobble_count, unique_artists, top_artists
    total = scrobble_count()
    artists = unique_artists()
    click.echo(f"Scrobbles: {total:,}   Unique artists: {artists:,}")
    df = top_artists(top)
    if not df.empty:
        click.echo(f"\nTop {top} artists:\n")
        for i, (_, row) in enumerate(df.iterrows(), 1):
            click.echo(f"  {i:>3}.  {row['play_count']:>6,}  {row['artist_name']}")


@dashboard.command("album")
@click.option("--top", "-n", default=10, type=int, help="Number of top albums to show.")
def dashboard_album(top: int) -> None:
    """Shows top albums by play count."""
    from helpers.query import top_albums
    df = top_albums(top)
    if df.empty:
        click.echo("No album data found.")
        return
    click.echo(f"Top {top} albums:\n")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        click.echo(f"  {i:>3}.  {row['play_count']:>6,}  {row['artist_name']}: {row['album_title']}")


@dashboard.command("track")
@click.option("--top", "-n", default=10, type=int, help="Number of top tracks to show.")
def dashboard_track(top: int) -> None:
    """Shows top tracks by play count."""
    from helpers.query import top_tracks
    df = top_tracks(top)
    if df.empty:
        click.echo("No track data found.")
        return
    click.echo(f"Top {top} tracks:\n")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        album = row["album_title"] or ""
        album_part = f" ({album})" if album else ""
        click.echo(f"  {i:>3}.  {row['play_count']:>6,}  {row['artist_name']}: {row['track_title']}{album_part}")


@dashboard.command("recent")
@click.option("-n", default=10, type=int, help="Number of recent scrobbles to show.")
def dashboard_recent(n: int) -> None:
    """Shows the most recent scrobbles."""
    from helpers.query import recent_scrobbles
    df = recent_scrobbles(n)
    if df.empty:
        click.echo("No scrobbles found.")
        return
    click.echo(f"Last {len(df)} scrobbles:\n")
    for _, row in df.iterrows():
        album = row["album_title"] or ""
        album_part = f" ({album})" if album else ""
        ts = str(row["play_time"])[:19]
        click.echo(f"  {row['artist_name']}: {row['track_title']}{album_part} | {ts}")


# ── purge ──────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--all", "purge_all", is_flag=True, help="Purge all Parquet files.")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt (only with --all).")
def purge(purge_all: bool, yes: bool) -> None:
    """Remove Parquet data files

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
    """Repair encoding-corrupted strings in scrobble & artist_info"""
    from corefunc.data_cleaning import fix_encoding
    click.echo("Scanning for encoding issues …")
    results = fix_encoding()
    any_fixed = False
    for label, (fixed, total) in results.items():
        if fixed:
            click.echo(f"  {label}: repaired {fixed} of {total:,} rows.")
            any_fixed = True
    if not any_fixed:
        click.echo("No encoding issues found.")


# ── qa ─────────────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def qa(ctx: click.Context) -> None:
    """Run or query post-ingestion quality checks"""
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
    # Enrichment fill rates (real values only, excluding None/empty)
    enr = report.get("enrichment", {})
    if enr:
        total = report["row_count"]
        click.echo("  Enrichment (real values):")
        for label, key in [("Country", "country"), ("Disambiguation", "disambiguation"), ("Aliases", "aliases")]:
            stats = enr.get(key, {})
            filled = stats.get("filled", 0)
            rate = stats.get("fill_rate", 0.0)
            click.echo(f"    {label}: {filled:,} / {total:,} ({rate}%)")
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
        elif target == "artist_info":
            # Showing enrichment fill rates for artist_info reports
            _cr = row.get("country_fill_rate", 0)
            country_rate = 0 if (_cr is None or (isinstance(_cr, float) and math.isnan(_cr))) else _cr
            _dr = row.get("disambiguation_fill_rate", 0)
            disambig_rate = 0 if (_dr is None or (isinstance(_dr, float) and math.isnan(_dr))) else _dr
            _ar = row.get("aliases_fill_rate", 0)
            aliases_rate = 0 if (_ar is None or (isinstance(_ar, float) and math.isnan(_ar))) else _ar
            click.echo(f"  {ts}  {status}{src_str}"
                       f"  rows={int(row['row_count']):,}"
                       f"  dupes={row['duplicate_pct']}%"
                       f"  mbid={mbid_rate}%"
                       f"  country={country_rate}%"
                       f"  disambig={disambig_rate}%"
                       f"  aliases={aliases_rate}%"
                       f"  bad_chars={int(row['bad_char_rows'])}")
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


# ── profile ─────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def profile(ctx: click.Context) -> None:
    """Data profiling - see subcommands"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@profile.command()
def overview() -> None:
    """Prints high-level scrobble and distribution statistics."""
    from corefunc.profile import overview_stats
    stats = overview_stats()
    if "error" in stats:
        click.echo(f"Error: {stats['error']}")
        return
    click.echo(f"Scrobbles: {stats['total_scrobbles']:,}")
    click.echo(f"Unique artists: {stats['unique_artists']:,}")
    click.echo(f"Unique tracks:  {stats['unique_tracks']:,}")
    click.echo(f"Unique albums:  {stats['unique_albums']:,}")
    click.echo(f"Date range: {stats['earliest']}  →  {stats['latest']}")
    click.echo("\nYearly scrobbles:")
    for year, plays in stats["yearly"]:
        bar = "█" * max(1, int(plays / max(p for _, p in stats["yearly"]) * 40))
        click.echo(f"  {int(year)}  {plays:>6,}  {bar}")
    d = stats["distribution"]
    click.echo("\nPlay-count distribution per artist:")
    click.echo(f"  min={d['min']}  Q1={d['q25']}  median={d['median']}  Q3={d['q75']}  max={d['max']:,}  mean={d['mean']:.1f}")
    click.echo(f"  Singletons (1 play): {d['singletons']:,} / {d['total_artists']:,}"
               f" ({100 * d['singletons'] / d['total_artists']:.1f}%)")
    click.echo(f"  ≤5 plays:           {d['lte5']:,} / {d['total_artists']:,}"
               f" ({100 * d['lte5'] / d['total_artists']:.1f}%)")


@profile.command()
@click.option("--threshold", "-t", default=85, type=int, help="Minimum fuzzy similarity score (0-100).")
@click.option("--min-plays", "-m", default=3, type=int, help="Minimum play count to consider.")
@click.option("--limit", "-l", default=500, type=int, help="Max artists to compare.")
@click.option("--top", "-n", default=20, type=int, help="Number of results to show.")
def variants(threshold: int, min_plays: int, limit: int, top: int) -> None:
    """Finds fuzzy-similar artist names that split scrobble counts (the Bohren problem)."""
    from corefunc.profile import variant_candidates
    click.echo(f"Scanning top {limit} artists (≥{min_plays} plays) for near-duplicates (threshold={threshold}) …")
    clusters = variant_candidates(threshold=threshold, min_plays=min_plays, limit=limit)
    if not clusters:
        click.echo("No near-duplicate pairs found.")
        return
    click.echo(f"\nFound {len(clusters)} candidate pair(s).  Top {min(top, len(clusters))} by combined count:\n")
    for i, c in enumerate(clusters[:top], 1):
        names = "  ↔  ".join(f"{v['name']} ({v['plays']:,})" for v in c["variants"])
        click.echo(f"  {i:>3}.  {names}")
        click.echo(f"        combined={c['combined_count']:,}  similarity={c['similarity']}%")
    if len(clusters) > top:
        click.echo(f"\n  … and {len(clusters) - top} more pair(s).  Use --top to show more.")


@profile.command("top")
@click.option("-n", default=20, type=int, help="Number of top artists.")
@click.option("--canonized", is_flag=True, help="Apply AVC canonisation before ranking.")
def profile_top(n: int, canonized: bool) -> None:
    """Shows top artists by play count, optionally after canonisation."""
    from corefunc.profile import top_artists_profile
    result = top_artists_profile(n, canonize=canonized)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(f"Top {n} artists (raw):\n")
    for entry in result["raw_top"]:
        click.echo(f"  {entry['rank']:>3}.  {entry['plays']:>6,}  {entry['name']}")
    if canonized and "canon_top" in result:
        click.echo(f"\nTop {n} artists (after AVC canonisation — {result['mapping_size']} mappings):\n")
        for entry in result["canon_top"]:
            click.echo(f"  {entry['rank']:>3}.  {entry['plays']:>6,}  {entry['name']}")
    elif canonized:
        click.echo("\navc.parquet not found — canonised ranking unavailable.")


@profile.command()
@click.option("--start", default=2006, type=int, help="Start year (inclusive).")
@click.option("--end", default=2025, type=int, help="End year (inclusive).")
@click.option("-n", default=10, type=int, help="Number of companions to show.")
def companions(start: int, end: int, n: int) -> None:
    """Finds artists listened to in every year of the range (trusted companions)."""
    from corefunc.profile import trusted_companions
    result = trusted_companions(start_year=start, end_year=end)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    total = len(result["companions"])
    if total == 0:
        click.echo(f"No artists appear in every year between {start}–{end}.")
        return
    years = result["years"]
    click.echo(f"Found {total} artist(s) present in all {result['year_count']} years ({years[0]}–{years[-1]}).")
    click.echo(f"\nMost consistent (lowest σ of yearly plays), top {min(n, total)}:\n")
    click.echo(f"  {'Artist':<40} {'Total':>7} {'Mean/yr':>8} {'σ':>7}")
    click.echo(f"  {'─' * 40} {'─' * 7} {'─' * 8} {'─' * 7}")
    for c in result["companions"][:n]:
        click.echo(f"  {c['name']:<40} {c['total_plays']:>7,} {c['mean_per_year']:>8.1f} {c['std_dev']:>7.1f}")


@profile.command()
@click.option("-n", default=15, type=int, help="Number of top countries to show.")
def countries(n: int) -> None:
    """Shows top countries by scrobble count from artist_info enrichment."""
    from corefunc.profile import country_breakdown
    rows = country_breakdown(top_n=n)
    if not rows:
        click.echo("No enriched country data available.")
        return
    click.echo(f"Top {len(rows)} countries by scrobble count:\n")
    click.echo(f"  {'#':>4}  {'CC':<4} {'Plays':>8} {'Artists':>8} {'Share':>7}  {'Name'}")
    click.echo(f"  {'─' * 4}  {'──':<4} {'─' * 8} {'─' * 8} {'─' * 7}  {'─' * 4}")
    for i, r in enumerate(rows, 1):
        click.echo(f"  {i:>3}.  {r['country']:<4} {r['play_count']:>8,} {r['artist_count']:>8,} {r['pct']:>6.1f}%  {r['name']}")


# ── flow ───────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE).")
@click.option("--full", is_flag=True, help="Fetch full history instead of incremental.")
def flow(source: str, full: bool) -> None:
    """Run the full Prefect orchestration flow"""
    source = _normalise_source(source)
    from flows.cf_ingest import weekly_ingest_flow
    click.echo("Starting Prefect flow …")
    result = weekly_ingest_flow(full=full, source=source)
    click.echo(f"Done — {result['new_scrobbles']} scrobbles, {result['enriched_artists']} enriched.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    cli()

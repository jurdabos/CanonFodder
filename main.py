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

try:
    from acidbase.cli_utils import group
    from acidbase.push import push_command
    from acidbase.versioning import bump_command
except ImportError:  # to keep the CLI usable when acidbase is absent
    group = click.group  # plain click help formatting as the fallback
    bump_command = None
    push_command = None

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


@group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """c9r — scrobble ingestion, enrichment, and canonisation toolkit."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("mlflow").setLevel(logging.ERROR)


# ── ingest ─────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--user", "-u", default=None, help="Username (env: LASTFM_USER or LB_USER).")
@click.option(
    "--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE)."
)
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
@click.option(
    "--source",
    "-s",
    type=_SOURCE_CHOICES,
    default="lastfm",
    envvar="C9R_SOURCE",
    help="Data source (only for --country; env: C9R_SOURCE).",
)
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


# ── canon (command group) ──────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def canon(ctx: click.Context) -> None:
    """Artist name canonisation — show, propagate, review, discover"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@canon.group(invoke_without_command=True)
@click.pass_context
def avc(ctx: click.Context) -> None:
    """Artist Variants Canonized table operations"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@avc.command("show")
@click.option("--decided", is_flag=True, help="Show only decided rows (to_link 0 or 1).")
@click.option("--undecided", is_flag=True, help="Show only undecided rows (to_link NULL).")
@click.option("--last", "last_n", default=None, type=int, help="Show only the last N rows.")
def avc_show(decided: bool, undecided: bool, last_n: int | None) -> None:
    """Print out the current state of the avc table"""
    from corefunc.canon.workflow import avc_summary

    rows = avc_summary(decided_only=decided, undecided_only=undecided, last_n=last_n)
    if not rows:
        click.echo("No avc rows found.")
        return
    click.echo(f"\nAVC table ({len(rows)} row{'s' if len(rows) != 1 else ''}):\n")
    click.echo(f"  {'#':>4}  {'Link':<4}  {'Canonical Name':<40}  {'Decided':<10}  {'Variants'}")
    click.echo(f"  {'─' * 4}  {'─' * 4}  {'─' * 40}  {'─' * 10}  {'─' * 8}")
    for r in rows:
        cn = r["canonical_name"][:40]
        click.echo(
            f"  {r['idx']:>4}  {r['to_link_display']:<4}  {cn:<40}  {r['stamp']:<10}  {r['artist_variants_text']}"
        )


@avc.command("propagate")
def avc_propagate() -> None:
    """Apply canonisation results to artist_info"""
    from corefunc.canon.workflow import propagate_avc

    click.echo("Propagating AVC decisions to artist_info …")
    result = propagate_avc()
    click.echo(f"Done — {result['updated']} row(s) updated, {result['aliases_added']} alias(es) added.")


@avc.command("seed")
@click.argument("sql_path", type=click.Path(exists=True))
def avc_seed_cmd(sql_path: str) -> None:
    """Seed avc.parquet from a MySQL dump file"""
    from corefunc.avc_seed import seed_avc_from_sql

    click.echo(f"Seeding avc.parquet from {sql_path} …")
    n = seed_avc_from_sql(sql_path)
    click.echo(f"Done — {n} rows written to avc.parquet.")


@avc.command("augment")
@click.option("--pos-limit", default=5000, type=int, help="Max positive (alias→canonical) pairs.")
@click.option("--neg-limit", default=5000, type=int, help="Max negative pairs.")
@click.option("--similarity-floor", default=60, type=int, help="WRatio floor for hard negatives (0-100).")
def avc_augment(pos_limit: int, neg_limit: int, similarity_floor: int) -> None:
    """Extract training pairs from local MB mirror into gs_mb"""
    from corefunc.canon.augment import augment_gold_standard

    click.echo(f"Extracting pairs from MBDB (pos={pos_limit}, neg={neg_limit}, floor={similarity_floor}) …")
    try:
        n = augment_gold_standard(
            pos_limit=pos_limit,
            neg_limit=neg_limit,
            similarity_floor=similarity_floor,
        )
        click.echo(f"Done — {n} rows written to gs_mb.parquet.")
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)


@canon.command("human")
def canon_human() -> None:
    """Tackle undecided artist name variants interactively"""
    import re

    from corefunc.canon.workflow import undecided_rows, update_avc_decision

    pending = undecided_rows()
    if pending.empty:
        click.echo("No undecided variant groups — all caught up.")
        return

    # Extracting model probabilities from comment field and sorting descending
    def _parse_prob(comment: str) -> float:
        """Extracts probability from 'p=0.XXXX' comment prefix."""
        m = re.match(r"p=(\d+\.\d+)", str(comment or ""))
        return float(m.group(1)) if m else 0.0

    pending["_prob"] = pending["comment"].apply(_parse_prob)
    pending = pending.sort_values("_prob", ascending=False).reset_index(drop=True)
    total = len(pending)
    click.echo(f"\n{total} undecided variant group(s) to review.\n")
    for i, (_, row) in enumerate(pending.iterrows(), 1):
        variants = [v.strip() for v in str(row["artist_variants_text"]).split("{") if v.strip()]
        if len(variants) < 2:
            continue
        prob = row["_prob"]
        prob_display = f"  [model p={prob:.4f}]" if prob > 0 else ""
        click.echo(f"\n── ({i} of {total}){prob_display} ──")
        click.echo("These artist names appear to be variants:")
        choices: dict[str, str] = {}
        for j, v in enumerate(variants, 1):
            click.echo(f"  [{j}] {v}")
            choices[str(j)] = v
        click.echo("  [c] Custom name")
        click.echo("  [s] Skip (not the same)")
        click.echo("  [q] Quit review")
        choice = click.prompt("Selection", type=str, default="s")
        if choice.lower() == "q":
            click.echo("Review stopped.")
            break
        if choice.lower() == "s":
            comment = click.prompt("Optional comment", default="", show_default=False) or ""
            update_avc_decision(row["artist_variants_hash"], False, "__SKIP__", comment)
            click.echo("  → skipped.")
            continue
        if choice.lower() == "c":
            canonical = click.prompt("Enter the custom canonical name").strip()
        elif choice in choices:
            canonical = choices[choice]
        else:
            click.echo("Invalid choice, skipping.")
            update_avc_decision(row["artist_variants_hash"], False, "__SKIP__", "invalid choice")
            continue
        if not canonical:
            click.echo("No name provided, skipping.")
            update_avc_decision(row["artist_variants_hash"], False, "__SKIP__", "no name")
            continue
        comment = click.prompt("Optional comment", default="", show_default=False) or ""
        update_avc_decision(row["artist_variants_hash"], True, canonical, comment)
        click.echo(f"  → linked as '{canonical}'.")
    click.echo("\nHuman review complete.")


@canon.command("experiment")
@click.option("--run-name", default=None, help="MLflow parent run name.")
@click.option("--augment/--no-augment", default=True, help="Include MBDB pairs from gs_mb.parquet.")
@click.option("--folds", default=5, type=int, help="Number of CV folds.")
@click.option("--models", default=None, help="Comma-separated model names to run (default: all).")
def canon_experiment(run_name: str | None, augment: bool, folds: int, models: str | None) -> None:
    """Run multi-model experiment with CV and MLflow logging"""
    from corefunc.canon.experiment_runner import run_experiment

    model_list = [m.strip() for m in models.split(",")] if models else None
    label = " (with MBDB augmentation)" if augment else ""
    click.echo(f"Running multi-model experiment{label}, {folds}-fold CV …")
    if model_list:
        click.echo(f"Models: {', '.join(model_list)}")
    run_experiment(
        augment=augment,
        n_folds=folds,
        run_name=run_name,
        models=model_list,
    )
    click.echo("\nExperiment complete. View results with 'c9r mlflow-ui'.")


@canon.command("machine")
@click.option("--cutoff", default=75, type=int, help="RapidFuzz WRatio pre-filter cutoff (0-100).")
@click.option("--threshold", default=0.5, type=float, help="ML model probability threshold.")
@click.option("--min-plays", default=2, type=int, help="Minimum play count to consider.")
@click.option("--limit", default=2000, type=int, help="Max artists to scan.")
def canon_machine(cutoff: int, threshold: float, min_plays: int, limit: int) -> None:
    """Finds new artist name variant candidates using ML"""
    from corefunc.canon.workflow import discover_candidates, write_new_candidates
    from helpers.inference import MODEL_PATH, load_model

    # Loading the persisted LightGBM pipeline
    try:
        model = load_model()
    except FileNotFoundError:
        click.echo(f"Model not found at {MODEL_PATH}. Train first with 'c9r train'.")
        return
    click.echo(f"Model loaded ({len(model.feature_names_in_)} features).")
    click.echo(f"Scanning for variant candidates (cutoff={cutoff}, threshold={threshold}) …")
    candidates = discover_candidates(
        model,
        wratio_cutoff=cutoff,
        proba_threshold=threshold,
        min_plays=min_plays,
        limit=limit,
    )
    if not candidates:
        click.echo("No new variant candidates found.")
        return
    click.echo(f"\nFound {len(candidates)} new candidate group(s):")
    for c in candidates[:20]:
        prob_str = f"  (p={c['max_prob']:.4f})" if "max_prob" in c else ""
        click.echo(f"  {' ↔ '.join(c['variants'])}{prob_str}")
    if len(candidates) > 20:
        click.echo(f"  … and {len(candidates) - 20} more.")
    if click.confirm("\nAdd these to avc.parquet for human review?", default=True):
        n = write_new_candidates(candidates)
        click.echo(f"Written {n} candidate group(s) to avc.parquet.")
    else:
        click.echo("Cancelled.")


# ── train (command group) ──────────────────────────────────────────────────────────
_DATA_SOURCE_CHOICES = click.Choice(
    ["avc", "mbdb", "mbdb-max", "dbscan", "dbscan-capped", "mixed"],
    case_sensitive=False,
)
_FEATURE_STRATEGY_CHOICES = click.Choice(["standard", "separated"], case_sensitive=False)
_NEG_MATCHING_CHOICES = click.Choice(["none", "distribution"], case_sensitive=False)
_SPLIT_CHOICES = click.Choice(["pair", "group"], case_sensitive=False)
_TEST_SOURCE_CHOICES = click.Choice(["holdout", "avc-full"], case_sensitive=False)
_FEATURE_CHOICES = click.Choice(["base", "interaction", "full"], case_sensitive=False)
_CAT_SOURCE_CHOICES = click.Choice(
    ["none", "scrobble", "mbdb", "unified"],
    case_sensitive=False,
)
_CAT_DESIGN_CHOICES = click.Choice(["proportional", "presence"], case_sensitive=False)


@cli.group(invoke_without_command=True)
@click.pass_context
def train(ctx: click.Context) -> None:
    """Train canonisation models — see subcommands"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@train.command("run")
@click.option("--run-name", default=None, help="MLflow parent run name (auto-generated if omitted).")
@click.option("--folds", default=5, type=int, help="Number of CV folds.")
@click.option("--test-size", default=0.20, type=float, help="Held-out test fraction.")
@click.option("--models", default=None, help="Comma-separated model names (default: LightGBM).")
@click.option("--catalogue/--no-catalogue", default=True, help="Include catalogue features.")
@click.option("--data-source", type=_DATA_SOURCE_CHOICES, default="avc", help="Training data origin.")
@click.option(
    "--split", "split_strategy", type=_SPLIT_CHOICES, default="pair", help="Split strategy (pair or group level)."
)
@click.option(
    "--test-source",
    type=_TEST_SOURCE_CHOICES,
    default="holdout",
    help="Test data origin (holdout from split, or full AVC).",
)
@click.option(
    "--features", type=_FEATURE_CHOICES, default="full", help="Feature tiers: base (23), interaction (53), full (71)."
)
@click.option("--catalogue-source", type=_CAT_SOURCE_CHOICES, default="unified", help="Catalogue data origin.")
@click.option("--catalogue-design", type=_CAT_DESIGN_CHOICES, default="proportional", help="Catalogue feature style.")
@click.option("--group-features/--no-group-features", default=False, help="Include group-level length_stats features.")
@click.option("--wratio-lower", default=60, type=int, help="WRatio band lower bound.")
@click.option("--wratio-upper", default=100, type=int, help="WRatio band upper bound.")
@click.option(
    "--experiment", "experiment_num", default=None, type=int, help="Experiment number for backfill labelling."
)
@click.option("--include-composites", is_flag=True, help="Include composite models (Voting, Stacking, Bagging).")
@click.option(
    "--cluster-cap", default=0, type=int, help="Max cluster size for dbscan-capped (Exp 7, default 30 when used)."
)
@click.option(
    "--neg-ratio", default=0, type=int, help="Target neg:pos ratio for dbscan-capped (Exp 7, default 10 when used)."
)
@click.option(
    "--feature-strategy",
    type=_FEATURE_STRATEGY_CHOICES,
    default="standard",
    help="Feature strategy: standard or separated (Exp 8).",
)
@click.option(
    "--neg-matching",
    type=_NEG_MATCHING_CHOICES,
    default="none",
    help="Negative matching: none or distribution (Exp 8).",
)
@click.option("--neg-count", default=5000, type=int, help="Target count for distribution-matched negatives (Exp 8).")
def train_run(
    run_name: str | None,
    folds: int,
    test_size: float,
    models: str | None,
    catalogue: bool,
    data_source: str,
    split_strategy: str,
    test_source: str,
    features: str,
    catalogue_source: str,
    catalogue_design: str,
    group_features: bool,
    wratio_lower: int,
    wratio_upper: int,
    experiment_num: int | None,
    include_composites: bool,
    cluster_cap: int,
    neg_ratio: int,
    feature_strategy: str,
    neg_matching: str,
    neg_count: int,
) -> None:
    """Train canonisation models (unified pipeline with MLflow tracking)"""
    from corefunc.canon.trainer import run_training

    model_list = [m.strip() for m in models.split(",")] if models else None
    cat_label = " +catalogue" if catalogue and features == "full" else " (no catalogue)"
    strategy_label = f", strategy={feature_strategy}" if feature_strategy != "standard" else ""
    click.echo(
        f"Training pipeline — data={data_source}, split={split_strategy}, "
        f"features={features}{cat_label}{strategy_label}, {folds}-fold CV …"
    )
    if model_list:
        click.echo(f"Models: {', '.join(model_list)}")
    run_training(
        run_name=run_name,
        n_folds=folds,
        test_size=test_size,
        models=model_list,
        catalogue=catalogue,
        data_source=data_source,
        split_strategy=split_strategy,
        test_source=test_source,
        features=features,
        catalogue_source=catalogue_source,
        catalogue_design=catalogue_design,
        group_features=group_features,
        wratio_lower=wratio_lower,
        wratio_upper=wratio_upper,
        experiment_num=experiment_num,
        include_composites=include_composites,
        cluster_cap=cluster_cap,
        neg_ratio=neg_ratio,
        feature_strategy=feature_strategy,
        neg_matching=neg_matching,
        neg_count=neg_count,
    )
    click.echo("\nTraining complete. View results with 'c9r mlflow-ui'.")


_TCN_MODEL_CHOICES = click.Choice(["siamese", "hybrid"], case_sensitive=False)


@train.command("tcn")
@click.option(
    "--model",
    "model_type",
    type=_TCN_MODEL_CHOICES,
    default="siamese",
    help="TCN architecture: siamese (Exp 9) or hybrid (Exp 10).",
)
@click.option("--epochs", default=80, type=int, help="Max training epochs.")
@click.option("--batch-size", default=None, type=int, help="Mini-batch size (default: 256 siamese, 512 hybrid).")
@click.option("--lr", default=None, type=float, help="Learning rate (default: 1e-3 siamese, 3e-4 hybrid).")
@click.option("--patience", default=12, type=int, help="Early stopping patience.")
@click.option("--experiment", "experiment_num", default=None, type=int, help="Experiment number for MLflow labelling.")
@click.option("--run-name", default=None, help="MLflow run name.")
def train_tcn(
    model_type: str,
    epochs: int,
    batch_size: int | None,
    lr: float | None,
    patience: int,
    experiment_num: int | None,
    run_name: str | None,
) -> None:
    """Train TCN-based canonisation models (Siamese or Hybrid architecture)"""
    from corefunc.canon.tcn_trainer import run_tcn_training

    click.echo(f"TCN training — model={model_type}, epochs={epochs}, patience={patience} …")
    run_tcn_training(
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        experiment_num=experiment_num,
        run_name=run_name,
    )
    click.echo("\nTCN training complete. View results with 'c9r mlflow-ui'.")


# ── tune ───────────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--run-name", default=None, help="MLflow parent run name (auto-generated if omitted).")
@click.option("--models", default=None, help="Comma-separated model names (default: LightGBM).")
@click.option("--trials", default=60, type=int, help="Optuna trials per model.")
@click.option("--folds", default=3, type=int, help="CV folds for tuning inner loop.")
@click.option("--test-size", default=0.20, type=float, help="Held-out test fraction.")
@click.option("--min-precision", default=0.90, type=float, help="Precision floor for the objective.")
@click.option("--catalogue/--no-catalogue", default=True, help="Include catalogue features.")
def tune(
    run_name: str | None,
    models: str | None,
    trials: int,
    folds: int,
    test_size: float,
    min_precision: float,
    catalogue: bool,
) -> None:
    """Optuna hyperparameter tuning with precision-biased objective"""
    from corefunc.canon.tuner import run_tuning

    model_list = [m.strip() for m in models.split(",")] if models else None
    click.echo(f"Optuna tuning — {trials} trials/model, {folds}-fold CV, min_precision={min_precision} …")
    if model_list:
        click.echo(f"Models: {', '.join(model_list)}")
    run_tuning(
        run_name=run_name,
        models=model_list,
        n_trials=trials,
        n_folds=folds,
        test_size=test_size,
        min_precision=min_precision,
        catalogue=catalogue,
    )
    click.echo("\nTuning complete. View results with 'c9r mlflow-ui'.")


# ── mlflow-ui ──────────────────────────────────────────────────────────────────
@cli.command("mlflow-ui")
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", "-p", default=5000, type=int, help="Port to listen on.")
def mlflow_ui(host: str, port: int) -> None:
    """Launches the MLflow tracking UI for experiment comparison"""
    from helpers.experiment import TRACKING_URI

    click.echo(f"Starting MLflow UI at http://{host}:{port}  (store: {TRACKING_URI})")
    os.execvp(
        "mlflow",
        ["mlflow", "ui", "--backend-store-uri", TRACKING_URI, "--host", host, "--port", str(port), "--workers", "1"],
    )


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
    """Show top artists by play count"""
    from helpers.query import scrobble_count, top_artists, unique_artists

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
    """Show top albums by play count"""
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
    """Show top tracks"""
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
    """Show most recent scrobbles"""
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


_MEDAL = {1: "GOLD", 2: "SILVER", 3: "BRONZE"}


@dashboard.command("yearly")
@click.option("--top", "-n", default=3, type=int, help="Number of top artists per year.")
def dashboard_yearly(top: int) -> None:
    """Show top artists per year (gold / silver / bronze)"""
    from corefunc.profile import yearly_top_artists_profile

    result = yearly_top_artists_profile(top_n=top)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(f"Top {result['top_n']} artists by year:\n")
    for yr in result["years"]:
        click.echo(f"  {yr['year']}")
        for a in yr["artists"]:
            medal = _MEDAL.get(a["rank"], f"#{a['rank']}")
            click.echo(f"    [{medal}] {a['name']}  ({a['plays']:,} plays)")


@dashboard.command("festival")
@click.argument("artists", required=False)
@click.option(
    "--from-file",
    "from_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Read artist names from a file (one per line, or comma-separated).",
)
def dashboard_festival(artists: str | None, from_file: str | None) -> None:
    """Tally scrobble counts for a comma-separated list of artists.

    Each name is matched case-insensitively against ``artist_name`` using a
    substring (``ILIKE '%name%'``) so partial / lower-case input works.
    Handy for sizing up a festival lineup before the gig.

    Examples:

        c9r dashboard festival "the molotovs, primal scream, anna calvi"

        c9r dashboard festival --from-file lineup.txt
    """
    from helpers.query import scrobble_counts_for_artist_patterns

    raw_parts: list[str] = []
    if from_file:
        from pathlib import Path

        file_text = Path(from_file).read_text(encoding="utf-8")
        for line in file_text.splitlines():
            raw_parts.extend(line.split(","))
    if artists:
        raw_parts.extend(artists.split(","))
    labels = [p.strip() for p in raw_parts if p and p.strip()]
    if not labels:
        raise click.UsageError("Provide a comma-separated list of artists or --from-file.")
    df = scrobble_counts_for_artist_patterns(labels)
    if df.empty:
        click.echo("No scrobble data available.")
        return
    total = int(df["scrobble_count"].sum())
    matched_labels = int((df["scrobble_count"] > 0).sum())
    name_width = max(len(str(name)) for name in df["canonical_artist_name"])
    click.echo(f"Festival lineup tally — {matched_labels}/{len(df)} artist(s) found, {total:,} matching scrobble(s):\n")
    for _, row in df.iterrows():
        marker = " " if row["scrobble_count"] > 0 else "·"  # to flag zero-count rows
        click.echo(f"  {marker} {row['scrobble_count']:>6,}  {row['canonical_artist_name']:<{name_width}}")


# ── purge
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


# ── migrate-scrobbles ─────────────────────────────────────────────────────────
@cli.command("migrate-scrobbles")
@click.option("--remove-legacy", is_flag=True, help="Delete legacy scrobble.parquet after migration.")
def migrate_scrobbles_cmd(remove_legacy: bool) -> None:
    """Convert legacy scrobble.parquet to year-partitioned layout"""
    from helpers.io import SCROBBLE_PQ, migrate_scrobble_to_partitioned

    n = migrate_scrobble_to_partitioned()
    if n == 0:
        click.echo("Nothing to migrate — legacy scrobble.parquet not found or empty.")
        return
    click.echo(f"Migrated {n:,} scrobbles to partitioned layout.")
    if remove_legacy:
        SCROBBLE_PQ.unlink(missing_ok=True)
        click.echo("Removed legacy scrobble.parquet.")
    else:
        click.echo("Legacy file kept. Use --remove-legacy to delete it.")


# ── schema ─────────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def schema(ctx: click.Context) -> None:
    """Inspect or migrate Parquet schema versions"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@schema.command("show")
def schema_show() -> None:
    """Display schema version status of all Parquet files."""
    from helpers.io import PQ_DIR, SCROBBLE_PQ_DIR
    from helpers.schema import validate_schema

    rows: list[dict[str, object]] = []

    # Collecting rows first so the "Table" column can be sized to the longest name.
    if SCROBBLE_PQ_DIR.exists() and any(SCROBBLE_PQ_DIR.rglob("*.parquet")):
        first = next(SCROBBLE_PQ_DIR.rglob("*.parquet"))
        info = validate_schema(first)
        rows.append(
            {
                "table": info.get("table") or "?",
                "file_version": str(info.get("file_version", "?")),
                "current_version": str(info.get("current_version") or "?"),
                "status": info.get("status") or "?",
                "path": "scrobble/",
                "missing_cols": info.get("missing_cols") or [],
            }
        )

    for pf in sorted(PQ_DIR.glob("*.parquet")):
        info = validate_schema(pf)
        tbl = info.get("table") or pf.stem
        cur = info.get("current_version")
        rows.append(
            {
                "table": tbl,
                "file_version": str(info.get("file_version", "?")),
                "current_version": str(cur) if cur is not None else "?",
                "status": info.get("status") or "?",
                "path": pf.name,
                "missing_cols": info.get("missing_cols") or [],
            }
        )

    table_w = max([len("Table")] + [len(str(r["table"])) for r in rows]) if rows else len("Table")
    status_w = max([len("Status")] + [len(str(r["status"])) for r in rows]) if rows else len("Status")

    click.echo(f"{'Table':<{table_w}} {'File ver':>8} {'Current':>8}  {'Status':<{status_w}} Path")
    click.echo(f"{'─' * table_w} {'─' * 8} {'─' * 8}  {'─' * status_w} {'─' * 4}")

    for r in rows:
        click.echo(
            f"{r['table']:<{table_w}} {r['file_version']:>8} {r['current_version']:>8}  "
            f"{r['status']:<{status_w}} {r['path']}"
        )
        missing = r.get("missing_cols")
        if missing:
            click.echo(f"{'':<{table_w}} {'':>8} {'':>8}  missing: {', '.join(missing)}")


@schema.command("migrate")
def schema_migrate() -> None:
    """Migrate all Parquet files to current schema versions"""
    from helpers.io import PQ_DIR
    from helpers.schema import migrate_all

    click.echo("Migrating Parquet schemas \u2026")
    results = migrate_all(PQ_DIR)
    if not results:
        click.echo("No Parquet files found to migrate.")
        return
    for name, status in results.items():
        click.echo(f"  {name:<30} {status}")
    click.echo("Done.")


# ── qa ─────────────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def qa(ctx: click.Context) -> None:
    """Run or query post-ingestion quality checks"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@qa.command("scrobble")
@click.option("--hours", "-h", default=None, type=int, help="Only check scrobbles from the last N hours.")
@click.option(
    "--source", "-s", type=_SOURCE_CHOICES, default=None, envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE)."
)
def qa_scrobble(hours: int | None, source: str | None) -> None:
    """Run QA checks on scrobble.parquet"""
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
        click.echo(
            f"  Reconciliation: fetched={rec['fetched']:,}, stored={rec['stored']:,}, diff={rec.get('diff', 0):,}"
        )
    _print_qa_history("scrobble")


@qa.command("a_i")
def qa_artist_info_cmd() -> None:
    """Run QA checks on artist_info"""
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
    _print_qa_history("artist_info")


@qa.command("avc")
def qa_avc_cmd() -> None:
    """Run QA checks on avc"""
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
    _print_qa_history("artist_variants_canonized")


@qa.command("gs_mb")
def qa_gs_mb_cmd() -> None:
    """Run QA checks on the MB gold-standard pairs"""
    from corefunc.qa import qa_gs_mb

    click.echo("Running gs_mb QA checks …")
    report = qa_gs_mb()
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
    enc = report["encoding"]
    if not enc["pass"]:
        click.echo(f"  Encoding: {enc['bad_char_rows']} rows with bad characters")
    # Printing label distribution and source breakdown
    dist = report.get("label_distribution", {})
    click.echo(
        f"  Labels: {dist.get('positive', 0):,} positive, "
        f"{dist.get('negative', 0):,} negative, {dist.get('null', 0):,} null"
    )
    src_bk = report.get("source_breakdown", {})
    if src_bk:
        parts = ", ".join(f"{k}={v:,}" for k, v in src_bk.items())
        click.echo(f"  Sources: {parts}")
    _print_qa_history("gs_mb")


@qa.command("uc")
def qa_uc_cmd() -> None:
    """Show summary stats for user country history"""
    from corefunc.qa import qa_uc

    click.echo("Running uc summary …")
    report = qa_uc()
    if report.get("status") == "skipped":
        click.echo(f"Skipped: {report['reason']}")
        return
    click.echo(f"\n  Entries: {report['row_count']:,}")
    click.echo(f"  Unique countries: {report['unique_countries']}")
    _print_qa_history("user_country")


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


def _format_qa_row(row) -> str:
    """Formats a single QA report row as a one-liner string."""
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
        return f"  {ts}  {status}{src_str}  rows={int(row['row_count']):,}  countries={countries}"
    if target == "artist_info":
        _cr = row.get("country_fill_rate", 0)
        country_rate = 0 if (_cr is None or (isinstance(_cr, float) and math.isnan(_cr))) else _cr
        _dr = row.get("disambiguation_fill_rate", 0)
        disambig_rate = 0 if (_dr is None or (isinstance(_dr, float) and math.isnan(_dr))) else _dr
        _ar = row.get("aliases_fill_rate", 0)
        aliases_rate = 0 if (_ar is None or (isinstance(_ar, float) and math.isnan(_ar))) else _ar
        return (
            f"  {ts}  {status}{src_str}"
            f"  rows={int(row['row_count']):,}"
            f"  dupes={row['duplicate_pct']}%"
            f"  mbid={mbid_rate}%"
            f"  country={country_rate}%"
            f"  disambig={disambig_rate}%"
            f"  aliases={aliases_rate}%"
            f"  bad_chars={int(row['bad_char_rows'])}"
        )
    if target == "gs_mb":
        return (
            f"  {ts}  {status}{src_str}"
            f"  rows={int(row['row_count']):,}"
            f"  dupes={row['duplicate_pct']}%"
            f"  bad_chars={int(row['bad_char_rows'])}"
        )
    if target == "artist_variants_canonized":
        fill_str = f"hash_fill={hash_rate}%"
    else:
        fill_str = f"mbid_fill={mbid_rate}%"
    return (
        f"  {ts}  {status}{src_str}"
        f"  rows={int(row['row_count']):,}"
        f"  dupes={row['duplicate_pct']}%"
        f"  {fill_str}"
        f"  bad_chars={int(row['bad_char_rows'])}"
    )


def _print_qa_history(target: str, last_n: int = 5) -> None:
    """Prints recent QA one-liners for a given target."""
    from helpers.query import qa_reports

    df = qa_reports(last_n=last_n, target=target)
    if df.empty:
        return
    click.echo(f"\nLast {len(df)} {target} QA reports:\n")
    for _, row in df.iterrows():
        click.echo(_format_qa_row(row))


@qa.command()
@click.option("--last", "last_n", default=5, type=int, help="Number of recent reports to show.")
@click.option("--all", "show_all", is_flag=True, help="Show all reports (overrides --last).")
@click.option("--fail-only", is_flag=True, help="Only show failed reports.")
def show(last_n: int, show_all: bool, fail_only: bool) -> None:
    """Display past QA reports from qa_report.parquet"""
    from helpers.query import qa_reports

    limit = None if show_all else last_n
    df = qa_reports(last_n=limit, fail_only=fail_only)
    if df.empty:
        click.echo("No QA reports found.")
        return
    click.echo(f"{'All' if show_all else f'Last {last_n}'} QA reports{' (failures only)' if fail_only else ''}:\n")
    for _, row in df.iterrows():
        click.echo(_format_qa_row(row))


# ── profile ─────────────────────────────────────────────────────────────────
@cli.group(invoke_without_command=True)
@click.pass_context
def profile(ctx: click.Context) -> None:
    """Data profiling - see subcommands"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@profile.command()
def overview() -> None:
    """Print eagle-level scrobble stats"""
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
    click.echo(
        f"  min={d['min']}  Q1={d['q25']}  median={d['median']}  Q3={d['q75']}  max={d['max']:,}  mean={d['mean']:.1f}"
    )
    click.echo(
        f"  Singletons (1 play): {d['singletons']:,} / {d['total_artists']:,}"
        f" ({100 * d['singletons'] / d['total_artists']:.1f}%)"
    )
    click.echo(
        f"  ≤5 plays:           {d['lte5']:,} / {d['total_artists']:,} ({100 * d['lte5'] / d['total_artists']:.1f}%)"
    )


@profile.command()
@click.option("--threshold", "-t", default=91, type=int, help="Minimum fuzzy similarity score (0-100).")
@click.option("--min-plays", "-m", default=3, type=int, help="Minimum play count to consider.")
@click.option("--limit", "-l", default=500, type=int, help="Max artists to compare.")
@click.option("--top", "-n", default=20, type=int, help="Number of results to show.")
def variants(threshold: int, min_plays: int, limit: int, top: int) -> None:
    """Show examples for the Bohren problem"""
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


def _parse_rank_ranges(raw: str) -> list[tuple[int, int]]:
    """
    Parses a custom rank-range string like '(1,5),(27,29)' into sorted,
    validated (start, end) tuples.
    """
    import re

    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", raw)
    if not pairs:
        raise click.BadParameter(f"Cannot parse rank ranges from: {raw}")
    ranges: list[tuple[int, int]] = []
    for a, b in pairs:
        lo, hi = int(a), int(b)
        if lo < 1 or hi < lo:
            raise click.BadParameter(f"Invalid range ({lo}, {hi}) — must be 1 ≤ start ≤ end.")
        ranges.append((lo, hi))
    ranges.sort()
    return ranges


def _echo_ranged_entries(entries: list[dict], ranges: list[tuple[int, int]]) -> None:
    """Prints entries filtered by rank ranges with '...' between gaps."""
    for idx, (lo, hi) in enumerate(ranges):
        if idx > 0:
            click.echo("  ...")
        for entry in entries:
            if lo <= entry["rank"] <= hi:
                click.echo(f"  {entry['rank']:>3}.  {entry['plays']:>6,}  {entry['name']}")


@profile.command("top")
@click.option("-n", default=20, type=int, help="Number of top artists.")
@click.option("--canonized", is_flag=True, help="Apply alias-based canonisation before ranking.")
@click.option("--custom", "custom_ranges", default=None, type=str, help="Custom rank ranges, e.g. '(1,5),(27,29)'.")
def profile_top(n: int, canonized: bool, custom_ranges: str | None) -> None:
    """Show top by scrobble optionally after canonisation"""
    from corefunc.profile import top_artists_profile

    # Determining how many rows to fetch
    ranges: list[tuple[int, int]] | None = None
    if custom_ranges:
        ranges = _parse_rank_ranges(custom_ranges)
        fetch_n = max(hi for _, hi in ranges)
    else:
        fetch_n = n
    result = top_artists_profile(fetch_n, canonize=canonized)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    # Displaying raw top
    if ranges:
        click.echo("Top artists (raw), custom ranges:\n")
        _echo_ranged_entries(result["raw_top"], ranges)
    else:
        click.echo(f"Top {n} artists (raw):\n")
        for entry in result["raw_top"]:
            click.echo(f"  {entry['rank']:>3}.  {entry['plays']:>6,}  {entry['name']}")
    # Displaying canonised top
    if canonized and "canon_top" in result:
        if ranges:
            click.echo("\nTop artists (canonised), custom ranges:\n")
            _echo_ranged_entries(result["canon_top"], ranges)
        else:
            click.echo(f"\nTop {n} artists (canonised):\n")
            for entry in result["canon_top"]:
                click.echo(f"  {entry['rank']:>3}.  {entry['plays']:>6,}  {entry['name']}")
    elif canonized:
        click.echo("\nartist_info.parquet not found — canonised ranking unavailable.")


@profile.command()
@click.option("--start", default=2006, type=int, help="Start year (inclusive).")
@click.option("--end", default=2025, type=int, help="End year (inclusive).")
@click.option("-n", default=10, type=int, help="Number of companions to show.")
def companions(start: int, end: int, n: int) -> None:
    """Find trusted companions"""
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
    """Show top countries by scrobble count"""
    from corefunc.profile import country_breakdown

    rows = country_breakdown(top_n=n)
    if not rows:
        click.echo("No enriched country data available.")
        return
    click.echo(f"Top {len(rows)} countries by scrobble count:\n")
    click.echo(f"  {'#':>4}  {'CC':<4} {'Plays':>8} {'Artists':>8} {'Share':>7}  {'Name'}")
    click.echo(f"  {'─' * 4}  {'──':<4} {'─' * 8} {'─' * 8} {'─' * 7}  {'─' * 4}")
    for i, r in enumerate(rows, 1):
        click.echo(
            f"  {i:>3}.  {r['country']:<4} {r['play_count']:>8,} {r['artist_count']:>8,} {r['pct']:>6.1f}%  {r['name']}"
        )


@profile.command()
def timeline() -> None:
    """Show monthly scrobble summary across all years"""
    from corefunc.profile import monthly_summary

    result = monthly_summary()
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo("Monthly scrobble summary (averaged across all years):\n")
    click.echo(f"  {'Month':<12} {'Mean':>7} {'Min':>7} {'Max':>7} {'Total':>8} {'Years':>5}")
    click.echo(f"  {'─' * 12} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 5}")
    for m in result["months"]:
        click.echo(
            f"  {m['name']:<12} {m['mean']:>7.1f} {m['min']:>7,} {m['max']:>7,} {m['total']:>8,} {m['year_count']:>5}"
        )
    s = result["strongest"]
    w = result["weakest"]
    if s and w:
        click.echo(f"\n  Strongest month: {s['name']} (mean {s['mean']:.1f})")
        click.echo(f"  Weakest month:  {w['name']} (mean {w['mean']:.1f})")


@profile.command()
def streaks() -> None:
    """Show listening streak and gap statistics"""
    from corefunc.profile import streak_analysis

    result = streak_analysis()
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo("Listening streaks & gaps:\n")
    click.echo(f"  Active days: {result['total_active_days']:,}  ({result['first_day']} → {result['last_day']})")
    click.echo(
        f"  Longest streak: {result['longest_streak']} day(s)"
        f"  ({result['longest_streak_start']} → {result['longest_streak_end']})"
    )
    click.echo(f"  Current streak: {result['current_streak']} day(s)")
    if result["longest_gap_days"] > 0:
        click.echo(
            f"  Longest gap: {result['longest_gap_days']} day(s)"
            f"  ({result['longest_gap_start']} → {result['longest_gap_end']})"
        )


@profile.command()
def clock() -> None:
    """Show when you listen — hour-of-day and day-of-week patterns"""
    from corefunc.profile import listening_clock_profile

    result = listening_clock_profile()
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    # Hourly breakdown
    if result["hours"]:
        click.echo("Hour of day:\n")
        max_cnt = max(h["count"] for h in result["hours"])
        for h in result["hours"]:
            bar_len = int(h["count"] / max_cnt * 30) if max_cnt else 0
            bar = "█" * max(1, bar_len)
            click.echo(f"  {h['label']}  {h['count']:>7,}  {h['pct']:>5.1f}%  {bar}")
        pk = result["peak_hour"]
        qt = result["quiet_hour"]
        if pk and qt:
            click.echo(f"\n  Peak: {pk['label']} ({pk['count']:,})   Quiet: {qt['label']} ({qt['count']:,})")
    # Weekly breakdown
    if result["weekdays"]:
        click.echo("\nDay of week:\n")
        max_cnt = max(d["count"] for d in result["weekdays"])
        for d in result["weekdays"]:
            bar_len = int(d["count"] / max_cnt * 30) if max_cnt else 0
            bar = "█" * max(1, bar_len)
            click.echo(f"  {d['name']:<4} {d['count']:>7,}  {d['pct']:>5.1f}%  {bar}")
        pk = result["peak_day"]
        qt = result["quiet_day"]
        if pk and qt:
            click.echo(f"\n  Peak: {pk['name']} ({pk['count']:,})   Quiet: {qt['name']} ({qt['count']:,})")


@profile.command()
@click.option("-n", default=20, type=int, help="Number of top countries to show per ranking.")
def population(n: int) -> None:
    """Correlate artist-origin country population with scrobble counts"""
    from corefunc.profile import population_vs_scrobbles

    result = population_vs_scrobbles(top_n=n)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(f"Countries matched: {result['total_countries']}\n")
    click.echo("By absolute scrobble count:\n")
    click.echo(f"  {'#':>4}  {'CC':<4} {'Plays':>8} {'Artists':>8} {'Population':>14} {'Per 1M':>10}  {'Name'}")
    click.echo(f"  {'─' * 4}  {'──':<4} {'─' * 8} {'─' * 8} {'─' * 14} {'─' * 10}  {'─' * 4}")
    for i, r in enumerate(result["by_absolute"], 1):
        click.echo(
            f"  {i:>3}.  {r['country']:<4} {r['play_count']:>8,} {r['artist_count']:>8,}"
            f" {r['population']:>14,} {r['per_million']:>10.2f}  {r['name']}"
        )
    click.echo("\nBy scrobbles per million population:\n")
    click.echo(f"  {'#':>4}  {'CC':<4} {'Per 1M':>10} {'Plays':>8} {'Population':>14}  {'Name'}")
    click.echo(f"  {'─' * 4}  {'──':<4} {'─' * 10} {'─' * 8} {'─' * 14}  {'─' * 4}")
    for i, r in enumerate(result["by_per_capita"], 1):
        click.echo(
            f"  {i:>3}.  {r['country']:<4} {r['per_million']:>10.2f} {r['play_count']:>8,}"
            f" {r['population']:>14,}  {r['name']}"
        )


@profile.command()
@click.option("--min-plays", "min_plays", default=1, type=int, help="Minimum scrobbles for an artist to count.")
def gender(min_plays: int) -> None:
    """Break down your library by MusicBrainz artist gender"""
    from corefunc.profile import gender_breakdown

    result = gender_breakdown(min_plays=min_plays)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(
        f"Gender breakdown: {result['total_scrobbles']:,} scrobbles / {result['total_artists']:,} artists "
        f"(MBID-matched {result['mbid_share']}% of artists, gender known for {result['known_share']}% of plays)\n"
    )
    click.echo(f"  {'gender':<16} {'artists':>8} {'%art':>7} {'scrobbles':>10} {'%scr':>7} {'%gendered':>10}")
    click.echo(f"  {'─' * 16} {'─' * 8} {'─' * 7} {'─' * 10} {'─' * 7} {'─' * 10}")
    for r in result["rows"]:
        pg = f"{r['pct_gendered_plays']:>9.1f}%" if r["pct_gendered_plays"] is not None else "        —"
        row = f"  {r['gender']:<16} {r['artists']:>8,} {r['pct_artists']:>6.1f}% "
        click.echo(row + f"{r['scrobbles']:>10,} {r['pct_scrobbles']:>6.1f}% {pg}")


@profile.command("where")
@click.option("-n", default=10, type=int, help="Number of top countries to show.")
def profile_where(n: int) -> None:
    """Show where you were when you scrobbled"""
    from corefunc.profile import user_country_profile

    result = user_country_profile(top_n=n)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(
        f"Scrobbles matched: {result['total_scrobbles_matched']:,}   Unique countries: {result['unique_countries']}"
    )
    click.echo(f"\n  {'#':>4}  {'CC':<4} {'Scrobbles':>10} {'Share':>7}  {'Name'}")
    click.echo(f"  {'─' * 4}  {'──':<4} {'─' * 10} {'─' * 7}  {'─' * 4}")
    for i, r in enumerate(result["countries"], 1):
        click.echo(f"  {i:>3}.  {r['country']:<4} {r['scrobble_count']:>10,} {r['pct']:>6.1f}%  {r['name']}")


def _parse_country_codes(raw: str) -> list[str]:
    """
    Parses a country-code list like '(HU, ES, DK)' or 'HU,ES,DK' into
    upper-cased ISO-2 codes.
    """
    import re

    stripped = raw.strip().strip("()")
    codes = re.split(r"[,\s]+", stripped)
    return [c.upper() for c in codes if c]


_CATEGORY_ALIASES: dict[str, str] = {
    "artist": "artists",
    "artists": "artists",
    "album": "albums",
    "albums": "albums",
    "track": "tracks",
    "tracks": "tracks",
}


def _parse_categories(raw: str) -> list[str]:
    """
    Parses a category filter like '(artist, album)' or 'track' into
    a list of internal keys ('artists', 'albums', 'tracks').
    """
    import re

    stripped = raw.strip().strip("()")
    tokens = re.split(r"[,\s]+", stripped)
    result: list[str] = []
    for t in tokens:
        key = _CATEGORY_ALIASES.get(t.lower())
        if key and key not in result:
            result.append(key)
    if not result:
        raise click.BadParameter(f"Unknown categories in: {raw}. Use artist, album, track.")
    return result


@profile.command("uc")
@click.option("-n", default=3, type=int, help="Number of entries per category (medal count).")
@click.option("--ucn", default=5, type=int, help="Number of top user-countries to include.")
@click.option(
    "-c", "countries_raw", default=None, type=str, help="Comma-separated country codes to filter, e.g. '(HU, ES, DK)'."
)
@click.option(
    "-s",
    "--show",
    "show_raw",
    default=None,
    type=str,
    help="Categories to display: artist, album, track, e.g. '(artist, track)'.",
)
@click.option(
    "--import",
    "import_path",
    default=None,
    type=str,
    help="Import the country timeline from the canonical SQL-VALUES txt into uc.parquet. "
    "Under WSL, Windows drive paths (D:\\… or D:/…) are converted to /mnt/d/… automatically.",
)
@click.option("--timeline", is_flag=True, help="Print the stored country timeline from uc.parquet.")
def profile_uc(
    n: int, ucn: int, countries_raw: str | None, show_raw: str | None, import_path: str | None, timeline: bool
) -> None:
    """Show medal tables per user-country (artists, albums, tracks)"""
    if timeline:
        # Printing the stored timeline instead of showing medals
        from corefunc.profile import user_country_timeline

        result = user_country_timeline()
        if "error" in result:
            click.echo(f"Error: {result['error']}")
            return
        click.echo(f"User-country timeline ({len(result['entries'])} entries):\n")
        for e in result["entries"]:
            end = e["end"] or "now"
            marker = "  ← current" if e["current"] else ""
            row = f"  {e['idx']:>3}.  {e['country']:<4} {e['start']} → {end:<10}  {e['days']:>5} days  {e['name']}"
            click.echo(row + marker)
        return
    if import_path is not None:
        # Importing the timeline from the canonical txt instead of showing medals
        import re

        from helpers.paths import to_wsl_mounted

        resolved = to_wsl_mounted(import_path)
        if not resolved.exists():
            hint = ""
            if re.match(r"^[A-Za-z]:", import_path) and "\\" not in import_path and "/" not in import_path:
                hint = (
                    " — the shell stripped your backslashes; "
                    "quote the path ('D:\\dir\\file.txt') or use D:/dir/file.txt"
                )
            click.echo(f"Error: source file not found: {resolved}{hint}")
            return
        from corefunc.profile import import_user_country_timeline

        result = import_user_country_timeline(resolved)
        if "error" in result:
            click.echo(f"Error: {result['error']}")
            return
        click.echo(f"Imported {result['rows']} rows into uc.parquet (previously {result['previous_rows'] or 0}).")
        if result["backup"]:
            click.echo(f"Backup of the previous table: {result['backup']}")
        if result["open_country"]:
            click.echo(f"Current country: {result['open_country']} (since {result['open_since']}).")
        return
    from corefunc.profile import user_country_medal_profile

    country_codes = _parse_country_codes(countries_raw) if countries_raw else None
    categories = _parse_categories(show_raw) if show_raw else ["artists", "albums", "tracks"]
    all_labels = [("Artists", "artists"), ("Albums", "albums"), ("Tracks", "tracks")]
    visible = [(label, key) for label, key in all_labels if key in categories]
    result = user_country_medal_profile(top_n=n, ucn=ucn, country_codes=country_codes)
    if "error" in result:
        click.echo(f"Error: {result['error']}")
        return
    click.echo(
        f"Top {result['top_n']} per category across {result['ucn']} countr{'y' if result['ucn'] == 1 else 'ies'}:\n"
    )
    for c in result["countries"]:
        click.echo(f"  {c['country']} ({c['name']}) — {c['scrobble_count']:,} scrobbles")
        for label, key in visible:
            click.echo(f"    {label}")
            entries = c[key]
            if not entries:
                click.echo("      (no data)")
                continue
            for e in entries:
                medal = _MEDAL.get(e["rank"], f"#{e['rank']}")
                click.echo(f"      [{medal}] {e['name']}  ({e['plays']:,} plays)")
        click.echo()


# ── flow ───────────────────────────────────────────────────────────────────
@cli.command()
@click.option(
    "--source", "-s", type=_SOURCE_CHOICES, default="lastfm", envvar="C9R_SOURCE", help="Data source (env: C9R_SOURCE)."
)
@click.option("--full", is_flag=True, help="Fetch full history instead of incremental.")
def flow(source: str, full: bool) -> None:
    """Run the full Prefect orchestration flow"""
    source = _normalise_source(source)
    from flows.cf_ingest import weekly_ingest_flow

    click.echo("Starting Prefect flow …")
    result = weekly_ingest_flow(full=full, source=source)
    click.echo(
        f"Done — {result['new_scrobbles']} scrobbles, {result['enriched_artists']} enriched, "
        f"{result['flagged_for_review']} flagged, {result['avc_propagated']} propagated, "
        f"{result['gs_rows_written']} GS rows, {result['models_trained']} models trained."
    )


# ── push ─────────────────────────────────────────────────────────────────────
if bump_command is not None:
    cli.add_command(bump_command)
else:
    log.debug("acidbase not installed — `c9r bump` is unavailable.")

if push_command is not None:
    # Attaching the shared acidbase commit-and-push workflow as `c9r push`.
    cli.add_command(push_command)
else:
    log.debug("acidbase not installed — `c9r push` is unavailable.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    cli()

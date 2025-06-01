"""
Populates the user_country dimension from another table
and emits a Parquet snapshot.
• Works with any SQLAlchemy URL.
• Safe to re-run: TRUNCATEs dst.user_country first.
• Can run non-interactively (env/CLI) **or** interactive if URLs are absent.
    python -m canonfodder.scripts.uc_populate \
           --src  mysql+pymysql://u:pw@host/db \
           --dst  sqlite:///DB/canonfodder.db
"""
from __future__ import annotations
from pathlib import Path
import argparse
import getpass
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text


# ──────────────────────────────────────────────────────────────────────────────
# CLI & interactive helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ask(prompt: str, secret: bool = False) -> str:
    """Prompt once until the user enters a non-blank string."""
    get = getpass.getpass if secret else input
    while True:
        value = get(prompt).strip()
        if value:
            return value
        print("Please enter a value (or Ctrl-C to abort).")


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", help="SQLAlchemy URL of source DB")
    p.add_argument("--dst", help="SQLAlchemy URL of destination DB")
    p.add_argument("--if-not-empty", action="store_true",
                   help="Abort if dst.user_country already has rows")
    return p.parse_args()


def resolve_urls(args) -> tuple[str, str]:
    """Return (src_url, dst_url), asking interactively if necessary."""
    src = args.src or os.getenv("UC_SRC")
    dst = args.dst or os.getenv("UC_DST")

    if not src:
        print("UC_SRC is not set – enter source connection info.")
        src = _ask("  SQLAlchemy URL for *source* DB: ")
    if not dst:
        print("UC_DST is not set – enter destination connection info.")
        dst = _ask("  SQLAlchemy URL for *destination* DB: ")

    return src, dst


# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_cli()
    try:
        src_url, dst_url = resolve_urls(args)
    except KeyboardInterrupt:
        sys.exit("\n⨯ aborted by user")

    src = create_engine(src_url)
    dst = create_engine(dst_url)

    with src.begin() as sc, dst.begin() as dc:
        df = pd.read_sql_table("place_dimension", sc)

        if args.if_not_empty and dc.execute(
                text("SELECT 1 FROM user_country LIMIT 1")
        ).first():
            sys.exit("Destination already populated; aborting.")

        dc.execute(text("TRUNCATE TABLE user_country"))
        df.to_sql("user_country", dc, if_exists="append", index=False)

    # ── Parquet snapshot ─────────────────────────────────────────────────────
    pq_dir = Path(__file__).resolve().parents[1] / "PQ"
    pq_dir.mkdir(exist_ok=True)
    pq_file = pq_dir / "uc.parquet"
    df.to_parquet(pq_file, compression="zstd", index=False)
    print(f"✔ user_country rows: {len(df):,}   → {pq_file}")


if __name__ == "__main__":
    main()

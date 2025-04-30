"""
Populates the user_country dimension from another table
and emits a Parquet snapshot.
* Works with any SQLAlchemy URL.
* Safe to re-run: TRUNCATEs dst.user_country first.
* Can be invoked as a module or plain script:
      python -m canonfodder.scripts.uc_populate \
             --src $UC_SRC --dst $UC_DST
"""
from pathlib import Path
import argparse
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--src', default=os.getenv('UC_SRC'),
                   help='SQLAlchemy URL of source DB')
    p.add_argument('--dst', default=os.getenv('UC_DST'),
                   help='SQLAlchemy URL of destination DB')
    p.add_argument('--if-not-empty', action='store_true',
                   help='Abort if dst.user_country already has rows')
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    if not args.src or not args.dst:
        sys.exit('UC_SRC and UC_DST must be supplied (env or CLI).')
    src = create_engine(args.src)
    dst = create_engine(args.dst)
    with src.begin() as sc, dst.begin() as dc:
        df = pd.read_sql_table('place_dimension', sc)
        if args.if_not_empty and dc.execute(
                text('SELECT 1 FROM user_country LIMIT 1')
        ).first():
            sys.exit('Destination already populated; aborting.')
        dc.execute(text('TRUNCATE TABLE user_country'))
        df.to_sql('user_country', dc, if_exists='append', index=False)
    # … Parquet
    pq_dir = Path(__file__).resolve().parents[1] / 'PQ'
    pq_dir.mkdir(exist_ok=True)
    df.to_parquet(pq_dir / 'uc.parquet', compression='zstd', index=False)
    print(f'✔ user_country rows: {len(df):,}   → {pq_dir / "uc.parquet"}')


if __name__ == '__main__':
    main()

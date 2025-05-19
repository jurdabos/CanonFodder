CanonFodder is a data-engineering pipeline with the following building blocks in development:
    — srcs: automated scrobble harvesting via last.FM API, MusicBrainz-powered artist info enrichment, user input
    — ETL process: hosted in Python script with a hybrid of manual/deterministic fuzzy-logic canonisation offered
    — DWH: backend-agnostic SQL storage with tables for scrobble, artist info, canonisations, user country and codes
    — Evaluation DB: .parquet files offering a tiny analytics stack in a star schema where scrobble is the fact table
    — BI frontend: an interactive menu launched from main.py where the user can launch data harvest, and see visuals

For all EDA and frontend visualizations, we want to use our
custom colour palettes to be found at JSON/palettes.json.
You can see the functions used in each relevant file in scripts/function_report.txt.
You can see the whole BI architecture planned in BI_architecture.mmd.

Sources:
1. last.fm API
    – user.getRecentTracks method to get scrobble info: "Artist" (+ "mbid"), "Album", "Song", "uts"
    – user.getInfo to get current user country
2. MusicBrainz API
    – lookups and searches to get artist info from MBDB, such as aliases, disambiguation comments, and artist country
3. User input
    – user country timeline: In which country was the user located on past days?
    – artist variants canonised: artist name variant groups dealt with by the user manually (from scratch or via review)

ETL pipeline:
    — rename "Artist" to "artist_name", "Album" to "album_title", "Song" to "track_title"
    – convert UTS to SQL datetime in the form of YYYY-MM-DD HH:MM:SS
    – de-dup scrobble records

DWH:
For own production, housed in local MySQL.
Code is provided to allow for users to opt for Postgres or sqlite.
1. scrobble: id, artist_name, artist_mbid, album_title, track_title, play_time
2. artist_info: id, artist_name, mbid, disambiguation_comment, aliases, country
3. artist_variants_canonized: artist_variants_hash, artist_variants_text, canonical_name, mbid, to_link, comment, stamp
4. user_country: id, country_code, start_date, end_date

Eval DB = .parquet files overwritten periodically and on-demand from DB
1. scrobble.parquet: artist_name, album_title, track_title, play_time
2. artist_info.parquet: artist_name, disambiguation_comment, country
3. avc.parquet: artist_variants, canonical_name, to_link, comment, stamp
4. uc.parquet: country_code, start_date, end_date
5. c.parquet: ISO-2, ISO-3, en_name, hu_name

We want thorough metadata management with mappings, PK/FK rules, data type coercions, Alembic migrations documented.
In the end, we want to package everything (Python code, models, dashboards, docs) into a Docker image,
and we want the application to be run by docker run canonfodder.

Stack used:
Data storage:        RDMBSs and Parquet
Languages:           Mako, markups, Python, SQL
Python libraries:    contextlib, datetime, dotenv, hashlib, os, pathlib, pickle, RapidFuzz, typing, urllib etc.
Default RDBMS:       MySQL
Static hosting:      GitHub
VCS:                 git
Workflow management: Airflow (a one-click Airflow DAG to graduate the hobby code into reproducible production)

We need proper docstrings for all functions.

Style: PEP 8
Full sentences with a finite verb present end with the appropriate punctuation mark.
Sentences without a finite verb only need a closing punctuation mark if they are followed by other sentences.
Docstrings state what the code/file/function does in third person singular
(e. g. """Returns a pandas Series indexed by ascii_char with the number of…").

# Use modern, zone-aware replacement for utcnow()
from datetime import datetime, UTC
now = datetime.now(UTC)

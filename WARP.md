# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

CanonFodder is a data engineering pipeline for music listening analytics that:
- Ingests scrobble data from Last.fm API
- Enriches artist metadata from MusicBrainz
- Canonizes artist name variants for accurate analytics
- Stores data in MySQL/SQLite via SQLAlchemy ORM
- Exports to Parquet for BI/analytics
- Provides data profiling and visualizations

## Common Development Commands

### Environment Setup
```powershell
# Using uv (recommended)
uv venv .venv
.\.venv\Scripts\activate
uv sync
uv add ".[db]"  # For database features

# For Prefect workflow orchestration
uv add ".[prefect]"
```

### Running the Application
```powershell
# Main data pipeline (interactive CLI)
python main.py

# Legacy CLI mode (direct data gathering)
python main.py --legacy-cli --username <lastfm_username>

# Enrich missing artist MBIDs
python main.py --enrich-artist-mbid --username <username>

# Debug artist aliases
python main.py --debug-artist-aliases "<artist_name_or_mbid>"

# Run Prefect workflow
python flows/cf_ingest.py
```

### Testing
```powershell
# Run all tests with coverage
pytest -v --cov=. --cov-report=term-missing

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run a single test file
pytest tests/integration/test_data_workflow.py -v
```

### Database Operations
```powershell
# Run Alembic migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Check migration status
alembic current
```

### Data Exploration
```powershell
# Interactive data profiling
python dev/profile.py

# Artist canonization exploration
python dev/canon.py
```

## Architecture & Code Organization

### Core Data Flow
1. **Data Gathering** (`corefunc/data_gathering.py`): Fetches scrobbles from Last.fm API
2. **Enrichment** (`corefunc/enrich.py`): Adds MusicBrainz metadata (country, aliases)
3. **Canonization** (`corefunc/canonizer.py`): Groups artist name variants using fuzzy matching
4. **Pipeline** (`corefunc/pipeline.py`): Orchestrates the complete workflow
5. **Profiling** (`corefunc/dataprofiler.py`): Generates analytics and visualizations

### Database Architecture
- **ORM Models** (`DB/models.py`): SQLAlchemy 2.0 models for Scrobbles, ArtistInfo, ArtistVariantsCanonized
- **Operations** (`DB/ops.py`): Database CRUD operations and bulk inserts
- **Migrations** (`alembic/`): Schema version control
- **Star Schema**: Fact table (Scrobbles) with dimension tables (ArtistInfo)

### API Integration
- **Last.fm API** (`HTTP/lfAPI.py`): Handles authentication, rate limiting, pagination
- **MusicBrainz API** (`HTTP/mbAPI.py`): Artist lookups with 1-second rate limiting

### Workflow Orchestration
- **Prefect Flows** (`flows/cf_ingest.py`): Modern workflow with retry logic and monitoring
- **Legacy Workflow** (`corefunc/workflow.py`): Original step-by-step pipeline

### Data Storage
- **Parquet Files** (`PQ/`): Columnar storage for analytics
- **JSON Config** (`JSON/`): Color palettes and configuration
- **CSV Export** (`CSV/`): Backup and manual analysis

## Key Technical Details

### Environment Variables (.env)
Required:
- `LASTFM_API_KEY`: From https://www.last.fm/api/account/create
- `LASTFM_USER`: Target username for data fetching
- `DB_URL`: mysql+pymysql://user:pass@localhost/canonfodder (or SQLite fallback)

Optional:
- `MUSICBRAINZ_USER_AGENT`: For MB API identification
- `PREFECT_API_URL`: For workflow orchestration server

### Dependency Management
- Core app uses SQLAlchemy 2.0.40+
- Platform-specific dependencies handled via markers in requirements.txt
- Use `uv` for package management to avoid conflicts
- Workflow orchestration uses Prefect 3.0+

### Performance Targets
- Process 1 million scrobbles in < 15 minutes
- Bulk inserts with ON DUPLICATE KEY UPDATE
- Connection pooling via SQLAlchemy
- Efficient Parquet columnar storage

### Canonization Algorithm
1. Groups artists by normalized names (lowercase, remove punctuation)
2. Uses MusicBrainz aliases as ground truth
3. Applies fuzzy string matching (RapidFuzz library)
4. Stores mappings in ArtistVariantsCanonized table
5. Rules defined in `corefunc/canon_rules.yaml`

### Testing Strategy
- Unit tests for individual functions
- Integration tests for database operations
- End-to-end tests for complete workflows
- Fixtures in `tests/conftest.py` for test database setup
- Mock API responses to avoid external dependencies

##

Main tables:
- `scrobbles`: Music listening events (artist, track, timestamp, mbid)
- `artist_info`: Enriched artist data (mbid, country, aliases)
- `artist_variants_canonized`: Name variant mappings
- `file_ingestion_log`: Track processed files

Alembic manages schema migrations automatically.

## Project Organization

### Key Directories
- `alembic/` - Database migrations
- `config/` - Configuration files (supervisord.conf)
- `corefunc/` - Core pipeline functions
- `DB/` - Database models and operations
- `docs/` - Documentation including migration changelog
- `flows/` - Prefect workflow definitions
- `helpers/` - Utility functions and SQL queries
- `HTTP/` - API client implementations
- `scripts/` - Utility and maintenance scripts
  - `diagnosis/` - Data quality diagnosis and repair scripts
- `tests/` - Unit, integration, and e2e tests

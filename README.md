# CanonFodder

## Project Overview

CanonFodder is a reproducible data engineering pipeline that ingests music listening events (scrobbles), enriches them with metadata, stores them in a relational data warehouse, and offers interactive analytics and visualization.

## About Scrobble Data

Scrobble data is a tabular list of records, with each row representing an event of a user listening to a song. This data is valuable to musicologists and self-tracking enthusiasts who study personal music consumption patterns. Researchers apply terms like lifelogging, quantified self, and personal informatics to describe this phenomenon.

For demonstration purposes, the current instance is shipped with default last.fm scrobbles from a user account created in 2006, accessible at https://www.last.fm/user/jurda.
The main.py CLI should prompt for username, so any new instance pulled up can be configured with wiping the demo data, and building up a new db for the current user.

## Project Motivation

Common scrobble service providers often struggle with data quality issues, particularly with artist name canonization.
CanonFodder is a research project to address this challenge by providing tools for standardizing artist name variants, ensuring accurate music listening analytics.

## Technical Foundation

The project is built with Python and SQL, using:
- last.fm API for data retrieval (https://www.last.fm/api)
- MusicBrainz for metadata enrichment
- SQLAlchemy for database operations
- Pandas and Plotly for data analysis and visualization

## Repository Structure

- **DB/**: Database models, operations, and setup
  - `models.py`: SQLAlchemy ORM models
  - `ops.py`: Database operations
  - `common.py`: Database connection setup
- **docs/**: Project documentation
- **helpers/**: Utility functions for data processing and analysis
- **JSON/**: Configuration files including color palettes for visualizations
- **PQ/**: Parquet files for efficient data storage and quick loading
- **scripts/**: Utility scripts for development and maintenance
- **tests/**: Test files organized by type (unit, integration, e2e)
- **alembic/**: Database migration scripts
- **corefunc/**: Core functionality including canonization algorithms

## Installation

### Prerequisites
- Python 3.12
- Git
- uv (Python package manager)

### Installing uv

If you don't have uv installed yet:

```powershell
# Windows
winget install Astral-sh.Uv

# macOS
brew install uv

# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Steps

1. **Clone the repository**
   ```shell
   git clone https://github.com/jurdabos/CanonFodder.git
   cd CanonFodder
   ```

2. **Set up the environment**

   **Option 1: Using uv (Recommended)**

   ```powershell
   # Create a virtual environment
   uv venv .venv
   
   # Activate the virtual environment
   # On Windows:
   .\.venv\Scripts\activate
   # On Unix/MacOS:
   # source .venv/bin/activate
   
   # Install dependencies from pyproject.toml and uv.lock
   uv sync
   
   # Optional: Install with database extras (SQLAlchemy 2.x)
   uv sync --extra db
   
   # Optional: Install with Prefect extras for workflow orchestration
   uv sync --extra prefect
   ```

   **Common uv Tasks:**
   ```powershell
   # Add a new dependency
   uv add <package-name>
   
   # Add a development dependency
   uv add --dev <package-name>
   
   # Run commands in the environment
   uv run python main.py
   uv run pytest
   
   # Export dependencies for CI
   uv export --frozen --output-file=requirements.txt
   ```

3. **Configure the application**
   - Copy `.env.example` to `.env` and fill in the required values
   - Get a free last.fm API key at https://www.last.fm/api/account/create
   - For read-only demos, only `LASTFM_API_KEY` is mandatory

4. **Database Configuration**
   - Default: MySQL (`DB_URL=mysql+pymysql://user:pass@localhost/canonfodder`)
   - Alternative: SQLite (automatic fallback if MySQL not configured)
   - The system uses Alembic and SQLAlchemy to support multiple database backends

## Usage

### Data Pipeline

Run the complete data pipeline:
```shell
uv run python main.py
```

### Interactive Development

The repository includes example Parquet files for quick exploration:

1. **Data Profiling**
   ```shell
   uv run python dev\profile.py
   ```

2. **Artist Canonization Exploration**
   ```shell
   uv run python dev\canon.py
   ```

These scripts provide a notebook-style, step-by-step exploration of the data and canonization process.

## Contributing

Contributions to CanonFodder are welcome! If you'd like to contribute, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Last.fm for providing the API to access scrobble data
- MusicBrainz for their comprehensive music metadata database
- Ben Foxall's [lastfm-to-csv](https://github.com/benfoxall/lastfm-to-csv) for inspiration on scrobble data extraction
- Research by Elsden et al. (2016) on personal music tracking and lifelogging

## Contact

For questions or feedback about CanonFodder, please contact the project maintainer.

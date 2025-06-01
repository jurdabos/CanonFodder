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

### Setup Steps
1. **Clone the repository**
   ```shell
   git clone https://github.com/jurdabos/CanonFodder.git
   cd CanonFodder
   ```

2. **Set up the environment**

   Option 1: Using Docker

   The project includes a complete Docker setup with MySQL, Airflow, and Adminer:

   ```shell
   # Ensure Docker Desktop is running on your system
   # You'll get a "pipe/dockerDesktopLinuxEngine" error if it's not running

   # Copy environment variables template
   cp .env.example .env

   # Edit .env file to add your Last.fm API key and other settings
   # At minimum, set LASTFM_API_KEY and LASTFM_USER

   # Start the Docker containers
   docker-compose up --build
   ```

   This will start the following services:
   - **MySQL Database**: Stores scrobble data and metadata
   - **CanonFodder App with Airflow**: Runs the data pipeline and provides workflow orchestration
   - **Adminer**: Web interface for database management

   Access the services:
   - Airflow UI: http://localhost:8080 (username: admin, password: admin)
   - Adminer: http://localhost:8081 (server: db, username: canon, password: canon, database: canonfodder)

   The Docker setup includes:
   - Automatic database initialization
   - Airflow DAG for weekly data pipeline runs
   - Volume mapping for persistent data storage

   Option 2: Manual setup
   # Create a virtual environment
   ```shell
   python -m venv .venv
   ```

   # Activate the virtual environment
   # On Windows:
   ```shell
   .venv\Scripts\activate
   ```

   # On Unix/MacOS:
   ```shell
   source .venv/bin/activate
   ```

   # Install dependencies
   ```shell
   pip install -r requirements.txt
   ```

   # For Airflow compatibility (resolves dependency conflicts)
   ```shell
   pip install -r requirements-airflow.txt
   ```

   # or for development installation (recommended for contributors)
   ```shell
   pip install -e .
   ```

   # For development with Airflow compatibility
   ```shell
   pip install -e ".[airflow]"
   ```

   > **Note on Dependency Conflicts**: CanonFodder uses SQLAlchemy 2.0 for its ORM models, while Apache Airflow 3.0.1 requires SQLAlchemy 1.4. This creates a dependency conflict when installing both packages together. The `requirements-airflow.txt` file and the `[airflow]` extra in setup.py are provided to help resolve these conflicts. When working with Airflow, some core CanonFodder database functionality may be limited.

   > **Note on Platform-Specific Dependencies**: CanonFodder uses platform-specific markers in requirements.txt to handle dependencies that are only needed on certain operating systems (e.g., windows-curses for Windows). See [Platform-Specific Dependencies](docs/platform_specific_dependencies.md) for details.

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
python main.py
```

### Interactive Development

The repository includes example Parquet files for quick exploration:

1. **Data Profiling**
   ```shell
   python dev\profile.py
   ```

2. **Artist Canonization Exploration**
   ```shell
   python dev\canon.py
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

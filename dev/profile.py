"""
Interactive data‑profiling helper for CanonFodder
-------------------------------------------------
Run it as Jupyter-style cells
-------------------------------------------------
Per 2025‑05‑31
-------------------------------------------------
* **User‑country timeline stores ISO-2 country codes.
* **Artist‑country lookup first hits the local *ArtistInfo* table or
  `PQ/artist_info.parquet`**, then falls back to MusicBrainz if needed, and rewrites the
  cache + parquet on the fly.
* Adds a fast, vectorised join that assigns a ``UserCountry`` column to every
  scrobble by interval matching against the timeline.
* Keeps all guard‑rails (no overlaps, sensible dates) and rewrites
  ``PQ/uc.parquet`` automatically.
Performance Notes:
-----------------
* Use the `--no-interactive` flag to disable interactive visualizations for faster execution.
* When running in non-interactive mode, visualizations will be saved to files instead of
  being displayed in interactive windows.
* This can significantly improve performance, especially when running the script in a console
  or in environments where interactive display is not needed.
"""
# %%
from __future__ import annotations
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from DB import SessionLocal
from DB.models import (
    ArtistInfo,
    ArtistVariantsCanonized,
)
import argparse
from branca.colormap import LinearColormap, StepColormap
import calendar
from corefunc.data_cleaning import clean_artist_info_table
import folium
from folium import plugins as folium_plugins
from helpers import cli
from helpers import io
from helpers import stats
from HTTP import mbAPI
import json
import logging
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
import matplotlib.dates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

plt.ion()
import musicbrainzngs

musicbrainzngs.logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
import numpy as np
import os

os.environ["MPLBACKEND"] = "TkAgg"
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pypopulation
import re
from scipy.stats import gaussian_kde
import seaborn as sns
from sqlalchemy import select
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
import threading
import webbrowser

# %%
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("dev_profile")
logging.getLogger("musicbrainzngs.mbxml").setLevel(logging.WARNING)
# Get the database engine first
from DB import get_engine
engine = get_engine()

# Display database connection information
print(f"Database URL: {engine.url}")

# Initialize mbAPI with the engine to ensure it uses the correct database
mbAPI.init(engine=engine)

# Default to TkAgg backend, but this can be overridden by command line args
matplotlib.use("TkAgg")
os.environ["MPLBACKEND"] = "TkAgg"
log.addFilter(lambda rec: not rec.name.startswith("musicbrainzngs.mbxml"))


def parse_args() -> argparse.Namespace:
    pee = argparse.ArgumentParser(description="CanonFodder dev profiling helper")
    pee.add_argument("--no-interactive", action="store_true",
                     help="Disable interactive visualizations (faster execution)")
    sub = pee.add_subparsers(dest="cmd", help="Sub‑commands")
    sub.add_parser("country", help="Edit user‑country timeline interactively")
    sub.add_parser("cleanup-artists", help="Clean up the ArtistInfo table by removing duplicates and orphaned entries")
    # Use parse_known_args to ignore unrecognized arguments (like PyCharm's --mode, --host, --port)
    parsed_args, unknown = pee.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unrecognized arguments: {unknown}")
    return parsed_args


# Setting global variable to control interactive mode
INTERACTIVE_MODE = True

# Parsing command-line arguments and updating INTERACTIVE_MODE
args = None
# Checking if running as main script or in interactive console
if __name__ == "__main__" or "pydevconsole" in sys.argv[0] or "ipykernel" in sys.argv[0]:
    try:
        args = parse_args()
        if args and args.no_interactive:
            print("Running in non-interactive mode (visualizations will be saved to files)")
            matplotlib.use('Agg')  # Use non-interactive backend
            plt.ioff()  # Turn off interactive mode
            INTERACTIVE_MODE = False
    except Exception as e:
        print(f"Warning: Error parsing arguments: {e}")
        print("Continuing with default settings (interactive mode enabled)")


def show_or_save_plot(filename, dpi=100, description=None):
    """
    Either shows the matplotlib plot interactively or saves it to a file, depending on the mode.
    Args:
        filename: Name of the file to save the plot to (when in non-interactive mode)
        dpi: Resolution for the saved image
        description: Optional Markdown description to include with the plot when saving as HTML
    """
    pics_dir = PROJECT_ROOT / "pics"
    pics_dir.mkdir(exist_ok=True)
    filepath = pics_dir / filename

    # Check if running in a TTY (interactive terminal)
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    try:
        # If in interactive mode and in a TTY, show the plot
        # Otherwise, save it to a file
        if INTERACTIVE_MODE and is_tty:
            plt.show()
        else:
            # If not in a TTY, save the figure even if in interactive mode
            if INTERACTIVE_MODE and not is_tty:
                print(f"(no TTY – saving plot to file instead of displaying)")

            # Save the figure as an image
            plt.savefig(filepath, dpi=dpi)
            print(f"Plot saved to {filepath}")

            # If a description is provided, also save as HTML with the description
            if description:
                try:
                    from helpers.markdown import render_markdown

                    # Convert the plot to a base64 image for embedding in HTML
                    import io
                    import base64
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=dpi)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')

                    # Render the Markdown description to HTML
                    desc_html = render_markdown(description)

                    # Create HTML with both the description and the image
                    html_path = filepath.with_suffix('.html')
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .description {{ margin-bottom: 20px; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <div class="description">
        {desc_html}
    </div>
    <div class="image">
        <img src="data:image/png;base64,{img_str}" alt="Plot">
    </div>
</body>
</html>"""

                    with open(html_path, 'w', encoding='utf-8') as file_handle:
                        file_handle.write(html_content)

                    print(f"Plot with description saved to {html_path}")
                except ImportError:
                    # Define render_markdown as a simple function that returns the input text
                    def render_markdown(text):
                        return text
                    print("Showdown not available. HTML with description not generated.")
    except Exception as plot_error:
        print(f"Error showing/saving plot: {plot_error}")
        print(f"Attempting to save plot using alternative method...")
        try:
            plt.savefig(str(filepath), dpi=dpi, format='png')
            print(f"Plot saved to {filepath} using alternative method")
        except Exception as e2:
            print(f"Failed to save plot: {e2}")
    finally:
        # Close the figure if not in interactive mode or not in a TTY
        if not INTERACTIVE_MODE or not is_tty:
            plt.close()


def show_or_save_plotly(figura, filename, description=None, public=True):
    """
    Either shows the plotly figure interactively or saves it to a file, depending on the mode.
    Args:
        figura: The plotly figure to show or save
        filename: Name of the file to save the figure to (when in non-interactive mode)
        description: Optional Markdown description to include with the figure when saving as HTML
        public: If True, saves an additional copy in a public directory for teacher assessment
    """
    # Handle case where parameters might be passed in wrong order
    if hasattr(filename, 'layout') and isinstance(filename, go.Figure) and isinstance(figura, str):
        # Parameters are swapped, fix them
        figura, filename = filename, figura

    pics_dir = PROJECT_ROOT / "pics"
    pics_dir.mkdir(exist_ok=True)

    # Create a public directory for teacher assessment if it doesn't exist
    public_dir = PROJECT_ROOT / "public_visualizations"
    public_dir.mkdir(exist_ok=True)

    # Ensure filename is a string
    if not isinstance(filename, str):
        raise TypeError("filename must be a string, got {type(filename).__name__} instead")

    filepath = pics_dir / filename
    public_filepath = public_dir / filename
    if not INTERACTIVE_MODE:
        print(f"[PERFORMANCE OPTIMIZATION] Skipping actual figure rendering for {filename}")
        print(f"To view the figure, run the script with interactive mode enabled")
        with open(filepath.with_suffix('.txt'), 'w') as file_obj:
            file_obj.write(f"Figure would be saved here: {filepath}\n")
            file_obj.write("Running in non-interactive mode with performance optimizations enabled.\n")
            file_obj.write("To view the actual figure, run the script without the --no-interactive flag.\n")
        return

    # Check if running in a TTY (interactive terminal)
    is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not is_tty:
        print("(no TTY – assuming 'Y')")
        # When not in a TTY, just save to HTML without trying to open a browser
        html_path = filepath.with_suffix('.html')
        try:
            # Use a static configuration to reduce rendering complexity
            config = {
                'displayModeBar': False,  # Hide the modebar
                'responsive': True,  # Make the plot responsive
                'staticPlot': True,  # Make the plot static (no interactivity)
            }

            # If a description is provided, create HTML with both the description and the figure
            if description:
                try:
                    from helpers.markdown import render_markdown

                    # Render the Markdown description to HTML
                    desc_html = render_markdown(description)

                    # Get the figure HTML
                    figure_html = figura.to_html(
                        config=config,
                        include_plotlyjs='cdn',  # Use CDN for plotly.js (faster loading)
                        full_html=False  # Don't include HTML boilerplate
                    )

                    # Create HTML with both the description and the figure
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .description {{ margin-bottom: 20px; }}
        .assessment-header {{ 
            background-color: #f0f0f0; 
            padding: 10px; 
            margin-bottom: 20px; 
            border-left: 5px solid #007bff; 
        }}
    </style>
</head>
<body>
    <div class="description">
        {desc_html}
    </div>
    <div class="plotly-figure">
        {figure_html}
    </div>
</body>
</html>"""

                    # For public copies, add an assessment header
                    public_html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{filename} - Teacher Assessment</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .description {{ margin-bottom: 20px; }}
        .assessment-header {{ 
            background-color: #f0f0f0; 
            padding: 10px; 
            margin-bottom: 20px; 
            border-left: 5px solid #007bff; 
        }}
    </style>
</head>
<body>
    <div class="assessment-header">
        <h2>CanonFodder Visualization</h2>
        <p>This visualization was generated for assessment purposes.</p>
        <p>Filename: {filename}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="description">
        {desc_html}
    </div>
    <div class="plotly-figure">
        {figure_html}
    </div>
</body>
</html>"""

                    with open(html_path, 'w', encoding='utf-8') as file_out:
                        file_out.write(html_content)

                    print(f"Visualization with description saved to {html_path}")

                    # If public flag is set, also save to public directory
                    if public:
                        public_html_path = public_filepath.with_suffix('.html')
                        with open(public_html_path, 'w', encoding='utf-8') as public_file:
                            public_file.write(public_html_content)
                        print(f"Public copy for teacher assessment saved to {public_html_path}")
                except ImportError:
                    # Define render_markdown as a simple function that returns the input text
                    def render_markdown(text):
                        return text
                    print("Showdown not available. Using standard HTML output.")
                    # Fall back to standard HTML output
                    figura.write_html(
                        html_path,
                        config=config,
                        include_plotlyjs='cdn',
                        full_html=True
                    )
                    print(f"Visualization saved to {html_path}")

                    # If public flag is set, also save to public directory
                    if public:
                        public_html_path = public_filepath.with_suffix('.html')
                        figura.write_html(
                            public_html_path,
                            config=config,
                            include_plotlyjs='cdn',
                            full_html=True
                        )
                        print(f"Public copy for teacher assessment saved to {public_html_path}")
            else:
                # Save directly to HTML without opening browser
                figura.write_html(
                    html_path,
                    config=config,
                    include_plotlyjs='cdn',  # Use CDN for plotly.js (faster loading)
                    full_html=True
                )
                print(f"Visualization saved to {html_path}")

                # If public flag is set, also save to public directory
                if public:
                    public_html_path = public_filepath.with_suffix('.html')
                    figura.write_html(
                        public_html_path,
                        config=config,
                        include_plotlyjs='cdn',
                        full_html=True
                    )
                    print(f"Public copy for teacher assessment saved to {public_html_path}")
            return
        except Exception as html_error:
            print(f"Error saving HTML: {html_error}")
            # Fall through to other methods if HTML saving fails

    # Performance optimization for interactive mode
    try:
        # Configure a more efficient renderer
        pio.renderers.default = "browser"

        # Use a static configuration to reduce rendering complexity
        config = {
            'displayModeBar': False,  # Hide the modebar
            'responsive': True,  # Make the plot responsive
            'staticPlot': True,  # Make the plot static (no interactivity)
            'scrollZoom': False,  # Disable scroll zoom for better performance
            'showTips': False,  # Disable tips for better performance
        }

        # Progressive rendering approach:
        # 1. First create a simplified version of the figure for quick display
        print("Creating simplified version for quick display...")
        html_path = filepath.with_suffix('.html')

        # Create a simplified version of the figure
        simplified_fig = go.Figure()

        # Copy the layout from the original figure
        if hasattr(figura, 'layout'):
            for attr in dir(figura.layout):
                if not attr.startswith('_') and attr != 'template' and hasattr(simplified_fig.layout, attr):
                    try:
                        setattr(simplified_fig.layout, attr, getattr(figura.layout, attr))
                    except:
                        pass

        # Add a loading message
        simplified_fig.add_annotation(
            text="Loading visualization...",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20)
        )

        # Write the simplified figure to HTML with auto-refresh
        with open(html_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta http-equiv="refresh" content="2">
                <title>Loading Visualization...</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id="plot" style="width:100%;height:100vh;"></div>
                <script>
                    var data = {simplified_fig.to_json()};
                    Plotly.newPlot('plot', data.data, data.layout, {config});
                </script>
                <div style="text-align:center;margin-top:10px;">
                    <p>Preparing visualization, please wait...</p>
                </div>
            </body>
            </html>
            """)
        # Open the simplified version in the browser
        webbrowser.open(str(html_path))
        print("Simplified version displayed, now creating full visualization...")
        # 2. Then create the full version
        # Save the full figure to HTML
        figura.write_html(
            html_path,
            config=config,
            include_plotlyjs='cdn',  # Use CDN for plotly.js (faster loading)
            full_html=True
        )
        print(f"Full visualization saved to {html_path}")

        # If public flag is set, also save to public directory
        if public:
            public_html_path = public_filepath.with_suffix('.html')
            figura.write_html(
                public_html_path,
                config=config,
                include_plotlyjs='cdn',
                full_html=True
            )
            print(f"Public copy for teacher assessment saved to {public_html_path}")

        # The browser will auto-refresh to show the full version

    except Exception as render_error:
        print(f"Error with optimized rendering: {render_error}")
        print("Falling back to default show method...")
        try:
            # Try with a simpler approach
            print("Attempting to render with default settings...")
            figura.show()
        except Exception as e2:
            print(f"Error with fallback method: {e2}")
            print("Attempting to save as image instead...")
            try:
                # Last resort: try to save as a static image
                img_path = filepath.with_suffix('.png')
                figura.write_image(str(img_path), scale=0.5)  # Lower scale for better performance
                print(f"Saved as static image: {img_path}")

                # Open the image
                webbrowser.open(str(img_path))
            except Exception as e3:
                print(f"All rendering methods failed: {e3}")


def find_project_root():
    """Find the project root by looking for JSON and PQ directories."""
    if "__file__" in globals():
        # Try the standard approach first
        candidate = Path(__file__).resolve().parents[1]
        if (candidate / "JSON").exists() and (candidate / "PQ").exists():
            return candidate
    # If that fails, try the current directory and its parent
    current_dir = Path.cwd()
    if (current_dir / "JSON").exists() and (current_dir / "PQ").exists():
        return current_dir
    if (current_dir.parent / "JSON").exists() and (current_dir.parent / "PQ").exists():
        return current_dir.parent
    # If all else fails, use an absolute path
    return Path(r"C:\Users\jurda\PycharmProjects\CanonFodder")


PROJECT_ROOT = find_project_root()
JSON_DIR = PROJECT_ROOT / "JSON"
PQ_DIR = PROJECT_ROOT / "PQ"
UC_PARQUET = PQ_DIR / "uc.parquet"
AC_PARQUET = PQ_DIR / "artist_info.parquet"
AC_COLS = ["artist_name", "country", "mbid", "disambiguation_comment"]
PALETTES_FILE = JSON_DIR / "palettes.json"
SEPARATOR = "{"
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.options.display.float_format = "{: .2f}".format
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors = io.register_custom_palette("colorpalette_5", custom_palettes)
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette(custom_colors))


# %%
# =============================================================================
# Artist‑country helpers (parquet → DB → MusicBrainz)
# =============================================================================
def _append_to_parquet(path: Path, adatk: pd.DataFrame) -> None:
    if path.exists():
        old = pd.read_parquet(path)
        adatk = (pd.concat([old, adatk])
                 .drop_duplicates(subset="artist_name", keep="last"))
    adatk.to_parquet(path, compression="zstd", index=False)


def _country_for_series(series: pd.Series, cache: dict[str, str | None]) -> pd.Series:
    missing = [a for a in series.unique() if a and a not in cache]  # Skip empty artist names
    if missing:
        mb_results = {a: mbAPI.fetch_country(a) for a in missing}
        cache.update(mb_results)
        if mb_results:
            new_df = (pd.DataFrame
                      .from_dict(mb_results, orient="index", columns=["country"])
                      .assign(artist_name=lambda d: d.index))
            _append_to_parquet(AC_PARQUET, new_df)
    return series.map(cache)


def _df_from_db() -> pd.DataFrame:
    """Pull the entire ArtistInfo table into a DataFrame."""
    with SessionLocal() as sessio:
        rows = sessio.scalars(select(ArtistInfo)).all()
    if not rows:
        return pd.DataFrame(columns=AC_COLS)
    return pd.DataFrame(
        [
            {
                "artist_name": r.artist_name,
                "country": r.country,
                "mbid": r.mbid,
                "disambiguation_comment": r.disambiguation_comment,
            }
            for r in rows
        ],
        columns=AC_COLS
    )


df = _df_from_db()
# Check if the dataframe is unusually large
row_count = len(df)
if row_count > 1000000:
    print(f"WARNING: Artist info contains {row_count} rows, which is unusually large.")
    print("This may indicate duplicate or unnecessary entries in the database.")
    print("Consider running the clean_artist_info_table() function to remove duplicates and orphaned entries.")
    print("Example usage: cleaned, remaining = clean_artist_info_table()")

# Save to parquet with row count in the log message
df.to_parquet(AC_PARQUET, index=False, compression="zstd")
print(f"Saved {row_count} artist records to {AC_PARQUET}")

# Dump scrobble data
io.dump_parquet()

# %%
# -------------------------------------------------------------------------------------
#   Step 1: Load scrobbles parquet & deduplicate
# -------------------------------------------------------------------------------------
print("=" * 90)
print("Welcome to the CanonFodder data profiling workflow!")
print("We'll load your scrobble data, apply any previously saved artist name unifications,")
print("then explore on forward.")
print("=" * 90)
data, latest_filename = io.latest_parquet(return_df=True)
if data is None or data.empty:
    sys.exit("🚫  No scrobble data found – aborting EDA.")

# Create a mapping from original column names to the ones we want to use for analysis
# This preserves the original column names in the dataframe
column_mapping = {
    "artist_name": "Artist",
    "album_title": "Album",
    "play_time": "Datetime",
    "track_title": "Song",
    "artist_mbid": "MBID"
}

# Create a view of the data with renamed columns for analysis
# This doesn't modify the original dataframe's column names
data = data.rename(columns=column_mapping)

data.dropna(subset=["Datetime"], inplace=True)
data = data.drop_duplicates(["Artist", "Album", "Song", "Datetime"])
log.info("After dedup, %d rows remain.", len(data))

# %%
# -------------------------------------------------------------------------------------
#   Step 2: EDA - Why canonization of artist name variants matters?
# -------------------------------------------------------------------------------------
artist_counts = data["Artist"].value_counts()
top_artists = artist_counts.head(10)
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
custom_colors_10 = io.register_custom_palette("colorpalette_10", custom_palettes)
sns.set_style(style="whitegrid")
sns.set_palette(sns.color_palette(custom_colors_10))
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_artists.values,
    y=top_artists.index,
    palette=custom_colors_10[: len(top_artists)],
    hue=top_artists.index,
    legend=False,
)
plt.title("Top 10 Artists by Scrobbles", fontsize=16)
plt.xlabel("Scrobbles", fontsize=14)
plt.ylabel("Artist", fontsize=14)
plt.tight_layout()
print("Birds-eye stats for artist counts")
print(artist_counts.describe())
show_or_save_plot("top_artists.png")

# %%
# Canonization rationale snapshot
top3 = artist_counts.head(3).index
bohren_re = r'Bohren'
autechre_re = r'Autechre'
mask = (
        data['Artist'].isin(top3)
        | data['Artist'].str.contains(bohren_re, case=False, na=False)
        | data['Artist'].str.contains(autechre_re, case=False, na=False)
)
undesired = ['lustmord', 'sikora', 'sophie', 'orbital']
bad_pat = '|'.join(undesired)
mask &= ~data['Artist'].str.contains(bad_pat, case=False, na=False)
filtdata = data.loc[mask].copy()
bohren_canon = 'Bohren & der Club of Gore'
bohren_variants = {
    'Bohren & der Club of Gore': bohren_canon,
    'Bohren und der Club of Gore': bohren_canon,
}
filtdata['Artist_canon'] = filtdata['Artist'].replace(bohren_variants)
bohren_color = '#a24936'  # Chestnut color
autechre_color = '#16697a'  # Caribbean Current
default = '#ABB7C4'
bohren_artists = {
    'Bohren & der Club of Gore',
    'Bohren und der Club of Gore',
    'Bohren (canonized)',
}
autechre_artists = {'Autechre'}
highlight_artists = bohren_artists.union(autechre_artists)


def colour_for(artist_name: str) -> str:
    if artist_name in bohren_artists:
        return bohren_color
    elif artist_name in autechre_artists:
        return autechre_color
    else:
        return default


counts1 = (filtdata
           .groupby('Artist')
           .size()
           .reset_index(name='Scrobbles')
           .sort_values('Scrobbles', ascending=False))
text_colours = ['white' if colour_for(a) in [bohren_color, autechre_color] else 'black'
                for a in counts1['Artist']]
trace_left = go.Bar(
    x=counts1['Scrobbles'],
    y=counts1['Artist'],
    orientation='h',
    marker_color=[colour_for(a) for a in counts1['Artist']],
    text=counts1['Scrobbles'],
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color=text_colours),
    hovertemplate='%{y}: %{x}<extra></extra>'
)
wanted = [bohren_canon, 'Autechre', 'Radiohead', 'Secret Chiefs 3']
counts2 = (filtdata[filtdata['Artist_canon'].isin(wanted)]
           .groupby('Artist_canon')
           .size()
           .reset_index(name='Scrobbles')
           .replace({bohren_canon: 'Bohren (canonized)'}))
bohren_parts = (filtdata
                [filtdata['Artist'].isin(['Bohren & der Club of Gore',
                                          'Bohren und der Club of Gore'])]
                .groupby('Artist')
                .size())
autechre_cnt = counts2.loc[counts2['Artist_canon'] == 'Autechre', 'Scrobbles'].iat[0]
trace_bohren_amp = go.Bar(
    x=[bohren_parts['Bohren & der Club of Gore']],
    y=['Bohren (canonized)'],
    orientation='h',
    marker_color=bohren_color,
    text=[bohren_parts['Bohren & der Club of Gore']],
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='white'),
    hovertemplate='Bohren & der Club of Gore: %{x}<extra></extra>'
)
trace_bohren_und = go.Bar(
    x=[bohren_parts['Bohren und der Club of Gore']],
    y=['Bohren (canonized)'],
    orientation='h',
    marker_color=bohren_color,
    text=[bohren_parts['Bohren und der Club of Gore']],
    textposition='inside',
    insidetextanchor='middle',
    textfont=dict(color='white'),
    hovertemplate='Bohren und der Club of Gore: %{x}<extra></extra>'
)


def single_bar(artist_name):
    # Checking if the artist exists in counts2
    artist_rows = counts2.loc[counts2['Artist_canon'] == artist_name]
    if len(artist_rows) > 0:
        cnt = artist_rows['Scrobbles'].iat[0]
    else:
        # Using 0 as default for artist not found in filtered data
        cnt = 0
    # Determining text color based on background color for better contrast
    text_color = 'white' if colour_for(artist_name) in [bohren_color, autechre_color] else 'black'
    return go.Bar(
        x=[cnt],
        y=[artist_name],
        orientation='h',
        marker_color=colour_for(artist_name),
        text=[cnt],
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color=text_color),
        hovertemplate=f'{artist_name}: %{{x}}<extra></extra>'
    )


trace_autechre = single_bar('Autechre')
trace_radiohead = single_bar('Radiohead')
trace_sc3 = single_bar('Secret Chiefs 3')
print("Creating artist canonization visualization...")

# Create the figure with optimized settings
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Top artists before canonisation of Bohren variants",
                    "Top artists after canonisation of Bohren variants"),
    shared_xaxes=True,
    vertical_spacing=0.12
)
# Extreme performance optimization: Skip complex figure creation entirely in non-interactive mode
if not INTERACTIVE_MODE:
    print("[PERFORMANCE OPTIMIZATION] Skipping complex figure creation in non-interactive mode")
    # Just create a minimal figure with a title for the show_or_save_plotly function
    fig.update_layout(
        title="Artist canonization visualization (skipped for performance)",
        height=100,
        width=100
    )
else:
    print("Creating full figure in interactive mode with performance optimizations...")
    print("Adding traces to figure...")
    fig.add_trace(trace_left, row=1, col=1)
    second_subplot_traces = []
    if 'Bohren & der Club of Gore' in bohren_parts:
        second_subplot_traces.append(trace_bohren_amp)
    if 'Bohren und der Club of Gore' in bohren_parts:
        second_subplot_traces.append(trace_bohren_und)
    for artist_trace in [trace_autechre, trace_radiohead, trace_sc3]:
        if hasattr(artist_trace, 'x') and len(artist_trace.x) > 0 and artist_trace.x[0] > 0:
            second_subplot_traces.append(artist_trace)
    if second_subplot_traces:
        fig.add_traces(
            second_subplot_traces,
            rows=[2] * len(second_subplot_traces),
            cols=[1] * len(second_subplot_traces)
        )
    fig.update_xaxes(showticklabels=False, ticks="", row=1, col=1)
    fig.update_xaxes(showticklabels=False, ticks="", row=2, col=1)
    fig.update_yaxes(autorange='reversed', ticklabelposition='outside left', row=1, col=1)
    fig.update_yaxes(autorange='reversed', ticklabelposition='outside left', row=2, col=1)
    print("Optimizing figure layout...")
    fig.update_layout(
        barmode='stack',
        showlegend=False,
        height=700,
        width=700,
        margin=dict(l=120, r=40, t=80, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # Additional performance optimizations
        hovermode=False,  # Disable hover effects for better performance for now
        dragmode=False,  # Disable drag mode for better performance for the moment
        uirevision=False  # Disable UI revision tracking for the time being
    )
    bohren_total = bohren_parts.sum()
    if bohren_total > 0:
        fig.add_annotation(
            x=bohren_total, y='Bohren (canonized)',
            xref='x2', yref='y2',
            text=str(bohren_total),
            showarrow=False,
            font=dict(color='white', size=12),
            xanchor='left',
            xshift=10,
            bgcolor=bohren_color,
            bordercolor=bohren_color,
            borderwidth=1,
            borderpad=3,
            opacity=0.8
        )


# Adding a timeout mechanism to prevent hanging
def timeout_handler():
    print("\n[TIMEOUT] Visualization is taking too long to render.")
    print("The script will continue running, but the visualization may not complete.")
    print("Consider using a different approach if this happens frequently.")
    # Just to print a warning without exiting the process


# Setting a timeout for the visualization (30 seconds)
timer = threading.Timer(30.0, timeout_handler)
timer.start()

try:
    # Trying to show or save the plotly figure with a timeout
    print("Rendering visualization (timeout: 30 seconds)...")
    show_or_save_plotly(fig, "scrobbles.png")
    # To cancel the timeout if we get here
    timer.cancel()
    print("Visualization rendered successfully.")
except Exception as e:
    # Canceling the timeout if an exception occurs
    timer.cancel()
    print(f"Error rendering visualization: {e}")
    print("Consider using a different approach if this error persists.")

# %%
# -------------------------------------------------------------------------------------
#   Step 3: Apply artist canonicalisation from DB
# -------------------------------------------------------------------------------------
log.info("Fetching already canonised artist-name variants…")
with SessionLocal() as sess:
    canon_rows = (
        sess.query(ArtistVariantsCanonized)
        .filter(ArtistVariantsCanonized.to_link.is_(True))
        .all()
    )


def _split_variants(raw: str) -> list[str]:
    """
    Split the artist_variants field into its individual names.
    • Primary separator is “{”   (the new, unambiguous choice)
    • Strips whitespace and ignores empty items.
    """
    return [v.strip()
            for v in re.split(rf"[{re.escape(SEPARATOR)}]", raw)
            if v.strip()]


variant_to_canon: dict[str, str] = {}
for row in canon_rows:
    variants: list[str] = [v.strip() for v in row.artist_variants_text.split("{") if v.strip()]
    for variant in _split_variants(row.artist_variants_text):
        if variant and variant != row.canonical_name:
            variant_to_canon[variant] = row.canonical_name
if variant_to_canon:
    data["Artist"] = data["Artist"].replace(variant_to_canon)

# %%
# -------------------------------------------------------------------------------------
#   Step 4: Temporal enrichment
# -------------------------------------------------------------------------------------
adat = data.copy()
s = pd.to_datetime(adat["Datetime"], unit="s", utc=True)
s = s.dt.tz_convert(None)
data["Datetime"] = s
data.describe()
data["Year"] = data["Datetime"].dt.year
data["Month"] = data["Datetime"].dt.month
data["Day"] = data["Datetime"].dt.day
# Calculating total scrobbles per year
data.groupby("Year")["Song"].count().plot(kind="bar", figsize=(10, 4), rot=0,
                                          title="Total scrobbles per year")
plt.xlabel('')  # to remove the x-axis label 'Year' and reduce cognitive load

# Creating a time series by month for seasonal decomposition
monthly_scrobbles = data.groupby([data["Year"], data["Month"]])["Song"].count().reset_index()
monthly_scrobbles["Date"] = pd.to_datetime(monthly_scrobbles[["Year", "Month"]].assign(Day=1))
monthly_scrobbles = monthly_scrobbles.set_index("Date")["Song"]

# Applying seasonal decomposition if we have enough data points
if len(monthly_scrobbles) >= 12:  # to have at least one full year for seasonal decomposition
    decomp = seasonal_decompose(monthly_scrobbles, model='multiplicative', period=12)

    # Creating manual plots with more control
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 10))

    # Plotting each component
    decomp.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    ax1.grid(True)

    decomp.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    ax2.grid(True)

    decomp.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    ax3.grid(True)

    decomp.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    ax4.grid(True)

    # Explicitly specifying the years we want as ticks
    year_ticks = [2005, 2010, 2015, 2020, 2025]

    # Converting years to datetime objects for x-axis ticks
    tick_positions = [pd.Timestamp(f"{year}-01-01") for year in year_ticks]
    tick_labels = [str(year) for year in year_ticks]

    # Setting the same x-axis ticks for all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        # Only showing year in x-axis labels
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
        # Removing the "Date" label from x-axis
        ax.set_xlabel('')

    # Ensuring tick marks and labels are visible on all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', which='both', length=4, width=1, direction='out', bottom=True, labelbottom=True)

    # Getting the min and max dates from all subplots to set consistent x-axis limits
    all_dates = []
    for component in [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]:
        if not component.index.empty:
            all_dates.extend(component.index)

    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)

        # Set the same x-axis limits for all subplots to ensure alignment
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(min_date, max_date)

    plt.tight_layout()
    show_or_save_plot("decomposition.png")
else:
    print("Not enough data for seasonal decomposition. Need at least 12 months of data.")

# %%
# -------------------------------------------------------------------------------------
#   Step 5: Ridgeline plot on monthly scrobble variance
# -------------------------------------------------------------------------------------
# Calculate monthly scrobbles per year
monthly_data = (data
                .groupby(["Year", "Month"])["Song"]
                .count()
                .reset_index(name="Scrobbles"))

ordered_months = [calendar.month_name[i] for i in range(1, 13)]

anchor_hex = ["#16697a", "#dbf4a7", "#bf9f6f", "#a24936", "#e6beae"]
cmap = mcolors.LinearSegmentedColormap.from_list("scrobble_heat", anchor_hex, N=256)

x_min, x_max = 0, monthly_data["Scrobbles"].max() * 1.1
norm = mcolors.Normalize(vmin=x_min, vmax=x_max)

fig, ax = plt.subplots(figsize=(12, 8))

# vertical spacing between ridges
offset_step = 1.2
baseline = 0

for month_idx, month_name in enumerate(ordered_months, start=1):
    scrobs = monthly_data.loc[monthly_data["Month"] == month_idx, "Scrobbles"]
    if scrobs.empty:  # skip months with no data
        baseline += offset_step
        continue

    # KDE
    kde = gaussian_kde(scrobs)
    xgrid = np.linspace(x_min, x_max, 500)
    ygrid = kde(xgrid)
    ygrid = ygrid / ygrid.max()  # same height scaling for every ridge

    # build many narrow quads, each coloured by its x-mid-point
    polys = []
    colours = []
    for i in range(len(xgrid) - 1):
        verts = [(xgrid[i], baseline),
                 (xgrid[i], ygrid[i] + baseline),
                 (xgrid[i + 1], ygrid[i + 1] + baseline),
                 (xgrid[i + 1], baseline)]
        polys.append(verts)
        x_mid = 0.5 * (xgrid[i] + xgrid[i + 1])
        colours.append(cmap(norm(x_mid)))

    coll = PolyCollection(polys, facecolors=colours, edgecolor='none', linewidth=0)
    ax.add_collection(coll)

    # optional black outline for readability
    ax.plot(xgrid, ygrid + baseline, color="black", linewidth=0.8)

    baseline += offset_step

ax.set_yticks(np.arange(0, offset_step * 12, offset_step))
ax.set_yticklabels(ordered_months)
ax.set_xlim(x_min, x_max)
ax.set_xlabel("Monthly scrobble count")
ax.set_title('December and January going strong with July lagging behind')

plt.tight_layout()
show_or_save_plot("monthly_variance_ridgeline.png")

# %%
# -------------------------------------------------------------------------------------
#   Step 6: Overall distribution of artist counts
# -------------------------------------------------------------------------------------
# Calculate the number of tracks each artist contributed to
artist_track_counts = data["Artist"].value_counts()

# Display statistics about the distribution
print("Bird's eye stats for the number of tracks each artist contributed to:")
artist_stats = artist_track_counts.describe()
print(artist_stats)

# Create a figure for the violin plot with swarm plot
plt.figure(figsize=(12, 8))

# Sort the data from lower to higher scrobble counts
artist_track_counts_sorted = artist_track_counts.sort_values()
artist_track_counts_df = pd.DataFrame({'counts': artist_track_counts_sorted.values})

# Create the violin plot with horizontal orientation
ax8 = plt.gca()
# Choose more visible colors from the palette for better contrast
with PALETTES_FILE.open("r", encoding="utf-8") as fh:
    custom_palettes = json.load(fh)["palettes"]
# Use colorpalette_5 which has more vibrant colors
custom_colors_viz = io.register_custom_palette("colorpalette_5", custom_palettes)

# Use a very light color for the violin plot background
sns.violinplot(x='counts', data=artist_track_counts_df, inner=None, saturation=0.1, ax=ax8,
               color="#f0f8ff")  # Using a very light blue color (aliceblue)

# Add strip plot with horizontal orientation and jitter for better visualization
# Using a darker color with higher alpha for better visibility on light background
sns.stripplot(x='counts', data=artist_track_counts_df, jitter=0.25, size=2, alpha=0.8, ax=ax8,
              color="#16697a")  # Using a dark teal color (Caribbean Current)

# Set log scale for x-axis to better visualize the distribution
plt.xscale('log')

# Add vertical lines for key statistics with annotations directly next to them
for i, (stat, value) in enumerate(zip(['mean', '25%', '50%', '75%', 'max'],
                                      [artist_stats['mean'], artist_stats['25%'],
                                       artist_stats['50%'], artist_stats['75%'], artist_stats['max']])):
    # Create the line with darker colors
    line_color = "#a24936" if stat == 'mean' else "#16697a"  # Chestnut for mean, Caribbean Current for others
    plt.axvline(x=value, linestyle='--', alpha=0.9, color=line_color)

    # Format the text with appropriate suffix based on value
    if stat == 'mean':
        # For mean (float value), don't add any suffix
        text = f"{stat}: {value:.2f}"
    elif value == 1:
        # For value = 1, add "scrobble" (singular)
        text = f"{stat}: {int(value)} scrobble"
    else:
        # For value > 1, add "scrobbles" (plural)
        text = f"{stat}: {int(value)} scrobbles"

    # Position the text at different y-positions to avoid overlap
    y_pos = 0.9 - (i * 0.1)  # Stagger the vertical positions

    # Add text annotation with midpoint crossing the vertical line
    plt.annotate(text,
                 xy=(value, 0),
                 xytext=(value, y_pos),  # Position text so its midpoint crosses the line
                 xycoords=('data', 'axes fraction'),
                 textcoords=('data', 'axes fraction'),
                 horizontalalignment='center',  # Center alignment makes midpoint cross the line
                 verticalalignment='center',
                 fontsize=10,
                 fontweight='bold',
                 color=line_color,  # Using the same darker color for text
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc='#f8f8ff',  # Very light background (ghostwhite)
                           ec=line_color,
                           alpha=0.95))

# Customize the plot
plt.title("Most artists only have a few listens in the dataset", fontsize=16)
plt.xlabel("Number of tracks (log scale)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
# Legend removed as annotations are now directly on the plot

plt.tight_layout()
show_or_save_plot("artist_track_distribution.png")

# %%
# -------------------------------------------------------------------------------------
#   Step 7: Find trusted companions
# -------------------------------------------------------------------------------------
# Filter data to only include years between 2006 and 2025
filtered_data = data[(data["Year"] >= 2006) & (data["Year"] <= 2025)]

# Group filtered data by Year and Artist, count songs for each combination
yearly_artist_counts = filtered_data.groupby(["Year", "Artist"])["Song"].count().reset_index(name="Plays")

# Get the list of unique years in the filtered data
unique_years = filtered_data["Year"].unique()
num_years = len(unique_years)

# Find artists that appear in every year within the filtered range
artist_year_counts = yearly_artist_counts.groupby("Artist")["Year"].nunique()
trusted_companions = artist_year_counts[artist_year_counts == num_years].index.tolist()

if trusted_companions:
    print(f"Found {len(trusted_companions)} artists that appear in every year between 2006-2025 of your scrobble data.")

    # Create a DataFrame with just the trusted companions
    trusted_df = yearly_artist_counts[yearly_artist_counts["Artist"].isin(trusted_companions)]


    # Calculate percentile rank for each artist within each year
    def add_percentile_ranks(group):
        group["Percentile"] = group["Plays"].rank(pct=True) * 100
        return group


    try:
        # Use include_groups=False to prevent the 'Year' column from being included in the result
        trusted_with_percentiles = trusted_df.groupby("Year").apply(add_percentile_ranks,
                                                                    include_groups=False).reset_index()
    except ValueError as e:
        print(f"Warning: Error in groupby operation: {e}")
        print("Attempting alternative approach...")
        # Alternative approach: first apply the function, then handle the index separately
        result = trusted_df.groupby("Year").apply(add_percentile_ranks)
        # Check if we have a MultiIndex
        if isinstance(result.index, pd.MultiIndex):
            # Extract the Year from the MultiIndex and add it as a column
            result = result.reset_index(level=0)  # This will extract just the 'Year' level
        trusted_with_percentiles = result

    # Find artists that are consistently above certain percentiles
    percentile_thresholds = [25, 50, 75, 90]
    consistent_artists = {}

    for percentile in percentile_thresholds:
        # For each artist, check if they're above the percentile in every year
        above_percentile = {}
        for artist in trusted_companions:
            artist_data = trusted_with_percentiles[trusted_with_percentiles["Artist"] == artist]
            if all(artist_data["Percentile"] >= percentile):
                above_percentile[artist] = artist_data["Percentile"].mean()

        consistent_artists[percentile] = above_percentile

    # Calculate standard deviation for each trusted companion
    artist_std_devs = {}
    for artist in trusted_companions:
        artist_data = trusted_with_percentiles[trusted_with_percentiles["Artist"] == artist]
        artist_std_devs[artist] = artist_data["Plays"].std()

    # Create a DataFrame with standard deviation information
    std_dev_df = pd.DataFrame({
        "Artist": list(artist_std_devs.keys()),
        "StdDev": list(artist_std_devs.values())
    }).sort_values("StdDev")

    # Display artists with lowest standard deviation (most consistent listening patterns)
    top_consistent = std_dev_df.head(10)

    # Create visualization for trusted companions with lowest standard deviation
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="StdDev",
        y="Artist",
        hue="Artist",
        data=top_consistent,
        palette=[f"#{color['hex']}" for color in custom_palettes[3]["colors"][:len(top_consistent)]],
        legend=False
    )

    # Add value annotations to the bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_width():.1f}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=10,
            color="black",
        )

    plt.title("There are only a couple of artists consistently listened to across the years", fontsize=16)
    plt.xlabel("Standard deviation of play count", fontsize=14)
    plt.ylabel("Artist", fontsize=14)
    plt.tight_layout()
    show_or_save_plot("trusted_companions_std_dev.png")

    # Create a heatmap showing percentile ranks across years for top consistent artists
    top_artists = top_consistent["Artist"].head(8).tolist()  # Limit to 8 for readability

    # Filter data for the top artists
    heatmap_data = trusted_with_percentiles[trusted_with_percentiles["Artist"].isin(top_artists)]

    # Pivot the data for the heatmap
    try:
        # Check if required columns exist
        required_columns = ["Artist", "Year", "Percentile"]
        missing_columns = [col for col in required_columns if col not in heatmap_data.columns]

        if missing_columns:
            print(f"Warning: Missing columns in heatmap_data: {missing_columns}")
            print("Available columns:", heatmap_data.columns.tolist())

            # If Year is missing but we have a MultiIndex with Year, extract it
            if "Year" in missing_columns and isinstance(heatmap_data.index, pd.MultiIndex):
                year_level = heatmap_data.index.names.index("Year") if "Year" in heatmap_data.index.names else None
                if year_level is not None:
                    print("Extracting Year from MultiIndex")
                    heatmap_data = heatmap_data.reset_index()

        heatmap_pivot = heatmap_data.pivot_table(
            index="Artist",
            columns="Year",
            values="Percentile",
            aggfunc="mean"
        )
    except KeyError as e:
        print(f"Error creating pivot table: {e}")
        print("Columns in heatmap_data:", heatmap_data.columns.tolist())
        print("Creating a simplified pivot table as a fallback")

        # Create a simplified version as a fallback
        artists = heatmap_data["Artist"].unique()
        years = sorted(heatmap_data["Year"].unique()) if "Year" in heatmap_data.columns else []

        if not years and isinstance(heatmap_data.index, pd.MultiIndex):
            # Try to extract years from the index if it's a MultiIndex
            for level_name in heatmap_data.index.names:
                if level_name == "Year":
                    years = sorted(heatmap_data.index.get_level_values("Year").unique())
                    heatmap_data = heatmap_data.reset_index()
                    break

        # If we still don't have years, create a dummy pivot table
        if not years:
            print("Could not extract years from the data, creating a dummy pivot table")
            heatmap_pivot = pd.DataFrame(index=artists)
        else:
            # Create the pivot table manually if needed
            if "Percentile" in heatmap_data.columns:
                heatmap_pivot = pd.pivot_table(
                    heatmap_data,
                    values="Percentile",
                    index="Artist",
                    columns="Year",
                    aggfunc="mean"
                )
            else:
                print("Percentile column not found, creating a dummy pivot table")
                heatmap_pivot = pd.DataFrame(index=artists)

    # Create the heatmap only if we have valid data
    if heatmap_pivot.empty:
        print("Warning: No data available for heatmap. Skipping heatmap creation.")
    else:
        try:
            plt.figure(figsize=(12, 8))
            # Create a custom colormap from the palette
            custom_colors_hex = [f"#{color['hex']}" for color in custom_palettes[3]["colors"]]
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors_hex)

            # Check if the pivot table has at least one row and one column
            if heatmap_pivot.shape[0] > 0 and heatmap_pivot.shape[1] > 0:
                sns.heatmap(
                    heatmap_pivot,
                    annot=True,
                    fmt=".1f",
                    cmap=custom_cmap,
                    linewidths=0.5,
                    cbar_kws={"label": "Percentile rank"}
                )

                plt.title("Percentile ranks of most consistent artists (2006-2025)", fontsize=16)
                plt.tight_layout()
                show_or_save_plot("trusted_companions_heatmap.png")
            else:
                print(f"Warning: Pivot table has invalid dimensions: {heatmap_pivot.shape}. Skipping heatmap creation.")
                # Create a simple message plot instead
                plt.text(0.5, 0.5, "Insufficient data for heatmap visualization",
                         horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.axis('off')
                plt.title("Data Visualization Error", fontsize=16)
                plt.tight_layout()
                show_or_save_plot("trusted_companions_heatmap_error.png")
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            # Create a simple error message plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Error creating heatmap: {e}",
                     horizontalalignment='center', verticalalignment='center', fontsize=12, wrap=True)
            plt.axis('off')
            plt.title("Visualization Error", fontsize=16)
            plt.tight_layout()
            show_or_save_plot("trusted_companions_heatmap_error.png")

    # Print summary of consistent artists above percentile thresholds
    print("\nArtists consistently above percentile thresholds across years 2006-2025:")
    for percentile, artists in consistent_artists.items():
        if artists:
            print(f"\nAbove {percentile}th percentile in every year between 2006-2025 ({len(artists)} artists):")
            for artist, avg_percentile in sorted(artists.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {artist} (avg percentile: {avg_percentile:.1f})")
else:
    print("No artists appear in every year between 2006-2025 of your scrobble data.")

# %%
# -------------------------------------------------------------------------------------
#   Step 8: Enrich with country via MusicBrainz & chart
# -------------------------------------------------------------------------------------
ac_cache = (
    pd.read_parquet(AC_PARQUET)
    .set_index("artist_name")["country"]
    .to_dict()
)
data["ArtistInfo"] = _country_for_series(data["Artist"], ac_cache)
country_count = data.ArtistInfo.value_counts().sort_values(ascending=False).to_frame()[:15]
country_count = country_count.rename(columns={"ArtistInfo": "count"})
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=country_count.index,
    y="count",
    data=country_count,
    palette="Spectral",
    hue=country_count.index,
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 9),
        textcoords="offset points",
        fontsize=10,
        color="black",
    )
ax.set_xticks(range(len(country_count)))
ax.set_xticklabels(country_count.index, rotation=45, ha="right", fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_title("Top 15 Countries", fontsize=16)
ax.set_xlabel("ArtistInfo", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
plt.tight_layout()
show_or_save_plot("top_countries.png")

# -------------------------------------------------------------------------------------
#   Step 8: Artist country world map
# -------------------------------------------------------------------------------------
print("\nCreating geoplot of artist countries...")
# Get all countries with their scrobble counts
all_country_counts = data.ArtistInfo.value_counts().to_dict()

# Note: We're using the built-in "YlOrRd" colormap for the choropleth
# This provides a good color gradient for visualizing the scrobble counts
# The custom palette loading is kept for reference but not currently used
with open("JSON/palettes.json", 'r', encoding='utf-8') as f:
    palette_data = json.load(f)
    palettes = palette_data["palettes"]

# Create a map centered at the average latitude and longitude of all countries
# Create a map with a more zoomed out view for world perspective
avg_lat, avg_lon = 20, 0  # Default center position for better world view
m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2, control_scale=True)

# Load the GeoJSON file
geojson_path = Path("JSON/countries.geojson")
if not geojson_path.exists():
    print(f"Error: GeoJSON file not found at {geojson_path}")
else:
    # Load the GeoJSON data to extract country coordinates
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Create a dictionary to store country centroids
    country_centroids = {}

    # Extract centroids for each country
    # Debug: Check if Norway and France exist in the GeoJSON data
    norway_feature = None
    france_feature = None

    for feature in geojson_data['features']:
        country_code = feature['properties'].get('ISO3166-1-Alpha-2')
        country_name = feature['properties'].get('ADMIN')

        # Debug: Store Norway and France features
        if country_code == 'NO':
            norway_feature = feature
        elif country_code == 'FR':
            france_feature = feature

        if country_code and country_code != '-99':  # Skip invalid country codes
            # Calculate centroid from geometry
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # First polygon
                lats = [coord[1] for coord in coords]
                lons = [coord[0] for coord in coords]
                centroid = [sum(lats) / len(lats), sum(lons) / len(lons)]
                country_centroids[country_code] = {
                    'centroid': centroid,
                    'name': country_name
                }
            elif feature['geometry']['type'] == 'MultiPolygon':
                # For MultiPolygon, use the first polygon's centroid
                coords = feature['geometry']['coordinates'][0][0]  # First polygon
                lats = [coord[1] for coord in coords]
                lons = [coord[0] for coord in coords]
                centroid = [sum(lats) / len(lats), sum(lons) / len(lons)]
                country_centroids[country_code] = {
                    'centroid': centroid,
                    'name': country_name
                }

    # Debug: Print Norway and France features
    print(f"Norway feature found: {norway_feature is not None}")
    print(f"France feature found: {france_feature is not None}")
    if norway_feature:
        print(f"Norway properties: {norway_feature['properties']}")
    if france_feature:
        print(f"France properties: {france_feature['properties']}")

    # Fix country codes for France and Norway
    for feature in geojson_data['features']:
        if feature['properties'].get('name') == 'France' and feature['properties'].get('ISO3166-1-Alpha-2') == '-99':
            print("Fixing ISO code for France")
            feature['properties']['ISO3166-1-Alpha-2'] = 'FR'
        elif feature['properties'].get('name') == 'Norway' and feature['properties'].get('ISO3166-1-Alpha-2') == '-99':
            print("Fixing ISO code for Norway")
            feature['properties']['ISO3166-1-Alpha-2'] = 'NO'

    # Add scrobble counts to GeoJSON features
    for feature in geojson_data['features']:
        country_code = feature['properties'].get('ISO3166-1-Alpha-2')
        if country_code in all_country_counts:
            # Add the scrobble count to the feature properties
            feature['properties']['scrobble_count'] = int(all_country_counts[country_code])
        else:
            # Set to 0 if no scrobbles for this country
            feature['properties']['scrobble_count'] = 0

    # Create a DataFrame for the choropleth
    choropleth_data = pd.DataFrame(
        list(all_country_counts.items()),
        columns=["country", "count"]
    )

    # Create a dictionary mapping country codes to their scrobble counts
    country_count_dict = dict(zip(choropleth_data["country"], choropleth_data["count"]))

    # Print min and max values to understand the data range
    if not choropleth_data.empty:
        min_val = choropleth_data["count"].min()
        max_val = choropleth_data["count"].max()
        print(f"Min count: {min_val}, Max count: {max_val}")

        # Debug Norway and France counts
        norway_count = choropleth_data[choropleth_data["country"] == "NO"]["count"].values
        france_count = choropleth_data[choropleth_data["country"] == "FR"]["count"].values
        print(f"Norway count: {norway_count if len(norway_count) > 0 else 'Not found'}")
        print(f"France count: {france_count if len(france_count) > 0 else 'Not found'}")

        # Check if Norway and France are in the threshold scale
        if len(norway_count) > 0:
            norway_val = norway_count[0]
            print(f"Norway value: {norway_val}")
        if len(france_count) > 0:
            france_val = france_count[0]
            print(f"France value: {france_val}")

    # Add the choropleth layer with custom color scale
    # Use fixed threshold scale based on the specified intervals
    # 1-9, 10-99, 100-999, 1000-9999, 10000-99999
    threshold_scale = [0, 1, 10, 100, 1000, 10000, 100000]

    if not choropleth_data.empty:
        max_val = choropleth_data["count"].max()
        # If max value is larger than our highest threshold, extend the scale
        if max_val > 100000:
            # Add additional thresholds if needed
            current = 100000
            while current < max_val:
                current = current * 10
                threshold_scale.append(current)

        print(f"Using threshold scale: {threshold_scale}")

        # Check if Norway and France values are within the threshold scale
        if len(norway_count) > 0:
            norway_val = norway_count[0]
            if norway_val < threshold_scale[0] or norway_val > threshold_scale[-1]:
                print(f"Warning: Norway value {norway_val} is outside the threshold scale!")
            else:
                print(f"Norway value {norway_val} is within the threshold scale.")

        if len(france_count) > 0:
            france_val = france_count[0]
            if france_val < threshold_scale[0] or france_val > threshold_scale[-1]:
                print(f"Warning: France value {france_val} is outside the threshold scale!")
            else:
                print(f"France value {france_val} is within the threshold scale.")

        legend_name = f"Scrobble Count (0-{int(max_val)})"
    else:
        # Fallback if data is empty
        threshold_scale = [0, 1, 10, 100, 1000]
        legend_name = "Scrobble Count"

    # Create a custom colormap with logarithmic scale for better readability

    # Initialize norway_count and france_count with empty arrays to avoid undefined variable issues
    norway_count = np.array([])
    france_count = np.array([])

    # Remove the default legend by setting legend_name to None
    # Use a standard ColorBrewer color scheme
    # YlOrRd is a good choice for sequential data (yellow to red)

    choropleth = folium.Choropleth(
        geo_data=geojson_data,  # Use the modified geojson_data with scrobble_count
        name="Artist Countries",
        data=choropleth_data,
        columns=["country", "count"],
        key_on="feature.properties.ISO3166-1-Alpha-2",
        fill_color="YlOrRd",  # Using a standard ColorBrewer scheme
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=None,  # Remove default legend
        highlight=True,
        threshold_scale=threshold_scale,
    ).add_to(m)

    # Remove any colormap children (which create the default legend)
    # Note: Using _children is necessary here as it's part of folium's internal API
    # We're accessing a protected member, but it's the recommended way to modify the legend
    if hasattr(choropleth, '_children'):
        for child in choropleth._children.copy():
            if isinstance(choropleth._children[child], (StepColormap, LinearColormap)):
                choropleth._children.pop(child)

    # Create a custom legend with color squares for better readability
    if not choropleth_data.empty:
        min_val = max(1, choropleth_data["count"].min())  # Avoid log(0)
        max_val = choropleth_data["count"].max()

        # Use predefined intervals as requested: 1-9, 10-99, 100-999, 1000-9999, 10000-99999
        # Define the intervals (5 ranges = 6 boundary values)
        tick_values = [1, 10, 100, 1000, 10000, 100000]

        # Define 5 colors for the legend (one for each range)
        # Using colors from the YlOrRd ColorBrewer scheme to match the choropleth
        colors = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']

        if True:  # Always execute this block

            # Create a custom legend HTML with color squares
            legend_html = '''
            <div id="custom-legend" style="
                position: fixed;
                bottom: 50px;
                left: 10px;
                z-index: 1000;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid grey;
                font-family: Arial, sans-serif;
                font-size: 12px;
                ">
                <div style="margin-bottom: 5px;"><strong>{}</strong></div>
            '''.format(legend_name)

            # Add color squares with specific ranges
            ranges = ["1-9", "10-99", "100-999", "1000-9999", "10000-99999"]
            for i, (color, range_text) in enumerate(zip(colors, ranges)):
                legend_html += '''
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <div style="background-color: {}; width: 15px; height: 15px; margin-right: 5px;"></div>
                    <span>{}</span>
                </div>
                '''.format(color, range_text)

            legend_html += '</div>'

            # Add the custom legend to the map
            # Using get_root().html is a valid approach in folium, even if IDE shows it as unresolved
            root = m.get_root()
            if hasattr(root, 'html'):
                root.html.add_child(folium.Element(legend_html))
            else:
                print("Warning: Map root does not have 'html' attribute, legend not added")

            # We don't need to create a separate colormap since we're using a custom HTML legend
            # and the choropleth already has its coloring defined

    # Add tooltips to the choropleth layer
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['name', 'ISO3166-1-Alpha-2', 'scrobble_count'],
            aliases=['Country:', 'Code:', 'Scrobble count:'],
            style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
        )
    )

    folium.LayerControl().add_to(m)
    marker_cluster = folium_plugins.MarkerCluster(name="Top Artist Countries").add_to(m)
    for country, count in all_country_counts.items():
        if country and country != "None" and len(country) == 2:
            country_artists = data[data.ArtistInfo == country]["Artist"].value_counts().head(5).to_dict()
            country_name = country_centroids.get(country, {}).get('name', country)
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; padding: 10px;">
                <h3 style="color: #16697a; margin-bottom: 10px;">{country_name} ({country})</h3>
                <p style="font-size: 16px; margin-bottom: 15px;"><b>{count}</b> scrobbles</p>
                <h4 style="color: #16697a; margin-bottom: 5px;">Top artists:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
            """
            for artist, artist_count in country_artists.items():
                percentage = (artist_count / count) * 100
                popup_content += f'<li style="margin-bottom: 5px;"><b>{artist}</b>: {artist_count} scrobbles ({percentage:.1f}%)</li>'
            popup_content += """
                </ul>
            </div>
            """
            # Add marker if we have coordinates for this country
            if country in country_centroids:
                # Use different icon colors based on scrobble count
                if count > 10000:
                    icon_color = "red"
                elif count > 5000:
                    icon_color = "orange"
                elif count > 1000:
                    icon_color = "green"
                else:
                    icon_color = "blue"
                folium.Marker(
                    location=country_centroids[country]['centroid'],
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"{country_name} ({country}): {count} scrobbles",
                    icon=folium.Icon(color=icon_color, icon="music", prefix="fa"),
                ).add_to(marker_cluster)
    # Add a fullscreen control
    folium_plugins.Fullscreen().add_to(m)
    # Add a search control to find countries
    folium_plugins.Search(
        layer=choropleth.geojson,
        geom_type="Polygon",
        placeholder="Search for a country",
        collapsed=True,
        search_label="name",
    ).add_to(m)
    # Adding a measure control for distance measurement
    folium_plugins.MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    # Save the map
    map_path = Path("pics/artist_countries_map.html")
    m.save(str(map_path))
    print(f"Geoplot saved to {map_path}")
    # Display the map if in interactive mode
    if INTERACTIVE_MODE:
        print("Opening map in browser...")
        webbrowser.open(str(map_path))

# %%
# -------------------------------------------------------------------------------------
#   Step 9: Country population vs. scrobble count analysis
# -------------------------------------------------------------------------------------
print("\nAnalyzing relationship between country population and scrobble counts...")

# Get all countries with their scrobble counts
country_scrobble_counts = data.ArtistInfo.value_counts().to_dict()
try:
    print("Using pypopulation library for country population data")
except ImportError:
    print("The pypopulation library is not installed. Please install it using:")
    print("pip install pypopulation")
    raise RuntimeError("Failed to import pypopulation. This step requires the pypopulation library.")

# Create a dictionary mapping country codes to their population
population_data = {}

# Get population data for all countries in the scrobble data
print("Fetching population data for all countries...")
for country_code in country_scrobble_counts.keys():
    if country_code and country_code != "None" and len(country_code) == 2:
        try:
            # Get population for this country using pypopulation
            population = pypopulation.get_population(country_code)
            if population:
                population_data[country_code] = population
                print(f"  {country_code}: {population:,}")
            else:
                print(f"  No population data found for {country_code}")
        except Exception as e:
            print(f"  Error fetching population for {country_code}: {e}")

# Check if we have population data
if not population_data:
    raise RuntimeError("Failed to fetch population data for any country. This step requires population data.")

# Print some debug information about the population_data
print(f"Population data type: {type(population_data)}")
print(f"Population data has {len(population_data)} entries")
if population_data:
    # Print a sample of the data
    sample_key = next(iter(population_data))
    print(f"Sample key: {sample_key}, type: {type(sample_key)}")
    print(f"Sample value: {population_data[sample_key]}, type: {type(population_data[sample_key])}")

# Create a dataframe with country codes, scrobble counts, and population
country_data = []
for country_code, scrobble_count in country_scrobble_counts.items():
    if country_code and country_code != "None" and len(country_code) == 2:
        # Try to get population data for this country
        population = None

        # Direct lookup
        if country_code in population_data:
            population = population_data[country_code]
        else:
            # Try to find the country code in the keys (for complex keys)
            for key, value in population_data.items():
                if isinstance(key, tuple) and len(key) > 0 and key[0] == country_code:
                    population = value
                    break

        if population is not None:
            # Ensure population is a number
            try:
                population = float(population)
                country_data.append({
                    "country_code": country_code,
                    "scrobble_count": scrobble_count,
                    "population": population
                })
            except (ValueError, TypeError) as e:
                print(f"Error converting population for {country_code}: {e}, value: {population}")

# Create the dataframe
if country_data:
    country_pop_df = pd.DataFrame(country_data)
    print(f"Created DataFrame with {len(country_pop_df)} rows and columns: {country_pop_df.columns.tolist()}")
else:
    # If no data was processed successfully, create an empty DataFrame with the expected columns
    print("Warning: No country data could be processed. Creating empty DataFrame.")
    country_pop_df = pd.DataFrame(columns=["country_code", "scrobble_count", "population"])

# Add country names from the c.parquet file if available
c_parquet_path = Path("PQ/c.parquet")
if c_parquet_path.exists():
    country_codes_df = pd.read_parquet(c_parquet_path)
    # Create a dictionary mapping ISO-2 codes to English names
    country_names = dict(zip(country_codes_df["ISO-2"], country_codes_df["en_name"]))
    # Add country names to the dataframe
    country_pop_df["country_name"] = country_pop_df["country_code"].map(country_names)

# Calculate per capita scrobble count (scrobbles per million people)
country_pop_df["scrobbles_per_million"] = country_pop_df["scrobble_count"] / (country_pop_df["population"] / 1000000)

# Print summary statistics
print(f"\nAnalyzed {len(country_pop_df)} countries with both scrobble and population data")
print("\nTop 10 countries by absolute scrobble count:")
print(country_pop_df.sort_values("scrobble_count", ascending=False).head(10)[
          ["country_code", "country_name", "scrobble_count", "population"]])

print("\nTop 10 countries by scrobbles per million population:")
print(country_pop_df.sort_values("scrobbles_per_million", ascending=False).head(10)[
          ["country_code", "country_name", "scrobbles_per_million", "scrobble_count"]])

# Create a scatter plot of population vs. scrobble count
plt.figure(figsize=(12, 8))

# Use a custom palette from JSON/palettes.json
custom_colors_pop = io.register_custom_palette("caribbean_current_shades_13_l2d", custom_palettes)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors_pop)

scatter = plt.scatter(
    country_pop_df["population"],
    country_pop_df["scrobble_count"],
    alpha=0.7,
    s=100,  # Marker size
    c=country_pop_df["scrobbles_per_million"],  # Color by scrobbles per million
    cmap=custom_cmap
)

# Add a colorbar to show the scrobbles per million scale
cbar = plt.colorbar(scatter)
cbar.set_label("Scrobbles per Million Population", fontsize=12)

# Add country labels for the top N countries by scrobble count
top_n = 25  # Increased from 15 to show more countries
for _, row in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n).iterrows():
    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8
    )

# Add labels for top countries by scrobbles per million (which might not be in the top by absolute count)
top_per_capita = 10
for _, row in country_pop_df.sort_values("scrobbles_per_million", ascending=False).head(top_per_capita).iterrows():
    # Skip if already labeled in the top_n by scrobble count
    if row["country_code"] in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n)[
        "country_code"].values:
        continue
    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(5, -10),  # Offset downward to avoid overlap with other labels
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        color="darkgreen"  # Different color to distinguish from top by count
    )
# Add labels for countries with large populations (which might be interesting data points)
top_population = 5
for _, row in country_pop_df.sort_values("population", ascending=False).head(top_population).iterrows():
    # Skip if already labeled in previous groups
    if (row["country_code"] in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n)[
        "country_code"].values or
            row["country_code"] in
            country_pop_df.sort_values("scrobbles_per_million", ascending=False).head(top_per_capita)[
                "country_code"].values):
        continue

    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(-15, 10),  # Offset to the left and up to avoid overlap
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        color="navy"  # Different color for population-based labels
    )

# Add a trend line
z = np.polyfit(country_pop_df["population"], country_pop_df["scrobble_count"], 1)
p = np.poly1d(z)
plt.plot(
    country_pop_df["population"],
    p(country_pop_df["population"]),
    "r--",
    alpha=0.7,
    label=f"Trend line (y = {z[0]:.2e}x + {z[1]:.2f})"
)
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8,
                          label=f'Top {top_n} by scrobble count'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=8,
                          label=f'Top {top_per_capita} by scrobbles per million'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=8,
                          label=f'Top {top_population} by population'),
                   Line2D([0], [0], color='r', linestyle='--', label=f"Trend line (y = {z[0]:.2e}x + {z[1]:.2f})")]

# Calculate correlation coefficient
corr = country_pop_df["population"].corr(country_pop_df["scrobble_count"])
plt.title(f"Country population vs. scrobble count (Correlation: {corr:.2f})", fontsize=16)
plt.xlabel("Population", fontsize=14)
plt.ylabel("Scrobble count", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(handles=legend_elements, loc='upper left', fontsize=9)
plt.tight_layout()
show_or_save_plot("population_vs_scrobbles.png")

# Create a second plot with log scales for better visualization
plt.figure(figsize=(12, 8))

# Use the same custom palette from JSON/palettes.json
scatter = plt.scatter(
    country_pop_df["population"],
    country_pop_df["scrobble_count"],
    alpha=0.7,
    s=100,
    c=country_pop_df["scrobbles_per_million"],
    cmap=custom_cmap  # Reuse the custom colormap from the first plot
)

# Add a colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Scrobbles per million population", fontsize=12)

# Add country labels for the top N countries by scrobble count
for _, row in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n).iterrows():
    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        alpha=0.8
    )

# Add labels for top countries by scrobbles per million (which might not be in the top by absolute count)
for _, row in country_pop_df.sort_values("scrobbles_per_million", ascending=False).head(top_per_capita).iterrows():
    # Skip if already labeled in the top_n by scrobble count
    if row["country_code"] in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n)[
        "country_code"].values:
        continue

    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(5, -10),  # Offset downward to avoid overlap with other labels
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        color="darkgreen"  # Different color to distinguish from top by count
    )

# Add labels for countries with large populations (which might be interesting data points)
for _, row in country_pop_df.sort_values("population", ascending=False).head(top_population).iterrows():
    # Skip if already labeled in previous groups
    if (row["country_code"] in country_pop_df.sort_values("scrobble_count", ascending=False).head(top_n)[
        "country_code"].values or
            row["country_code"] in
            country_pop_df.sort_values("scrobbles_per_million", ascending=False).head(top_per_capita)[
                "country_code"].values):
        continue

    country_label = row["country_code"]
    if "country_name" in row and not pd.isna(row["country_name"]):
        country_label = row["country_name"]
    plt.annotate(
        country_label,
        (row["population"], row["scrobble_count"]),
        xytext=(-15, 10),  # Offset to the left and up to avoid overlap
        textcoords="offset points",
        fontsize=9,
        alpha=0.8,
        color="navy"  # Different color for population-based labels
    )

# Set log scales for both axes
plt.xscale("log")
plt.yscale("log")

# Add a legend for the different label categories with more space for readability
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8,
           label=f'Top {top_n} by scrobble count'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', markersize=8,
           label=f'Top {top_per_capita} by scrobbles per million'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=8,
           label=f'Top {top_population} by population')
]

plt.title(f"Iceland boasts an impressive scrobble count per population", fontsize=16)
plt.xlabel("Population (log scale)", fontsize=14)
plt.ylabel("Scrobble count (log scale)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)

# Add the legend with improved readability
# Use bbox_to_anchor to position the legend outside the plot area
# Increase fontsize and add padding with borderpad
# Legend removed as per requirements
plt.tight_layout()
show_or_save_plot("population_vs_scrobbles_log.png")

# %%
# -------------------------------------------------------------------------------------
#   Step 10: User-country analytics
# -------------------------------------------------------------------------------------
if UC_PARQUET.exists():
    use = cli.choose_timeline()
    if use == "e":
        uc_df = cli.edit_country_timeline()
    elif use == "n":
        UC_PARQUET.unlink(missing_ok=True)
        uc_df = cli.edit_country_timeline()
    else:  # "y"
        uc_df = pd.read_parquet(UC_PARQUET)
else:
    uc_df = cli.edit_country_timeline()
data["UserCountry"] = stats.assign_user_country(data, uc_df)
user_country_count = data.UserCountry.value_counts().sort_values(ascending=False).to_frame()[:10]
user_country_count = user_country_count.rename(columns={"UserCountry": "count"})

# Load custom palette for better visualization of skewed data
with open("JSON/palettes.json", 'r', encoding='utf-8') as f:
    custom_palettes = json.load(f)["palettes"]
# Use caribbean_current_shades_13_d2l palette which orders colors from dark to light (more scrobbles = darker color)
custom_colors = io.register_custom_palette("caribbean_current_shades_13_d2l", custom_palettes)

plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=user_country_count.index,
    y="count",
    data=user_country_count,
    palette=custom_colors,
    hue=user_country_count.index,
)
ax.set_xticks(range(len(user_country_count)))
ax.set_xticklabels(user_country_count.index, rotation=45, ha="right", fontsize=12)
ax.grid(True, axis="y", linestyle="--", alpha=0.7)
ax.set_yscale('log')
for p in ax.patches:
    if p.get_height() > 10:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=10,
            color="black",
            fontweight="bold",
            bbox=dict(facecolor='white', alpha=0.7, pad=2)
        )
ax.set_title("User countries per scrobble count (log scale)", fontsize=16)
ax.set_xlabel("UserCountry", fontsize=14)
ax.set_ylabel("Count (log scale)", fontsize=14)
plt.tight_layout()
show_or_save_plot("user_countries.png")

# %%
# --- entry point ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Handle cleanup-artists command
    if args and hasattr(args, 'cmd') and args.cmd == "cleanup-artists":
        print("Running ArtistInfo table cleanup...")
        cleaned, remaining = clean_artist_info_table()
        if cleaned > 0:
            print(f"Cleanup successful! Removed {cleaned} records, {remaining} remain.")
            print("Updating artist_info.parquet file with cleaned data...")
            df = _df_from_db()
            df.to_parquet(AC_PARQUET, index=False, compression="zstd")
            print(f"Saved {len(df)} artist records to {AC_PARQUET}")
        else:
            print("No records were cleaned. The ArtistInfo table is already optimized.")
        sys.exit(0)

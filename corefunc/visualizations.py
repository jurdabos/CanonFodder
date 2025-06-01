"""
Visualization functions for CanonFodder.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
matplotlib.use("Agg")


def load_data() -> pd.DataFrame:
    """
    Load scrobble data from the scrobble.parquet file.
    Returns:
        pd.DataFrame: The loaded data
    """
    try:
        # Getting the project root directory
        project_root = Path(__file__).resolve().parent.parent
        # Looking specifically for scrobble.parquet in the PQ directory
        pq_dir = project_root / "PQ"
        if not pq_dir.exists():
            raise FileNotFoundError(f"PQ directory not found: {pq_dir}")
        scrobble_file = pq_dir / "scrobble.parquet"
        if not scrobble_file.exists():
            raise FileNotFoundError(f"Scrobble parquet file not found: {scrobble_file}")
        # Loading the data
        data = pd.read_parquet(scrobble_file)

        # Renaming columns to match the expected format
        if "artist_name" in data.columns:
            data = data.rename(columns={
                "artist_name": "Artist",
                "album_name": "Album",
                "play_time": "Datetime",
                "track_name": "Song",
                "artist_mbid": "MBID"
            })
        # Converting timestamp to datetime if needed
        if "Datetime" in data.columns and not pd.api.types.is_datetime64_dtype(data["Datetime"]):
            data["Datetime"] = pd.to_datetime(data["Datetime"])
        # Adding year, month, day columns
        if "Datetime" in data.columns:
            data["Year"] = data["Datetime"].dt.year
            data["Month"] = data["Datetime"].dt.month
            data["Day"] = data["Datetime"].dt.day
        logger.info(f"Data loaded successfully from {scrobble_file}")
        logger.info(f"Data shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def scrobbles_per_year(data: pd.DataFrame) -> Figure:
    """
    Create a bar chart of scrobbles per year.

    Args:
        data (pd.DataFrame): The scrobble data
    Returns:
        Figure: The matplotlib figure
    """
    try:
        # Grouping by year and count
        year_counts = data.groupby("Year")["Song"].count()
        # Creating the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Loading custom color palette if available
        try:
            project_root = Path(__file__).resolve().parent.parent
            palette_path = project_root / "JSON" / "palettes.json"
            if palette_path.exists():
                with open(palette_path, "r") as f:
                    palettes = json.load(f)["palettes"]
                if "colorpalette_5" in palettes:
                    colors = palettes["colorpalette_5"]
                    color = colors[0]  # to use the first color
                else:
                    color = "#0D3C45"  # default color
            else:
                color = "#0D3C45"  # Default color
        except Exception:
            color = "#0D3C45"  # default color
        # Plotting the data
        bars = ax.bar(year_counts.index, year_counts.values, color=color)
        # Adding value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10
            )
        # Customizing the plot
        ax.set_title("Total Scrobbles per Year", fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("Scrobbles", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        # Rotating x-axis labels if there are many years
        if len(year_counts) > 10:
            plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error creating scrobbles per year plot: {e}")
        raise

def scrobbles_per_month(data: pd.DataFrame) -> Figure:
    """
    Create a bar chart of average scrobbles per month.
    Args:
        data (pd.DataFrame): The scrobble data
    Returns:
        Figure: The matplotlib figure
    """
    try:
        # Group by month and count, then divide by number of years
        month_counts = data.groupby("Month")["Song"].count()
        num_years = data["Year"].nunique()
        avg_month_counts = month_counts / num_years

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Load custom color palette if available
        try:
            project_root = Path(__file__).resolve().parent.parent
            palette_path = project_root / "JSON" / "palettes.json"

            if palette_path.exists():
                with open(palette_path, "r") as f:
                    palettes = json.load(f)["palettes"]

                if "colorpalette_5" in palettes:
                    colors = palettes["colorpalette_5"]
                    color = colors[1]  # Use the second color
                else:
                    color = "#1A5E63"  # Default color
            else:
                color = "#1A5E63"  # Default color
        except Exception:
            color = "#1A5E63"  # Default color

        # Get month names
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        # Plot the data
        bars = ax.bar(month_names, avg_month_counts.values, color=color)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        # Customize the plot
        ax.set_title(f"Average Monthly Scrobbles ({num_years} years)", fontsize=16)
        ax.set_xlabel("Month", fontsize=14)
        ax.set_ylabel("Average Scrobbles", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating scrobbles per month plot: {e}")
        raise

def monthly_ridgeline(data: pd.DataFrame) -> Figure:
    """
    Create a ridgeline plot of monthly scrobbles.

    Args:
        data (pd.DataFrame): The scrobble data

    Returns:
        Figure: The matplotlib figure
    """
    try:
        # Prepare data for ridgeline plot
        # Group by year and month, count scrobbles
        monthly_counts = data.groupby(["Year", "Month"])["Song"].count().reset_index()
        monthly_counts.columns = ["Year", "Month", "Count"]

        # Create a column with month names
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        monthly_counts["MonthName"] = monthly_counts["Month"].apply(lambda x: month_names[x-1])

        # Sort by year and month
        monthly_counts = monthly_counts.sort_values(["Year", "Month"])

        # Get unique years
        years = sorted(monthly_counts["Year"].unique())

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Load custom color palette if available
        try:
            project_root = Path(__file__).resolve().parent.parent
            palette_path = project_root / "JSON" / "palettes.json"

            if palette_path.exists():
                with open(palette_path, "r") as f:
                    palettes = json.load(f)["palettes"]

                if "colorpalette_10" in palettes:
                    colors = palettes["colorpalette_10"]
                else:
                    colors = sns.color_palette("viridis", len(years))
            else:
                colors = sns.color_palette("viridis", len(years))
        except Exception:
            colors = sns.color_palette("viridis", len(years))

        # Plot each year as a separate ridge
        for i, year in enumerate(years):
            year_data = monthly_counts[monthly_counts["Year"] == year]

            # Plot the data
            ax.plot(year_data["Month"], year_data["Count"], color=colors[i % len(colors)], label=str(year))

            # Fill the area under the curve
            ax.fill_between(
                year_data["Month"],
                year_data["Count"],
                alpha=0.3,
                color=colors[i % len(colors)]
            )

        # Customize the plot
        ax.set_title("Monthly Scrobbles by Year (Ridgeline)", fontsize=16)
        ax.set_xlabel("Month", fontsize=14)
        ax.set_ylabel("Scrobbles", fontsize=14)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.legend(title="Year", loc="upper right")

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating monthly ridgeline plot: {e}")
        raise

def seasonal_decomposition(data: pd.DataFrame) -> Figure:
    """
    Create a seasonal decomposition plot of scrobbles.

    Args:
        data (pd.DataFrame): The scrobble data

    Returns:
        Figure: The matplotlib figure
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # Prepare data for seasonal decomposition
        # Group by year and month, count scrobbles
        monthly_counts = data.groupby(["Year", "Month"])["Song"].count().reset_index()

        # Create a datetime index
        monthly_counts["Date"] = pd.to_datetime(
            monthly_counts["Year"].astype(str) + "-" + 
            monthly_counts["Month"].astype(str) + "-01"
        )
        monthly_counts = monthly_counts.set_index("Date")

        # Sort by date
        monthly_counts = monthly_counts.sort_index()

        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            monthly_counts["Song"],
            model="multiplicative",
            period=12  # 12 months in a year
        )

        # Create the plot
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Plot observed data
        decomposition.observed.plot(ax=axes[0], color="#0D3C45")
        axes[0].set_ylabel("Observed")
        axes[0].set_title("Seasonal Decomposition of Scrobbles")

        # Plot trend
        decomposition.trend.plot(ax=axes[1], color="#1A5E63")
        axes[1].set_ylabel("Trend")

        # Plot seasonal
        decomposition.seasonal.plot(ax=axes[2], color="#C2A83E")
        axes[2].set_ylabel("Seasonal")

        # Plot residual
        decomposition.resid.plot(ax=axes[3], color="#9A3838")
        axes[3].set_ylabel("Residual")

        # Customize the plot
        for ax in axes:
            ax.grid(linestyle="--", alpha=0.3)

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating seasonal decomposition plot: {e}")
        raise

def top_artists_bar(data: pd.DataFrame, n: int = 10) -> Figure:
    """
    Create a bar chart of top N artists by scrobble count.

    Args:
        data (pd.DataFrame): The scrobble data
        n (int, optional): Number of top artists to show. Defaults to 10.

    Returns:
        Figure: The matplotlib figure
    """
    try:
        # Count scrobbles per artist
        artist_counts = data["Artist"].value_counts().head(n)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Load custom color palette if available
        try:
            project_root = Path(__file__).resolve().parent.parent
            palette_path = project_root / "JSON" / "palettes.json"

            if palette_path.exists():
                with open(palette_path, "r") as f:
                    palettes = json.load(f)["palettes"]

                if f"colorpalette_{n}" in palettes:
                    colors = palettes[f"colorpalette_{n}"]
                else:
                    colors = sns.color_palette("viridis", n)
            else:
                colors = sns.color_palette("viridis", n)
        except Exception:
            colors = sns.color_palette("viridis", n)

        # Plot the data
        bars = ax.barh(
            artist_counts.index[::-1],  # Reverse to have highest at the top
            artist_counts.values[::-1],
            color=colors
        )

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)}",
                ha="left",
                va="center",
                fontsize=10
            )

        # Customize the plot
        ax.set_title(f"Top {n} Artists by Scrobble Count", fontsize=16)
        ax.set_xlabel("Scrobbles", fontsize=14)
        ax.set_ylabel("Artist", fontsize=14)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating top artists bar plot: {e}")
        raise

def artist_distribution_violin(data: pd.DataFrame) -> Figure:
    """
    Create a violin plot of artist scrobble distribution.

    Args:
        data (pd.DataFrame): The scrobble data

    Returns:
        Figure: The matplotlib figure
    """
    try:
        # Count scrobbles per artist
        artist_counts = data["Artist"].value_counts().reset_index()
        artist_counts.columns = ["Artist", "Scrobbles"]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the data
        sns.violinplot(
            y=artist_counts["Scrobbles"],
            ax=ax,
            color="#0D3C45",
            inner="quartile"
        )

        # Add strip plot for individual points
        sns.stripplot(
            y=artist_counts["Scrobbles"],
            ax=ax,
            color="black",
            alpha=0.3,
            jitter=True,
            size=3
        )

        # Customize the plot
        ax.set_title("Distribution of Scrobbles per Artist", fontsize=16)
        ax.set_ylabel("Scrobbles", fontsize=14)
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add statistics as text
        stats_text = (
            f"Count: {len(artist_counts)}\n"
            f"Mean: {artist_counts['Scrobbles'].mean():.2f}\n"
            f"Median: {artist_counts['Scrobbles'].median():.2f}\n"
            f"Std Dev: {artist_counts['Scrobbles'].std():.2f}\n"
            f"Min: {artist_counts['Scrobbles'].min()}\n"
            f"Max: {artist_counts['Scrobbles'].max()}"
        )

        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()

        return fig

    except Exception as e:
        logger.error(f"Error creating artist distribution violin plot: {e}")
        raise

def find_trusted_companions(data: pd.DataFrame, percentile: int = 25) -> Tuple[pd.DataFrame, Figure]:
    """
    Find artists that appear in every year of the data and have low variance in scrobble count.

    Args:
        data (pd.DataFrame): The scrobble data
        percentile (int, optional): Percentile threshold for "trusted" status. Defaults to 25.

    Returns:
        Tuple[pd.DataFrame, Figure]: DataFrame of trusted companions and visualization figure
    """
    try:
        # Get all years in the data
        all_years = sorted(data["Year"].unique())

        # Count scrobbles per artist per year
        artist_year_counts = data.groupby(["Artist", "Year"])["Song"].count().reset_index()
        artist_year_counts.columns = ["Artist", "Year", "Scrobbles"]

        # Pivot to get artists as rows and years as columns
        pivot_df = artist_year_counts.pivot(index="Artist", columns="Year", values="Scrobbles")

        # Fill NaN with 0
        pivot_df = pivot_df.fillna(0)

        # Filter artists that appear in every year
        present_in_all_years = pivot_df[pivot_df > 0].count(axis=1) == len(all_years)
        artists_in_all_years = pivot_df[present_in_all_years]

        if artists_in_all_years.empty:
            logger.warning("No artists present in all years")
            return pd.DataFrame(), plt.figure()

        # Calculate coefficient of variation (std/mean) for each artist
        artists_in_all_years["mean"] = artists_in_all_years.mean(axis=1)
        artists_in_all_years["std"] = artists_in_all_years.std(axis=1)
        artists_in_all_years["cv"] = artists_in_all_years["std"] / artists_in_all_years["mean"]

        # Sort by coefficient of variation (ascending)
        sorted_artists = artists_in_all_years.sort_values("cv")

        # Select artists below the specified percentile of CV
        cv_threshold = np.percentile(sorted_artists["cv"], percentile)
        trusted_companions = sorted_artists[sorted_artists["cv"] <= cv_threshold]

        # Create a DataFrame with the results
        result_df = trusted_companions.reset_index()[["Artist", "mean", "std", "cv"]]
        result_df = result_df.sort_values("cv")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Load custom color palette if available
        try:
            project_root = Path(__file__).resolve().parent.parent
            palette_path = project_root / "JSON" / "palettes.json"

            if palette_path.exists():
                with open(palette_path, "r") as f:
                    palettes = json.load(f)["palettes"]

                if "colorpalette_5" in palettes:
                    colors = palettes["colorpalette_5"]
                    color = colors[2]  # Use the third color
                else:
                    color = "#C2A83E"  # Default color
            else:
                color = "#C2A83E"  # Default color
        except Exception:
            color = "#C2A83E"  # Default color

        # Plot the data (top 15 trusted companions)
        top_n = min(15, len(result_df))
        top_artists = result_df.head(top_n)

        bars = ax.barh(
            top_artists["Artist"][::-1],  # Reverse to have lowest CV at the top
            top_artists["mean"][::-1],
            color=color,
            alpha=0.7
        )

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}",
                ha="left",
                va="center",
                fontsize=10
            )

        # Add CV values as text
        for i, (_, row) in enumerate(top_artists[::-1].iterrows()):
            ax.text(
                5,
                i,
                f"CV: {row['cv']:.3f}",
                ha="left",
                va="center",
                fontsize=8,
                color="black"
            )

        # Customize the plot
        ax.set_title(f"Top {top_n} Trusted Companions (Artists Present in All Years)", fontsize=16)
        ax.set_xlabel("Average Scrobbles per Year", fontsize=14)
        ax.set_ylabel("Artist", fontsize=14)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()

        return result_df, fig

    except Exception as e:
        logger.error(f"Error finding trusted companions: {e}")
        raise

def save_figure(fig: Figure, filename: str) -> str:
    """
    Save a matplotlib figure to a file.

    Args:
        fig (Figure): The matplotlib figure
        filename (str): The filename to save to

    Returns:
        str: The full path to the saved file
    """
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parent.parent

        # Create the visualizations directory if it doesn't exist
        viz_dir = project_root / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Save the figure
        filepath = viz_dir / filename
        fig.savefig(filepath, dpi=100, bbox_inches="tight")

        logger.info(f"Figure saved to {filepath}")

        return str(filepath)

    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        raise

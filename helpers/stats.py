"""
Supplies small statistical helpers for feature selection and outlier checks.
"""
from DB import SessionLocal, engine
from DB.models import ArtistInfo
from .io import PQ_DIR
import logging
from HTTP import mbAPI
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix, classification_report
from sqlalchemy import select, text, func, inspect
from tabulate import tabulate
AC_PARQUET = PQ_DIR / "artist_info.parquet"
AC_COLS = ["artist_name", "country", "mbid", "disambiguation_comment"]
COUNTRY_MAPPING_PARQUET = PQ_DIR / "c.parquet"


def _load_country_mapping():
    """
    Load country code mapping from c.parquet file.
    Returns a dictionary mapping ISO-2 country codes to English country names.
    If the file doesn't exist, returns an empty dictionary.
    """
    if not COUNTRY_MAPPING_PARQUET.exists():
        print(f"Warning: Country mapping file not found: {COUNTRY_MAPPING_PARQUET}")
        return {}

    try:
        country_df = pd.read_parquet(COUNTRY_MAPPING_PARQUET)
        # Create a dictionary mapping ISO-2 codes to English names
        return dict(zip(country_df["ISO-2"], country_df["en_name"]))
    except Exception as e:
        print(f"Error loading country mapping: {str(e)}")
        return {}


def _df_from_db() -> pd.DataFrame:
    """Pull the entire artist_info table into a DataFrame."""
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


def _load_ac_cache() -> pd.DataFrame:
    """
    Return the artist-country cache as DataFrame.
    • If the Parquet is newer than the last DB update → read Parquet
    • otherwise
        – pull from DB
        – overwrite Parquet
    """
    if AC_PARQUET.exists():
        pq_mtime = AC_PARQUET.stat().st_mtime
        with SessionLocal() as session:
            db_mtime = session.scalar(
                select(func.max(ArtistInfo.id))  # monotonic surrogate pk
            ) or 0
        if pq_mtime > db_mtime:
            return pd.read_parquet(AC_PARQUET)
    dataf = _df_from_db()
    dataf.to_parquet(AC_PARQUET, index=False, compression="zstd")
    return dataf


def _upsert_artist_country(new_rows: list[dict]) -> None:
    """Insert *or* enrich existing rows in a backend-agnostic way."""
    if not new_rows:
        return
    with SessionLocal() as se:
        for r in new_rows:
            obj = (
                se.query(ArtistInfo)
                .filter_by(artist_name=r["artist_name"])
                .one_or_none()
            )
            if obj:
                if not obj.mbid and r["mbid"]:
                    obj.mbid = r["mbid"]
                if not obj.disambiguation_comment and r["disambiguation_comment"]:
                    obj.disambiguation_comment = r["disambiguation_comment"]
                if not obj.country and r["country"]:
                    obj.country = r["country"]
            else:
                se.add(ArtistInfo(**r))
        se.commit()


def artist_countries(series: pd.Series) -> pd.Series:
    """
    Vectorised ISO-country lookup with on-disk/DB cache;
    back-fills MBID + disambiguation when available.
    """
    cache_df = _load_ac_cache()
    cached = cache_df.set_index("artist_name").to_dict("index")
    missing = [a for a in series.unique() if a not in cached]
    new_rows = []
    for artist in missing:
        mb_res = mbAPI.search_artist(artist, limit=1)
        mb_row = mb_res[0] if mb_res else {}
        new_rows.append(
            dict(
                artist_name=artist,
                country=mb_row.get("country"),
                mbid=mb_row.get("id"),
                disambiguation_comment=mb_row.get("disambiguation"),
            )
        )
    _upsert_artist_country(new_rows)
    # Refreshing cache after insert
    cache_df = _load_ac_cache()
    cached = cache_df.set_index("artist_name").country.to_dict()
    return series.map(cached)


def assign_user_country(datafr, timeline):
    """
    Assign user country information to a dataframe based on a timeline.
    Parameters:
    -----------
    datafr : pandas.DataFrame
        The dataframe to assign country information to. Should contain a datetime column
        (named 'Datetime' or similar).
    timeline : pandas.DataFrame
        A dataframe containing country information with columns that should include
        either 'UserCountry', 'country', 'country_code', or some column containing 
        location-related keywords.
    Returns:
    --------
    pandas.Series
        A series containing the assigned country information.
    """
    if timeline is None:
        raise ValueError("Timeline dataframe cannot be None")
    if not isinstance(timeline, pd.DataFrame):
        raise ValueError(f"Timeline must be a pandas DataFrame, got {type(timeline)}")
    if timeline.empty:
        raise ValueError("Timeline dataframe is empty")
    timeline = timeline.copy()
    columns_list = list(timeline.columns)
    # Initialize country_col to None to avoid "referenced before assignment" warnings
    country_col = None
    # Handling the case where the timeline df has a "country_code" column but no "country" or "UserCountry" column
    if "country_code" in columns_list and "UserCountry" not in columns_list and "country" not in columns_list:
        # Loading country code mapping
        country_mapping = _load_country_mapping()
        if country_mapping:
            # Mapping country codes to full country names if mapping is available
            timeline["country"] = timeline["country_code"].map(country_mapping)
            # Fall back to the original code if mapping fails for some codes
            timeline["country"] = timeline["country"].fillna(timeline["country_code"])
            print("Mapped country codes to full country names using c.parquet")
        else:
            # If mapping is not available, just use the country code as is
            timeline["country"] = timeline["country_code"]
            print("Created 'country' column from 'country_code' for compatibility (no mapping available)")

    # Debug information about the timeline dataframe
    try:
        print(f"Timeline dataframe columns: {list(timeline.columns)}")
    except Exception as e:
        print(f"Error accessing timeline columns: {str(e)}")
        raise ValueError(f"Invalid timeline dataframe: {str(e)}")

    temp = datafr.copy()

    # Check if 'Datetime' column exists
    datetime_col = None
    if "Datetime" in temp.columns:
        datetime_col = "Datetime"
    else:
        # Look for columns that might contain datetime information
        datetime_candidates = [col for col in temp.columns if
                               any(keyword in col.lower() for keyword in
                                   ["datetime", "date", "time", "timestamp"])]
        if datetime_candidates:
            datetime_col = datetime_candidates[0]

    if not datetime_col:
        raise KeyError(f"No datetime column found in dataframe. Available columns: {list(temp.columns)}")

    try:
        temp["day"] = (
            pd.to_datetime(temp[datetime_col], unit="s", errors="coerce")
            .dt.normalize()
        )
        temp = temp.dropna(subset=["day"])
    except Exception as e:
        raise ValueError(f"Error converting '{datetime_col}' to datetime: {str(e)}. "
                         f"Make sure the column contains valid timestamp data.")

    # Make a copy of the timeline dataframe to avoid modifying the original
    timeline = timeline.copy()

    # Check which column to use for country information
    # Convert columns to list first to avoid potential KeyError
    columns_list = list(timeline.columns)

    # Print more detailed debug information
    print(f"Looking for country information in columns: {columns_list}")

    # First, check if "UserCountry" column already exists
    if "UserCountry" in columns_list:
        print("Found 'UserCountry' column")
        country_col = "UserCountry"
    # Then check for exact match with "country"
    elif "country" in columns_list:
        print("Found 'country' column")
        # Create a new "UserCountry" column for consistency
        timeline["UserCountry"] = timeline["country"]
        country_col = "UserCountry"
    # Check for "country_code" which is used in the uc.parquet file
    elif "country_code" in columns_list:
        print("Found 'country_code' column")
        # Load country code mapping
        country_mapping = _load_country_mapping()

        if country_mapping:
            # Map country codes to full country names if mapping is available
            timeline["UserCountry"] = timeline["country_code"].map(country_mapping)
            # Fall back to the original code if mapping fails for some codes
            timeline["UserCountry"] = timeline["UserCountry"].fillna(timeline["country_code"])
            print("Mapped country codes to full country names using c.parquet")
        else:
            # If mapping is not available, just use the country code as is
            timeline["UserCountry"] = timeline["country_code"]
            print("Using country_code as UserCountry (no mapping available)")
        country_col = "UserCountry"
    # Check for case-insensitive match with "country_code"
    elif any(col.lower() == "country_code" for col in columns_list):
        # Find the actual column name with case-insensitive match
        for col in columns_list:
            if col.lower() == "country_code":
                country_col_actual = col
                print(f"Found case-insensitive match for 'country_code': '{country_col_actual}'")

                # Load country code mapping
                country_mapping = _load_country_mapping()

                if country_mapping:
                    # Map country codes to full country names if mapping is available
                    timeline["UserCountry"] = timeline[country_col_actual].map(country_mapping)
                    # Fall back to the original code if mapping fails for some codes
                    timeline["UserCountry"] = timeline["UserCountry"].fillna(timeline[country_col_actual])
                    print("Mapped country codes to full country names using c.parquet")
                else:
                    # If mapping is not available, just use the country code as is
                    timeline["UserCountry"] = timeline[country_col_actual]
                    print(f"Using {country_col_actual} as UserCountry (no mapping available)")

                country_col = "UserCountry"
                break
    else:
        # If not, look for any column that contains "country" (case-insensitive)
        country_cols = [col for col in columns_list if "country" in col.lower()]

        if country_cols:
            # Use the first column that matches
            country_col_actual = country_cols[0]
            print(f"Found column containing 'country': '{country_col_actual}'")
            # Create a new "UserCountry" column for consistency
            timeline["UserCountry"] = timeline[country_col_actual]
            country_col = "UserCountry"
        else:
            # If no country column found, try a more flexible approach
            # Look for any column that might contain location information
            location_keywords = ["location", "region", "place", "nation", "code"]
            found_location_col = False
            for keyword in location_keywords:
                location_cols = [col for col in columns_list if keyword in col.lower()]
                if location_cols:
                    country_col_actual = location_cols[0]
                    print(f"Found column containing '{keyword}': '{country_col_actual}'")
                    # Create a new "UserCountry" column for consistency
                    timeline["UserCountry"] = timeline[country_col_actual]
                    country_col = "UserCountry"
                    found_location_col = True
                    break

            if not found_location_col:
                # If we still haven't found a suitable column, try using the first column that's not 'id',
                # 'start_date', or 'end_date'
                potential_cols = [col for col in columns_list if col not in ['id', 'start_date', 'end_date']]
                if potential_cols:
                    country_col_actual = potential_cols[0]
                    print(f"Using first non-date/id column as country column: '{country_col_actual}'")
                    timeline["UserCountry"] = timeline[country_col_actual]
                    country_col = "UserCountry"
                else:
                    # If we still haven't found a suitable column, raise an error with more information
                    raise KeyError(f"No country column found in timeline dataframe. Available columns: {columns_list}")

    # Ensure the country_col exists in the timeline dataframe
    if country_col not in timeline.columns:
        raise KeyError(
            f"Column '{country_col}' not found in timeline dataframe. Available columns: {list(timeline.columns)}")

    # Perform the merge_asof operation
    # Make a copy of the timeline dataframe to avoid modifying the original
    timeline_for_merge = timeline.copy()

    # Ensure the country_col is included in the merge
    if country_col == "UserCountry" and "country" in timeline.columns:
        # If we're looking for UserCountry but only have country, create UserCountry from country
        timeline_for_merge["UserCountry"] = timeline["country"]

    # Print debug information
    print(f"Timeline columns for merge: {list(timeline_for_merge.columns)}")
    print(f"Country column for merge: {country_col}")

    # Perform the merge_asof operation to find the most recent country entry
    out = pd.merge_asof(
        temp.sort_values("day"),
        timeline_for_merge[["start_date", country_col]],
        left_on="day",
        right_on="start_date",
        direction="backward",
    )

    # Check if end_date column exists in the timeline
    if "end_date" in timeline_for_merge.columns:
        # Create a temporary dataframe with end_date information
        end_dates = timeline_for_merge[["start_date", "end_date"]].copy()
        end_dates = end_dates.dropna(subset=["end_date"])  # Drop rows with no end_date

        if not end_dates.empty:
            # Merge the end_date information
            out = pd.merge_asof(
                out,
                end_dates,
                left_on="day",
                right_on="start_date",
                direction="backward",
                suffixes=("", "_end")
            )

            # Filter out rows where the day is after the end_date
            # Only apply this filter if end_date is not null
            mask = (out["end_date"].notna()) & (out["day"] > out["end_date"])
            out.loc[mask, country_col] = None

    # Print debug information about the output dataframe
    print(f"Output columns after merge: {list(out.columns)}")

    # Check if the country_col exists in the output dataframe
    if country_col not in out.columns:
        # Check if the column was renamed due to a naming conflict during merge
        if f"{country_col}_x" in out.columns and f"{country_col}_y" in out.columns:
            print(
                f"{country_col} was renamed to {country_col}_x and {country_col}_y during merge, using {country_col}_y")
            return out[f"{country_col}_y"]  # Use the value from the timeline dataframe
        # If UserCountry is not in the output but country is, use country instead
        elif country_col == "UserCountry" and "country" in out.columns:
            print("UserCountry not found in output, using country instead")
            return out["country"]
        else:
            raise KeyError(
                f"Column '{country_col}' not found in output dataframe after merge_asof."
                f"Available columns: {list(out.columns)}")

    # Use try-except to handle any potential KeyError when accessing the column
    try:
        return out[country_col]
    except KeyError:
        # Check if the column was renamed due to a naming conflict during merge
        if f"{country_col}_x" in out.columns and f"{country_col}_y" in out.columns:
            print(f"KeyError when accessing '{country_col}', using '{country_col}_y' instead")
            return out[f"{country_col}_y"]  # Use the value from the timeline dataframe
        # If UserCountry is not in the output but country is, use country instead
        elif country_col == "UserCountry" and "country" in out.columns:
            print(f"KeyError when accessing '{country_col}', falling back to 'country'")
            return out["country"]
        else:
            # Re-raise the error with more context
            raise KeyError(
                f"Column '{country_col}' not found in output dataframe. Available columns: {list(out.columns)}")


def cramers_v(x, y):
    """
    Returns Cramér’s V between two categorical pandas Series.
    """
    from scipy import stats
    contingency_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    return np.sqrt(phi2 / min(r - 1, k - 1))


def drop_high_corr_features(cm, threshold, var_table):
    """
    Identifies and returns highly correlated column pairs and the drop list.
    """
    high_corr_pairs = []
    features_to_drop = []
    variance_dict = dict(zip(var_table['features'], var_table['variances']))
    for _i_ in range(len(cm.columns)):
        for j in range(_i_):
            if abs(cm.iloc[_i_, j]) > threshold:
                high_corr_pairs.append((cm.columns[_i_], cm.columns[j]))
                feature_i = cm.columns[_i_]
                feature_j = cm.columns[j]
                if variance_dict[feature_i] < variance_dict[feature_j]:
                    features_to_drop.append(feature_i)
                else:
                    features_to_drop.append(feature_j)
    return high_corr_pairs, features_to_drop


def iterative_correlation_dropper(current_data, cutoff, varframe, min_features=8):
    """
    Iteratively drops correlated columns until `min_features` remain.
    Args:
        current_data: dataframe to prune
        cutoff: absolute correlation threshold
        varframe: dataframe with per-feature variances
        min_features: minimal column count to keep
    Returns:
        dataframe with reduced multicollinearity
    """
    all_dropped_features = []
    all_corr_pairs = []
    while len(current_data.columns) > min_features:
        # Calculating the correlation matrix
        corr_matrix = current_data.corr(method='spearman')
        # Masking to ignore redundant correlations
        mask = np.triu(np.ones(corr_matrix.shape), k=0)
        corr_matrix = corr_matrix.where(mask == 0)
        # Finding pairs with correlation above cutoff
        corr_pairs = corr_matrix.stack()
        vs_corr_pairs = corr_pairs[abs(corr_pairs) > cutoff].sort_values(ascending=False, key=abs)
        if vs_corr_pairs.empty:
            print(f"No more pairs above an absolute correlation of {cutoff}. Stopping.")
            break
        f1, f2 = vs_corr_pairs.index[0]
        if f1 not in current_data.columns or f2 not in current_data.columns:
            continue
        var_f1 = varframe.loc[varframe['features'] == f1, 'variances'].values[0]
        var_f2 = varframe.loc[varframe['features'] == f2, 'variances'].values[0]
        feature_to_drop = f1 if var_f1 > var_f2 else f2
        current_data = current_data.drop(columns=[feature_to_drop])
        all_dropped_features.append(feature_to_drop)
        all_corr_pairs.append((f1, f2))
        print(f"Dropped feature: {feature_to_drop} (Correlation: {vs_corr_pairs.iloc[0]:.3f})")
        # Stopping condition to retain a minimum number of features
        if len(current_data.columns) <= min_features:
            print(f"Reached the minimum number of features: {min_features}. Stopping.")
            break
    print(f"Final feature count: {len(current_data.columns)}")
    return current_data


def length_stats(input_text):
    parts = re.split(r"{", input_text)
    lens = [len(p.strip()) for p in parts if p.strip()]
    return pd.Series({
        "sig_len": sum(lens),
        "n_variants": len(lens),
        "avg_name_len": np.mean(lens),
        "max_name_len": np.max(lens),
        "var_len": np.std(lens)
    })


def show_misclassified(gs_df, model_pipe, X_matrix, extra_cols=None,
                       only_test=False, idx_test=None, top_n=20):
    """
    gs_df      : original DataFrame with 'variants' and target
    model_pipe : the fitted sklearn Pipeline
    X_matrix   : numeric matrix corresponding to gs_df
    extra_cols : list of additional column names to show (optional)
    only_test  : if True, show mis-classifications on test set only
    idx_test   : indices of rows that formed the held-out test set
    """
    extra_cols = extra_cols or []
    y_predicted = model_pipe.predict(X_matrix)
    y_prob1 = model_pipe.predict_proba(X_matrix)[:, 1]
    if only_test:
        mask = (gs_df.index.isin(idx_test)) & (gs_df["to_link"] != y_predicted)
    else:
        mask = gs_df["to_link"] != y_predicted
    mis = gs_df.loc[mask, ["variants", "to_link"] + extra_cols].copy()
    mis["predicted"] = y_predicted[mask]
    mis["prob_1"] = y_prob1[mask].round(3)
    print(f"\nMis-classified rows ({'test' if only_test else 'all'} set): "
          f"{len(mis)}\n")
    print(tabulate(mis.head(top_n), headers="keys",
                   tablefmt="pretty", maxcolwidths=70))


def missing_value_ratio(col):
    """
    Returns the percentage of missing values in a pandas Series.
    """
    return (col.isnull().sum() / len(col)) * 100


def show_cm_and_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )
    if title:
        print(f"\n{title}")
    print(tabulate(cm_df, headers="keys", tablefmt="pretty"))
    print(classification_report(y_true, y_pred, target_names=["no link", "link"]))


def top_n_artists_by_country(adatkeret, country, n=24):
    """
    Return the N most-played artists for one country,
    sorted by descending play-count.
    """
    (adatkeret
     .loc[adatkeret["Country"] == country, "Artist"]
     .value_counts()
     .head(n)
     )


def variance_testing(dframe, varthresh):
    """
    Applies sklearn.VarianceThreshold and returns (variance_df selected_cols).
    """
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=varthresh)
    _ = selector.fit_transform(dframe)
    variances = selector.variances_
    variance_df = pd.DataFrame({"features": dframe.columns, "variances": variances})
    variance_df = variance_df.sort_values(by="variances", ascending=False)
    selected_features = dframe.columns[selector.get_support(indices=True)]
    return variance_df, selected_features


def winsorization_outliers(df):
    """
    Prints and returns numeric outliers detected via 1st and 99th percentile.
    """
    out = []
    for n in df:
        q1 = np.percentile(df, 1)
        q3 = np.percentile(df, 99)
        if n > q3 or n < q1:
            out.append(n)
    return out


def get_db_statistics():
    """
    Get statistics about the database tables.

    Returns
    -------
    tuple
        Tuple containing two pandas DataFrames with statistics about the scrobble and artist_info tables
    """
    try:
        # Get the inspector
        inspector = inspect(engine)

        # Get table names
        table_names = inspector.get_table_names()

        # Initialize DataFrames
        scrobble_stats_df = None
        artist_info_stats_df = None

        # Process scrobble table
        if 'scrobble' in table_names:
            # Get column information for the scrobble table
            columns = inspector.get_columns('scrobble')
            # Filter out 'id' and 'play_time' columns, ensure 'artist_mbid' is included
            column_names = [col['name'] for col in columns if col['name'] not in ['id', 'play_time'] or col['name'] == 'artist_mbid']

            # Create a DataFrame to store statistics
            stats_data = []

            # Connect to the database and get statistics
            with engine.connect() as conn:
                # Get total row count
                total_rows = conn.execute(text("SELECT COUNT(*) FROM scrobble")).scalar()

                if total_rows > 0:
                    # Get statistics for each column
                    for col_name in column_names:
                        # Count non-null values
                        non_null_count = conn.execute(
                            text(f"SELECT COUNT(*) FROM scrobble WHERE {col_name} IS NOT NULL")
                        ).scalar()

                        # Calculate null count and percentages
                        null_count = total_rows - non_null_count
                        non_null_percentage = (non_null_count / total_rows) * 100
                        null_percentage = (null_count / total_rows) * 100

                        # Add to statistics data
                        stats_data.append({
                            'column_name': col_name,
                            'total_rows': total_rows,
                            'non_null_count': non_null_count,
                            'null_count': null_count,
                            'non_null_percentage': non_null_percentage,
                            'null_percentage': null_percentage
                        })

                    # Create DataFrame from statistics data
                    scrobble_stats_df = pd.DataFrame(stats_data)

        # Process artist_info table
        if 'artist_info' in table_names:
            # Get column information for the artist_info table
            columns = inspector.get_columns('artist_info')
            # Filter to include only the required columns
            required_columns = ['artist_name', 'mbid', 'disambiguation_comment', 'country', 'aliases']
            column_names = [col['name'] for col in columns if col['name'] in required_columns]

            # Create a DataFrame to store statistics
            stats_data = []

            # Connect to the database and get statistics
            with engine.connect() as conn:
                # Get total row count
                total_rows = conn.execute(text("SELECT COUNT(*) FROM artist_info")).scalar()

                if total_rows > 0:
                    # Get statistics for each column
                    for col_name in column_names:
                        # Count non-null values
                        non_null_count = conn.execute(
                            text(f"SELECT COUNT(*) FROM artist_info WHERE {col_name} IS NOT NULL")
                        ).scalar()

                        # Calculate null count and percentages
                        null_count = total_rows - non_null_count
                        non_null_percentage = (non_null_count / total_rows) * 100
                        null_percentage = (null_count / total_rows) * 100

                        # Add to statistics data
                        # Use 'disambi' as display name instead of 'disambiguation_comment'
                        display_col_name = 'disambi' if col_name == 'disambiguation_comment' else col_name
                        stats_data.append({
                            'column_name': display_col_name,
                            'total_rows': total_rows,
                            'non_null_count': non_null_count,
                            'null_count': null_count,
                            'non_null_percentage': non_null_percentage,
                            'null_percentage': null_percentage
                        })

                    # Create DataFrame from statistics data
                    artist_info_stats_df = pd.DataFrame(stats_data)

        return (scrobble_stats_df, artist_info_stats_df)

    except Exception as e:
        logging.error(f"Error getting database statistics: {e}")
        return (None, None)

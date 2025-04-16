import csv
from datetime import datetime
from dotenv import load_dotenv
import hashlib
import matplotlib.pyplot as plt
import openai
import os
import pandas as pd
import pymysql
import requests
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from time import sleep

load_dotenv()

# Constants
CSV_URL = "https://benjaminbenben.com/lastfm-to-csv/"
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_SECRET = os.environ.get("LASTFM_SECRET")
USER_AGENT = "MyLifeInData/1.0"  # More explicit user-agent
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/artist/"

db_configuration = {
    "host": os.environ.get("MyDB_HOST", "localhost"),
    "user": os.environ.get("MyDB_USER", "root"),
    "password": os.environ.get("MyDB_PASSWORD", ""),
    "database": "canonfodder"
}

# Pandas display settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
pd.options.display.float_format = "{: .2f}".format


# --------------------------------------------------------------
# Authentication function for Last.fm API
# --------------------------------------------------------------
def generate_lastfm_signature(params, secret):
    """
    Generate the signature for Last.fm requests.
    """
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    signature = hashlib.md5((sorted_params + secret).encode("utf-8")).hexdigest()
    return signature


# --------------------------------------------------------------
# Download CSV
# --------------------------------------------------------------
def download_csv():
    """
    Download scrobbles CSV file from CSV_URL and save locally with timestamp.
    """
    response = requests.get(CSV_URL)
    if response.status_code == 200:
        file_name = f"scrobbles_{datetime.now().strftime('%Y%m%d')}.csv"
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_name}")
        return file_name
    else:
        print("Failed to download CSV.")
        return None


# --------------------------------------------------------------
# Connect to the MySQL database
# --------------------------------------------------------------
def connect_db():
    """
    Create a PyMySQL connection using the config in db_configuration.
    """
    try:
        connection = pymysql.connect(**db_configuration)
        return connection
    except pymysql.MySQLError as e:
        print(f"Error connecting to DB: {e}")
        return None


# --------------------------------------------------------------
# Create a new table and import scrobbles
# --------------------------------------------------------------
def import_csv_to_db(file_name, connection):
    """
    Create a new MySQL table and insert scrobble data from the CSV file.
    """
    table_name = f"scrobbles_{datetime.now().strftime('%Y%m%d')}"
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    artist_name VARCHAR(2550),
                    album_name VARCHAR(2550),
                    track_name VARCHAR(2550),
                    play_time DATETIME,
                    country VARCHAR(255)
                );
            """)
            with open(file_name, 'r', encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    # row = [artist, album, track, date string]
                    cursor.execute(f"""
                        INSERT INTO {table_name} (artist_name, album_name, track_name, play_time, country)
                        VALUES (%s, %s, %s, STR_TO_DATE(%s, '%%d %%b %%Y %%H:%%i'), NULL);
                    """, row)
            connection.commit()
            print(f"Data imported to table {table_name}.")
        return table_name
    except Exception as e:
        print(f"Error importing CSV to DB: {e}")
        return None


# --------------------------------------------------------------
# Graceful request function
# --------------------------------------------------------------
def make_request(url, params=None, headers=None, max_retries=8):
    """
    Make a GET request to a given URL, with optional headers (including User-Agent).
    If a 5xx error occurs, retry up to max_retries times with a small delay.
    """
    if headers is None:
        headers = {}
    # Ensure we have our user-agent set
    if "User-Agent" not in headers:
        headers["User-Agent"] = USER_AGENT

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            response = requests.get(url, params=params, headers=headers)
            if 200 <= response.status_code < 300:
                return response
            elif 500 <= response.status_code < 600:
                print(f"Server error {response.status_code}, attempt {attempt}/{max_retries}. Retrying...")
                sleep(2)  # small backoff
                continue
            else:
                # Some other HTTP error (4xx, etc.)
                print(f"Failed request (status={response.status_code}): {response.text[:300]}")
                return response
        except Exception as e:
            print(f"Request error: {e}. Attempt {attempt}/{max_retries}")
            sleep(2)
    return None


# --------------------------------------------------------------
# Generic function to call last.fm
# --------------------------------------------------------------
def apiconnecting(api_method, user, page=None, limit=None):
    """
    Establish a connection to the last.fm API for a given user.
    Optionally pass 'page' and 'limit' as query params.
    """
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": api_method,
        "user": user,
        "api_key": LASTFM_API_KEY,
        "format": "json"
    }
    if page:
        params["page"] = page
    if limit:
        params["limit"] = limit

    # Make the request with a custom user-agent and retries
    resp = make_request(url, params=params, headers={"User-Agent": USER_AGENT}, max_retries=10)
    if not resp:
        # None means we never got a valid response after 3 retries
        print(f"Failed to fetch {api_method}: no valid response after retries.")
        return None

    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"Failed to fetch {api_method}. (status={resp.status_code})")
        return None


# --------------------------------------------------------------
# Misc data fetch
# --------------------------------------------------------------
def fetch_misc_data_from_lastfmapi():
    """
    Example: fetch miscellaneous data from Last.fm for a chosen user.
    """
    user = input("Enter your Last.fm username: ").strip()
    top_artists = apiconnecting("user.getTopArtists", user)
    top_albums = apiconnecting("user.getTopAlbums", user)
    top_tracks = apiconnecting("user.getTopTracks", user)

    # Print top artists
    if top_artists and "topartists" in top_artists:
        print("Top Artists:")
        for artist in top_artists["topartists"]["artist"][:3]:
            print("   ", artist.get("name", "N/A"))

    # Print top albums
    if top_albums and "topalbums" in top_albums:
        print("\nTop Albums:")
        for album in top_albums["topalbums"]["album"][:3]:
            print("   ", album.get("name", "N/A"))

    # Print top tracks
    if top_tracks and "toptracks" in top_tracks:
        print("\nTop Tracks:")
        for track in top_tracks["toptracks"]["track"][:3]:
            print("   ", track.get("name", "N/A"))

    # Print friends
    friends = apiconnecting("user.getFriends", user)
    if friends and "friends" in friends:
        print("\nFriends:")
        for friend in friends["friends"]["user"][:50]:
            print("   ", friend.get("name", "N/A"), "=", friend.get("realname", "N/A"))

    # Print user info
    infos = apiconnecting("user.getInfo", user)
    if infos and "user" in infos:
        user_info = infos["user"]
        print("\nUser Info:")
        print("   Name:", user_info.get("name", "N/A"))
        print("   Real Name:", user_info.get("realname", "N/A"))
        print("   Country:", user_info.get("country", "N/A"))
        print("   Playcount:", user_info.get("playcount", "N/A"))
        print("   Playlists:", user_info.get("playlists", "N/A"))
        print("   URL:", user_info.get("url", "N/A"))

    # Print loved tracks, including the artist name
    lovedtracks = apiconnecting("user.getLovedTracks", user)
    if lovedtracks and "lovedtracks" in lovedtracks:
        print("\nLoved tracks:")
        for lovedtrack in lovedtracks["lovedtracks"]["track"][:10]:
            print("   Track Name:", lovedtrack.get("name", "N/A"))
            artist_dict = lovedtrack.get("artist", {})
            print("   Artist:", artist_dict.get("name", "N/A"))
            print("---")

    # Print recent tracks
    recenttracks = apiconnecting("user.getRecentTracks", user)
    if recenttracks and "recenttracks" in recenttracks:
        print("\nRecent tracks:")
        for recenttrack in recenttracks["recenttracks"]["track"][:10]:
            print("   ", recenttrack.get("name", "N/A"))


# --------------------------------------------------------------
# Fetch all pages of recent tracks -> DataFrame
# --------------------------------------------------------------
def fetch_recent_tracks_all_pages(user):
    """
    Fetch all recent tracks for a Last.fm user, from page=1 onward,
    until no more pages of results are returned.
    Returns a Pandas DataFrame with columns [Artist, Song, Album, Timestamp].
    """
    page = 1
    limit = 200  # We can fetch up to 200 tracks per page
    all_tracks = []

    while True:
        result = apiconnecting("user.getRecentTracks", user, page=page, limit=limit)
        if not result or "recenttracks" not in result:
            break

        recenttracks = result["recenttracks"]
        tracks = recenttracks.get("track", [])
        if not tracks:
            # no more results
            break

        for t in tracks:
            artist_name = t.get("artist", {}).get("#text", "Unknown")
            song_title = t.get("name", "Unknown")
            album_title = t.get("album", {}).get("#text", "Unknown")
            if "date" in t:
                timestamp = t["date"].get("#text", "N/A")
            else:
                timestamp = "Current"
            all_tracks.append(
                {
                    "Artist": artist_name,
                    "Song": song_title,
                    "Album": album_title,
                    "Timestamp": timestamp
                }
            )
        page += 1  # next page

    df = pd.DataFrame(all_tracks, columns=["Artist", "Song", "Album", "Timestamp"])
    return df


# --------------------------------------------------------------
# MusicBrainz country fetch
# --------------------------------------------------------------
def fetch_country(artist_name):
    """
    Fetch country information from MusicBrainz API for an artist.
    """
    try:
        # We'll also set a distinct user-agent here for courtesy
        response = requests.get(
            f"{MUSICBRAINZ_URL}?query={artist_name}&fmt=json",
            headers={"User-Agent": "MyScript/1.0"}
        )
        if response.status_code == 200:
            data = response.json()
            if data.get('artists'):
                return data['artists'][0].get('country', 'Unknown')
        return 'Unknown'
    except Exception as e:
        print(f"Error fetching country for {artist_name}: {e}")
        return 'Unknown'


def enrich_scrobbles_with_country(table_name, connection):
    """
    Update scrobbles table with country data for each distinct artist.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT DISTINCT artist_name FROM {table_name} WHERE country IS NULL")
            artists = cursor.fetchall()
            for (artist_name,) in artists:
                country = fetch_country(artist_name)
                cursor.execute(f"UPDATE {table_name} SET country = %s WHERE artist_name = %s", (country, artist_name))
                print(f"Updated {artist_name} with country: {country}")
            connection.commit()
    except Exception as e:
        print(f"Error enriching scrobbles with country: {e}")


# --------------------------------------------------------------
# Simple LLM-based example for data insights
# --------------------------------------------------------------
def train_llm_and_enrich_data(connection, table_name):
    """
    Example function: fetch scrobbles, cluster by country, visualize, and attempt LLM commentary.
    """
    print("Starting LLM-based data enrichment...")
    query = f"SELECT artist_name, country FROM {table_name}"
    df = pd.read_sql(query, connection)

    if df.empty:
        print("No data found to process.")
        return

    # Example: country-based stats
    print("Country-based counts of artists:")
    country_counts = df['country'].value_counts()
    print(country_counts.head(10))

    # Perform simple clustering
    df = df.dropna()
    if df.empty:
        print("All rows dropped due to NaN; cannot cluster.")
        return
    tfidf = TfidfVectorizer(stop_words='english')
    country_features = tfidf.fit_transform(df['country'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(country_features)

    # Simple bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df['cluster'].value_counts().index, df['cluster'].value_counts().values)
    plt.title("Artist Clusters by Country")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Artists")
    plt.show()

    # (Illustrative) LLM step
    print("\nUsing LLM to generate insights (dummy example):")
    insights_prompt = f"""
        Summarize the country-based artist data:

        {df.groupby('country').size().to_string()}
    """
    try:
        response = openai.Completion.create(
            engine="o1",  # example engine
            max_tokens=150
        )
        print("\nLLM-generated insights:")
        print(response['choices'][0]['text'].strip())
    except Exception as e:
        print(f"Error using LLM: {e}")


# --------------------------------------------------------------
# Store recent tracks in MySQL
# --------------------------------------------------------------
def store_recent_tracks_in_db(df, connection):
    """
    Creates a new MySQL table for recent tracks and inserts each row from df.
    The table columns: artist_name, album_name, track_name, play_time, country
    We'll parse the df['Timestamp'] if possible, else store NULL.
    """
    table_name = f"scrobbles_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        with connection.cursor() as cursor:
            # Create table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    artist_name VARCHAR(2550),
                    album_name VARCHAR(2550),
                    track_name VARCHAR(2550),
                    play_time DATETIME,
                    country VARCHAR(255)
                );
            """)

            # Insert DataFrame rows
            for _, row in df.iterrows():
                ts_str = row["Timestamp"]
                if ts_str == "Current" or ts_str == "Unknown":
                    # We'll store NULL if not scrobbled or unknown
                    cursor.execute(f"""
                        INSERT INTO {table_name} 
                        (artist_name, album_name, track_name, play_time, country)
                        VALUES (%s, %s, %s, NULL, NULL);
                    """, (row["Artist"], row["Album"], row["Song"]))
                else:
                    # Attempt to parse with MySQL's STR_TO_DATE if the format is "9 Jun 2008, 17:16"
                    cursor.execute(f"""
                        INSERT INTO {table_name} 
                        (artist_name, album_name, track_name, play_time, country)
                        VALUES (%s, %s, %s, STR_TO_DATE(%s, '%%d %%b %%Y, %%H:%%i'), NULL);
                    """, (row["Artist"], row["Album"], row["Song"], ts_str))

            connection.commit()
            print(f"Data inserted into table {table_name}.")
        return table_name
    except Exception as e:
        print(f"Error storing recent tracks in DB: {e}")
        return None


# ----------------------------------------------------------------
# Querying MySQL DB for latest scrobble data
# ---------------------------------------------------------------

def query_top_artists_from_db(connection):
    c = "2a"


# --------------------------------------------------------------
# Main logic with user choice
# --------------------------------------------------------------
def main():
    while True:
        print("\n=== Main Menu ===")
        print("1) Download CSV and import into DB")
        print("2) Fetch miscellaneous test data from last.fm API")
        print("3) Fetch all scrobbled tracks from last.fm API and store them in MySQL + JSON")
        print("4) Quit")

        choice = input("Choose an option (1-4): ").strip()

        if choice == '1':
            file_name = download_csv()
            if not file_name:
                continue

            connection = connect_db()
            if not connection:
                print("Could not connect to DB; skipping import.")
                continue

            table_name = import_csv_to_db(file_name, connection)
            if table_name:
                enrich_scrobbles_with_country(table_name, connection)
                train_llm_and_enrich_data(connection, table_name)

            connection.close()

        elif choice == '2':
            fetch_misc_data_from_lastfmapi()

        elif choice == '3':
            user = input("Enter your Last.fm username: ").strip()
            df_recent = fetch_recent_tracks_all_pages(user)
            print("\nRecent Tracks DataFrame (head):")
            print(df_recent.head(20))
            df_recent.info()

            connection = connect_db()
            if connection:
                new_table = store_recent_tracks_in_db(df_recent, connection)
                connection.close()

            # write to JSON
            json_file = f"recent_tracks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df_recent.to_json(json_file, orient="records", date_format="iso")
            print(f"\nSaved recent tracks to JSON: {json_file}")

        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

        # If you want to run this script monthly in production, do so outside of this menu:
        # sleep(30 * 24 * 60 * 60)


if __name__ == "__main__":
    main()

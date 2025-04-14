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
USER_AGENT = "MyLifeInData"
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/artist/"
db_configuration = {
    "host": os.environ.get("MyDB_HOST", "localhost"),
    "user": os.environ.get("MyDB_USER", "root"),
    "password": os.environ.get("MyDB_PASSWORD", ""),
    "database": "mylifeindata",
    "use_pure": True
}
country_list_file = './T_ORSZAG.csv'


# Authentication function for Last.fm API
def generate_lastfm_signature(params, secret):
    sorted_params = "".join(f"{k}{v}" for k, v in sorted(params.items()))
    signature = hashlib.md5((sorted_params + secret).encode('utf-8')).hexdigest()
    return signature


# Download .csv from the provided URL
def download_csv():
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


# Connect to the MySQL database
def connect_db():
    connection = pymysql.connect(**db_configuration)
    return connection


# Create a new table and import scrobbles
def import_csv_to_db(file_name, connection):
    table_name = f"scrobbles_{datetime.now().strftime('%Y%m%d')}"
    with connection.cursor() as cursor:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                artist_name VARCHAR(255),
                album_name VARCHAR(255),
                track_name VARCHAR(255),
                play_time DATETIME,
                country VARCHAR(255)
            );
        """)
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                cursor.execute(f"""
                    INSERT INTO {table_name} (artist_name, album_name, track_name, play_time, country)
                    VALUES (%s, %s, %s, STR_TO_DATE(%s, '%%d %%b %%Y %%H:%%i'), NULL);
                """, row)
        connection.commit()
        print(f"Data imported to table {table_name}.")
    return table_name


# Fetch country information using MusicBrainz API
def fetch_country(artist_name):
    try:
        response = requests.get(f"{MUSICBRAINZ_URL}?query={artist_name}&fmt=json")
        if response.status_code == 200:
            data = response.json()
            if data.get('artists'):
                return data['artists'][0].get('country', 'Unknown')
        return 'Unknown'
    except Exception as e:
        print(f"Error fetching country for {artist_name}: {e}")
        return 'Unknown'


# Fetch top data from Last.fm API
def fetch_top_data(api_method, user):
    url = f"https://ws.audioscrobbler.com/2.0/"
    params = {
        'method': api_method,
        'user': user,
        'api_key': LASTFM_API_KEY,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch {api_method}.")
        return None


# Enrich scrobbles with country information
def enrich_scrobbles_with_country(table_name, connection):
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT DISTINCT artist_name FROM {table_name} WHERE country IS NULL")
        artists = cursor.fetchall()

        for (artist_name,) in artists:
            country = fetch_country(artist_name)
            cursor.execute(f"UPDATE {table_name} SET country = %s WHERE artist_name = %s", (country, artist_name))
            print(f"Updated {artist_name} with country: {country}")
            connection.commit()


# Train an LLM to enrich country-based data
def train_llm_and_enrich_data(connection, table_name):
    print("Training LLM and creating country-enriched statistics...")
    # Step 1: Load data from MySQL
    print("Fetching scrobble data from database...")
    query = f"SELECT artist_name, country FROM {table_name}"
    df = pd.read_sql(query, connection)
    # Check if data exists
    if df.empty:
        print("No data found to process.")
        return
    # Step 2: Generate country-based statistics
    print("Generating country-based statistics...")
    country_counts = df['country'].value_counts()
    print("Top countries by artist count:")
    print(country_counts.head(10))
    # Step 3: Cluster artists based on country and metadata
    print("Clustering artists by country...")
    df = df.dropna()
    tfidf = TfidfVectorizer(stop_words='english')
    country_features = tfidf.fit_transform(df['country'])
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(country_features)
    print(f"Artists grouped into {len(df['cluster'].unique())} clusters.")
    # Step 4: Visualize clusters
    print("Visualizing clusters...")
    plt.figure(figsize=(10, 6))
    plt.bar(df['cluster'].value_counts().index, df['cluster'].value_counts().values)
    plt.title("Artist Clusters by Country")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Artists")
    plt.show()
    # Step 5: Use LLM to describe insights
    print("Using LLM to generate insights...")
    insights_prompt = f"""
        Analyze the following country-based music artist data and generate a summary:
        {df.groupby('country').size().to_string()}
    """
    try:
        response = openai.Completion.create(
            engine="o1",
            prompt=insights_prompt,
            max_tokens=150
        )
        print("LLM-generated insights:")
        print(response['choices'][0]['text'].strip())
    except Exception as e:
        print(f"Error using LLM: {e}")
    # Step 6: Store enriched insights back into the database (Optional)
    print("Storing insights back into the database...")
    try:
        enriched_table_name = f"{table_name}_enriched"
        df.to_sql(enriched_table_name, connection, if_exists="replace", index=False)
        print(f"Enriched data stored in table: {enriched_table_name}")
    except Exception as e:
        print(f"Error saving enriched data: {e}")


# Main function
def main():
    # Step 1: Download the CSV
    file_name = download_csv()
    if not file_name:
        return

    # Step 2: Connect to the database
    connection = connect_db()

    # Step 3: Import CSV to MySQL
    table_name = import_csv_to_db(file_name, connection)

    # Step 4: Fetch top artists, albums, and tracks
    # top_artists = fetch_top_data("user.getTopArtists", "your_lastfm_username")
    # top_albums = fetch_top_data("user.getTopAlbums", "your_lastfm_username")
    # top_tracks = fetch_top_data("user.getTopTracks", "your_lastfm_username")

    # Step 5: Enrich scrobbles with country information
    enrich_scrobbles_with_country(table_name, connection)

    # Step 6: Enrich statistics with LLM
    train_llm_and_enrich_data()

    # Close DB connection
    connection.close()


if __name__ == "__main__":
    while True:
        main()
        # Run every month
        sleep(30 * 24 * 60 * 60)

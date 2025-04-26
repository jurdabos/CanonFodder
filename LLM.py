import matplotlib.pyplot as plt
import openai
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


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

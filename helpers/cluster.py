"""
Wraps assorted clustering utilities including fuzzy string scoring and
quality metrics.
"""
from functools import partial
import itertools
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.cluster import DBSCAN
from typing import Any, Sequence, Tuple


def _clf_scorer(x: str, y: str, clf, **kwargs) -> float:
    """Returns the probability that `clf` judges `x` and `y` identical"""
    return clf_proba(x, y, clf)


def anchors_ok(labelek, anchor_idx_halmazok):
    for indexek in anchor_idx_halmazok:
        lab = labelek[indexek[0]]
        if lab == -1 or any(labelek[i] != lab for i in indexek[1:]):
            return False
    return True


def clf_proba(a: str, b: str, clf) -> float:
    """Returns clf.predict_proba on the fuzzy-score vector of `a` and `b`"""
    vec = np.fromiter(fuzzy_scores(a, b).values(), dtype=float)[None, :]
    return float(clf.predict_proba(vec)[0, 1])


def calculate_clustering_metrics(name, labels, data, cluster_centers=None, model=None):
    """
    Calculates noise ratio silhouette weighted WSS and BIC
    Args:
        name: experiment identifier
        labels: array of cluster labels with noise marked as −1
        data: original dataset as NumPy array or dataframe
        cluster_centers: centroid coordinates when available
        model: fitted clustering model used for BIC
    Returns:
        dict containing the four metrics
    """
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    non_noise_indices = labels != -1
    clustered_data = data[non_noise_indices].to_numpy()
    clustered_labels = labels[non_noise_indices]
    # Calculating noise percentage
    noise_percentage = 1 - np.sum(non_noise_indices, axis=0) / len(labels)
    # Calculating silhouette score
    if len(np.unique(clustered_labels)) > 1:
        silhouette = silhouette_score(clustered_data, clustered_labels)
    else:
        silhouette = np.nan
    # Calculating WSS
    data = np.array(data)
    cluster_centers = np.array(cluster_centers)
    if cluster_centers is not None:
        wss = 0
        total_points = 0
        for i in range(len(cluster_centers)):
            # Extract points belonging to the current cluster
            cluster_points = clustered_data[clustered_labels == i]
            total_points += len(cluster_points)
            for j in range(len(cluster_points)):
                squared_distance = np.sum(
                    (cluster_points[j] - cluster_centers[i]) ** 2, axis=0
                )
                wss += squared_distance
        if total_points > 0:
            weighted_wss = wss / total_points
        else:
            weighted_wss = np.nan
    else:
        weighted_wss = np.nan
    # Calculating BIC
    if model is not None:
        n_clusters = len(np.unique(clustered_labels))
        n_features = data.shape[1]
        n_samples = len(clustered_data)
        # Log likelihood approximation for BIC
        if cluster_centers is not None:
            distances = cdist(clustered_data, cluster_centers)
            min_distances = np.min(distances, axis=1)
            log_likelihood = -0.5 * np.sum(min_distances ** 2, axis=0)
        else:
            log_likelihood = np.nan
        # BIC calculation
        n_params = n_clusters * n_features  # Approximate number of parameters
        bic = -2 * log_likelihood + n_params * np.log(n_samples)
    else:
        bic = np.nan  # BIC requires a model
    return {
        "Clustering Name": name,
        "Noise Percentage": noise_percentage,
        "Weighted WSS": weighted_wss,
        "Silhouette Score": silhouette,
        "BIC": bic,
    }


def dbscan_with_anchors(artist_names, dist_matrix, anchor_idx_sets,
                        eps_range=np.arange(0.05, 1.0, 0.01),
                        min_samples=2):
    for eps in eps_range:
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="precomputed").fit_predict(dist_matrix)
        if anchors_ok(labels, anchor_idx_sets):
            return eps, labels
    raise RuntimeError("No ε satisfies anchor constraints.")


def expand_pairs(row):
    if 'artist_variants' in row:
        variants = row['artist_variants'].split('{')
    else:
        raise KeyError("Neither 'artist_variants_text' nor 'artist_variants' found in dataframe row.")
    pairs = list(itertools.combinations(sorted(set(variants)), 2))
    return [(row['artist_variants'], pair[0], pair[1], row['to_link']) for pair in pairs]


def fuzzy_scores(a: str, b: str) -> dict:
    """Returns rapidfuzz similarity measures between two strings"""
    return {
        "ratio": fuzz.ratio(a, b) / 100,
        "partial_ratio": fuzz.partial_ratio(a, b) / 100,
        "token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100,
        "token_set_ratio": fuzz.token_set_ratio(a, b) / 100,
        "WRatio": fuzz.WRatio(a, b) / 100,
        "QRatio": fuzz.QRatio(a, b) / 100
    }


def most_similar(name: str,
                 choices: Sequence[str],
                 clf,
                 threshold: float = 0.5) -> Tuple[str | None, float]:
    """
    Returns the best match above `threshold` together with its probability
    """
    scorer = partial(_clf_scorer, clf=clf)
    match, score, _ = process.extractOne(
        name, choices,
        scorer=scorer,
        score_cutoff=threshold
    )
    return match, score or 0.0


variant_sets = [
    ["Beatles", "The Beatles"],
    ["Bohren & der Club of Gore", "Bohren und der Club of Gore"],
    ["Gorillaz, Adeleye Omotayo", "GorillazAdeleye Omotayo"],
    ["Gorillaz, Bad Bunny", "GorillazBad Bunny"],
    ["Gorillaz, Beck", "GorillazBeck"],
    ["Gorillaz, Stevie Nicks", "GorillazStevie Nicks"],
    ["Gorillaz, Tame Impala, Bootie Brown", "GorillazTame ImpalaBootie Brown"],
    ["Gorillaz, Thundercat", "GorillazThundercat"],
    ["Robert Miles & Trilok Gurtu",
     "Robert Miles And Trilok Gurtu",
     "Robert Miles, Trilok Gurtu"],
    ["La Monte Young", "Lamonte Young"],
]

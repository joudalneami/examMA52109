"""
agglomerative.py
New module for hierarchical clustering using scikit-learn's
AgglomerativeClustering, designed to match the style of cluster_maker.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(X: np.ndarray, n_clusters: int = 3):
    """
    Run Agglomerative (hierarchical) clustering on dataset X.

    Parameters
    ----------
    X : ndarray
        Feature matrix of shape (n_samples, n_features).
    n_clusters : int
        Number of clusters to compute.

    Returns
    -------
    labels : ndarray
        Array of cluster labels for each sample.
    centroids : None
        Agglomerative clustering does not compute centroids,
        but return None to match the interface used by run_clustering.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    # No centroids in hierarchical clustering → return None
    return labels, None

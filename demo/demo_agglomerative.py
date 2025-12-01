"""
demo_agglomerative.py
Task 5 demonstration script for hierarchical clustering
on difficult_dataset.csv using the new agglomerative module.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.interface import run_clustering
from cluster_maker.plotting_clustered import plot_clusters_2d


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(root, "data", "difficult_dataset.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Could not find difficult_dataset.csv at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print("Loaded difficult dataset:")
    print(df.head())

    # Use first two numeric columns
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if len(numeric_cols) < 2:
        print("ERROR: Not enough numeric columns.")
        sys.exit(1)

    feature_cols = numeric_cols[:2]
    print(f"Using features: {feature_cols}")

    # -----------------------------
    # 2. Run agglomerative clustering
    # -----------------------------
    output_dir = os.path.join(os.path.dirname(__file__), "agglomerative_output")
    os.makedirs(output_dir, exist_ok=True)

    k = 3  # reasonable default for difficult dataset

    print(f"\n=== Running Agglomerative Clustering with k = {k} ===")

    result = run_clustering(
        input_path=data_path,
        feature_cols=feature_cols,
        algorithm="agglomerative",
        k=k,
        standardise=True,
        output_path=os.path.join(output_dir, f"difficult_clustered_k{k}.csv"),
        random_state=None,    # not used for hierarchical
        compute_elbow=False,  # elbow not meaningful for hierarchical
    )

    # -----------------------------
    # 3. Plot the clusters
    # -----------------------------
    fig, ax = plot_clusters_2d(
        result["data"][feature_cols].values,
        result["labels"],
        centroids=None,
        title=f"Difficult Dataset – Agglomerative Clustering (k={k})"
    )

    fig.savefig(os.path.join(output_dir, f"difficult_agglomerative_k{k}.png"))
    plt.close(fig)

    print("\nClustering complete. Outputs saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()

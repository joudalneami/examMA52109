###
# demo/simulated_clustering.py
#
# Task 4 – clustering analysis of simulated_data.csv
#
# This script uses only the cluster_maker toolkit to:
#   1. Load simulated data
#   2. Select features
#   3. Standardise them
#   4. Run clustering for several k values
#   5. Produce cluster plots and an elbow plot
#   6. Output a brief reasoning for which k is appropriate
###

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from cluster_maker.interface import run_clustering
from cluster_maker.plotting_clustered import plot_clusters_2d, plot_elbow


def main():

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    root = os.path.dirname(os.path.dirname(__file__))  # project root
    data_path = os.path.join(root, "data", "simulated_data.csv")

    if not os.path.exists(data_path):
        print(f"ERROR: Could not find simulated_data.csv at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print("Loaded dataset:")
    print(df.head())

    # Use the first two numeric columns as features
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if len(numeric_cols) < 2:
        print("ERROR: simulated_data.csv does not have enough numeric columns.")
        sys.exit(1)

    feature_cols = numeric_cols[:2]
    print(f"Using features: {feature_cols}")

    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), "simulated_output")
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 2. Run clustering for k = 2–6
    # -----------------------------
    K_values = [2, 3, 4, 5, 6]
    inertia_values = []

    for k in K_values:
        print(f"\n=== Running clustering with k = {k} ===")

        out_csv = os.path.join(output_dir, f"simulated_clustered_k{k}.csv")

        result = run_clustering(
            input_path=data_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=out_csv,
            random_state=42,
            compute_elbow=False,
        )

        # Cluster plot using the raw plotting function for clarity
        fig, ax = plot_clusters_2d(
            result["data"][feature_cols].values,
            result["labels"],
            centroids=result.get("centroids"),
            title=f"Simulated Data – k={k}",
        )

        fig.savefig(os.path.join(output_dir, f"simulated_k{k}.png"))
        plt.close(fig)

        inertia_values.append(result["metrics"]["inertia"])

    # -----------------------------
    # 3. Elbow plot
    # -----------------------------
    fig_elbow, ax_elbow = plot_elbow(K_values, inertia_values, title="Elbow Curve – Simulated Data")
    fig_elbow.savefig(os.path.join(output_dir, "elbow_plot.png"))
    plt.close(fig_elbow)

    # -----------------------------
    # 4. Reasoning: Pick plausible k
    # -----------------------------
    print("\n=== Interpretation ===")
    print("We plotted inertia vs k. A good clustering choice is often near the elbow point.")
    print("Inspect the elbow plot in simulated_output/elbow_plot.png to see where inertia")
    print("levels off. That point (often k=3 or k=4 in this dataset) is the plausible choice.")


if __name__ == "__main__":
    main()

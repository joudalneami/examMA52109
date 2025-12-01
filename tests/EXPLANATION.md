# Task 2 – Explanation

1. What was wrong and how it was fixed
In demo/cluster_plot.py, the script was supposed to run k-means clustering for k = 2, 3, 4, 5. However, inside the loop the argument was:

k = min(k, 3)

This caused incorrect behaviour:
- For k = 2 → it used k = 2 (correct)
- For k = 3 → it used k = 3 (correct)
- For k = 4 → it used k = 3 (incorrect)
- For k = 5 → it used k = 3 (incorrect)

The script still created files labelled as if k=4 and k=5 were used, even though the clustering was actually done with k=3. This made the results misleading.

Fix:
I replaced the argument with:

k = k

so the script now truly runs clustering with k = 2, 3, 4, and 5.

------------------------------------------------------------

2. What the corrected script does
After fixing the bug, the script:

- Loads the CSV file
- Selects the first two numeric columns as features
- Standardises the features
- Runs k-means clustering for k = 2, 3, 4, 5
- Saves a CSV file with cluster labels for each k
- Produces a 2D scatter plot (PNG) for each k value
- Computes metrics such as inertia and silhouette score
- Saves a metrics summary file comparing all values of k
- Produces a silhouette bar plot if available

The script now behaves correctly and matches the intended behaviour.

------------------------------------------------------------

3. Overview of the cluster_maker package

dataframe_builder.py
- Builds a seed DataFrame describing cluster centres
- Simulates clustered data around the centres

preprocessing.py
- Selects feature columns
- Standardises features using StandardScaler

algorithms.py
- Implements manual k-means
- Includes a wrapper for scikit-learn KMeans
- Provides helper functions such as centroid initialisation and cluster assignment

data_analyser.py
- Computes numeric summaries and correlations

evaluation.py
- Computes inertia
- Computes silhouette scores
- Provides elbow-curve tools

plotting_clustered.py
- Creates 2D scatter plots of clusters
- Creates elbow plots

data_exporter.py
- Exports DataFrames to CSV or text formats

interface.py
- High-level run_clustering() function
- Handles preprocessing, running algorithms, computing metrics, plotting, and exporting results

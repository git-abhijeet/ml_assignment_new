# Mall Customers K-Means Assignment

## Dataset
Place `Mall_Customers.csv` in the same folder as this notebook (`assignment_k-means_clustering/`).

## Environment Setup
```zsh
python -m venv abhi
source abhi/bin/activate
pip install -r assignment_k-means_clustering/requirements.txt
```

## Run
Open `assignment_k-means_clustering/assignment.ipynb` and run cells top-to-bottom. Outputs are saved to `assignment_k-means_clustering/outputs/`.

## Artifacts
- `outputs/scaler.joblib` — StandardScaler
- `outputs/kmeans_k{K}.joblib` — Trained KMeans model
- `outputs/customers_with_clusters.csv` — Labeled dataset
- `outputs/cluster_centroids.csv` — Centroid values in original units

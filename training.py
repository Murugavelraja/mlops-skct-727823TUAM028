# Murugavel Rajan S - 727823TUAM028

import mlflow
import time
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("SKCT_727823TUAM028_CustomerSegmentation")

# -------------------------------
# Create synthetic customer data
# -------------------------------
X, _ = make_blobs(
    n_samples=1000,
    n_features=6,
    centers=5,
    cluster_std=1.5,
    random_state=42
)

columns = ["Age", "Income", "SpendingScore",
           "Savings", "PurchaseFrequency", "LoanAmount"]

X = pd.DataFrame(X, columns=columns)

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

best_score = -1
best_model = None

# -------------------------------
# Run experiments (12 runs)
# -------------------------------
for i in range(12):
    with mlflow.start_run():

        seed = i
        np.random.seed(seed)

        start_time = time.time()

        # Alternate models
        if i < 6:
            k = 2 + i
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            model_name = "KMeans"
        else:
            k = 3 + (i - 6)
            model = AgglomerativeClustering(n_clusters=k)
            model_name = "Agglomerative"

        # Fit model
        labels = model.fit_predict(X_scaled)

        end_time = time.time()

        # Metrics
        sil_score = silhouette_score(X_scaled, labels)
        db_index = davies_bouldin_score(X_scaled, labels)

        # Save model temporarily
        joblib.dump(model, "model.pkl")
        model_size = os.path.getsize("model.pkl") / (1024 * 1024)

        # -------------------------------
        # MLflow logging
        # -------------------------------
        mlflow.log_param("model", model_name)
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("random_seed", seed)

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_metric("davies_bouldin_index", db_index)

        mlflow.log_metric("training_time_seconds", end_time - start_time)
        mlflow.log_metric("model_size_mb", model_size)
        mlflow.log_metric("n_features", X.shape[1])

        mlflow.set_tag("student_name", "Murugavel Rajan S")
        mlflow.set_tag("roll_number", "727823TUAM028")
        mlflow.set_tag("dataset", "CustomerSegmentation")

        mlflow.log_param("model_saved", "yes")

        # Save best model (higher silhouette is better)
        if sil_score > best_score:
            best_score = sil_score
            best_model = model

        print(f"Run {i} | Model: {model_name} | Silhouette: {sil_score:.3f}")

# -------------------------------
# Save best model
# -------------------------------
joblib.dump(best_model, "best_model.pkl")

print("\nBest Silhouette Score:", best_score)
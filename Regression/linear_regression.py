# =========================
# Linear Regression (CSV)
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    silhouette_score
)

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.data

# -------------------------
# Load Dataset
# -------------------------
def run_linear(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column name if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "LinearRegression")
        
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        
        mlflow.log_artifact("data.csv", artifact_path="data")
        
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def regression_random_forest(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column name if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )

def gbt_regressor(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column name if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )

def clustering_bisecting_kmeans(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    # Clustering → numeric features only
    X = df.select_dtypes(include=[np.number])

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # Bisecting K-Means Logic
    # -------------------------
    def bisecting_kmeans(X, n_clusters=3, random_state=42):
        clusters = {0: X}

        for i in range(1, n_clusters):
            # Pick the largest cluster to split
            largest_cluster_key = max(clusters, key=lambda k: len(clusters[k]))
            data_to_split = clusters.pop(largest_cluster_key)

            kmeans = KMeans(
                n_clusters=2,
                random_state=random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(data_to_split)

            clusters[f"{largest_cluster_key}_0"] = data_to_split[labels == 0]
            clusters[f"{largest_cluster_key}_1"] = data_to_split[labels == 1]

        return clusters

    # -------------------------
    # Train Model
    # -------------------------
    clusters = bisecting_kmeans(X_scaled, n_clusters=3)

    # Build labels array
    labels = np.zeros(len(X_scaled), dtype=int)
    current_label = 0
    start_idx = 0

    for cluster_data in clusters.values():
        size = len(cluster_data)
        labels[start_idx:start_idx + size] = current_label
        start_idx += size
        current_label += 1

    # -------------------------
    # Evaluation
    # -------------------------
    silhouette = silhouette_score(X_scaled, labels)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "BisectingKMeans")
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("silhouette_score", silhouette)

        mlflow.log_artifact("data.csv", artifact_path="data")

def clustering_gaussian_mixture(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    # Clustering → numeric features only
    X = df.select_dtypes(include=[np.number])

    # -------------------------
    # Scaling
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # Train Model
    # -------------------------
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(X_scaled)

    # -------------------------
    # Evaluation
    # -------------------------
    labels = gmm.predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "GaussianMixture")
        mlflow.log_param("n_components", 3)
        mlflow.log_param("covariance_type", "full")
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("silhouette_score", silhouette)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            gmm,
            name="model",
            input_example=input_example
        )
    
def clustering_kmeans(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    # Clustering → NO target column
    X = df.select_dtypes(include=[np.number])

    # -------------------------
    # Scaling (IMPORTANT)
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # Train Model
    # -------------------------
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    # -------------------------
    # Evaluation
    # -------------------------
    labels = kmeans.labels_
    silhouette = silhouette_score(X_scaled, labels)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_init", 10)

        mlflow.log_metric("silhouette_score", silhouette)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            kmeans,
            name="model",
            input_example=input_example
        )

def decision_tree(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )

def LogisticRegression(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column name if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Binary classification

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )

def classification_random_forest(experiment_type, run_type):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)   # CHANGE Target column if needed
    y = df["Target"]

    # -------------------------
    # Train / Test Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Train Model
    # -------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # -------------------------
    # Evaluation
    # -------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # -------------------------
    # MLflow Logging
    # -------------------------
    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example
        )


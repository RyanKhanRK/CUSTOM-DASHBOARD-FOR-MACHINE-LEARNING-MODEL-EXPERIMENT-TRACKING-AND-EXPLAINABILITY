import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
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

import mlflow
import mlflow.sklearn
import mlflow.data

def regression_run_linear(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_artifact("data.csv", artifact_path="data")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def regression_random_forest(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def regression_gbt_regressor(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def regression_decision_tree(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "DecisionTreeRegressor")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

def clustering_bisecting_kmeans(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def bisecting_kmeans(X, n_clusters=3, random_state=42):
        clusters = {0: X}

        for i in range(1, n_clusters):
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

    clusters = bisecting_kmeans(X_scaled, n_clusters=3)

    labels = np.zeros(len(X_scaled), dtype=int)
    current_label = 0
    start_idx = 0

    for cluster_data in clusters.values():
        size = len(cluster_data)
        labels[start_idx:start_idx + size] = current_label
        start_idx += size
        current_label += 1

    silhouette = silhouette_score(X_scaled, labels)
    mlflow.set_experiment(experiment_type)

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "BisectingKMeans")
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_artifact("data.csv", artifact_path="data")

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def clustering_gaussian_mixture(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=42
    )

    gmm.fit(X_scaled)

    labels = gmm.predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    mlflow.set_experiment(experiment_type)

    input_example = X.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "GaussianMixture")
        mlflow.log_param("n_components", 3)
        mlflow.log_param("covariance_type", "full")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            gmm,
            artifact_path="model",
            input_example=input_example
        )
    
        return {
            "run_id": mlflow.active_run().info.run_id,
        }
    
def clustering_kmeans(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    silhouette = silhouette_score(X_scaled, labels)

    mlflow.set_experiment(experiment_type)

    input_example = X.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "KMeans")
        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_init", 10)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            kmeans,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def clustering_lda(experiment_type, run_type, dataset_level):
    df = pd.read_csv("data.csv")

    TEXT_COLUMN = "text"

    documents = df[TEXT_COLUMN].astype(str).tolist()

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words="english"
    )

    X = vectorizer.fit_transform(documents)

    lda = LatentDirichletAllocation(
        n_components=5,
        learning_method="batch",
        random_state=42
    )

    lda.fit(X)

    perplexity = lda.perplexity(X)

    mlflow.set_experiment(experiment_type)

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "LDA")
        mlflow.log_param("n_topics", 5)
        mlflow.log_param("learning_method", "batch")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("perplexity", perplexity)
        mlflow.log_artifact("data.csv", artifact_path="data")
        mlflow.sklearn.log_model(
            lda,
            artifact_path="model",
            input_example=X[:5]
        )

def classification_decision_tree(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    # Create a copy to avoid modifying original data
    df = df_processed.copy()
    
    # Encode categorical columns
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    # Handle any missing values (optional but recommended)
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1) 
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("data.csv", artifact_path="data")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_LogisticRegression(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_random_forest(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.set_experiment(experiment_type)

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name=run_type):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_GBT_Classifier(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")
    
    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.set_experiment("Classification")

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name="Gradient_Boosting_Classifier"):
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_LinearSVC(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.set_experiment("Classification")

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name="Linear_SVC"):
        mlflow.log_param("model_type", "LinearSVC")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("max_iter", 5000)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1", f1)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_MLPC(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")
    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    mlflow.set_experiment("Classification")
    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name="MultilayerPerceptronClassifier"):
        mlflow.log_param("model_type", "MultilayerPerceptronClassifier")
        mlflow.log_param("hidden_layers", "(100, 50)")
        mlflow.log_param("activation", "relu")
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("solver", "adam")
        mlflow.log_param("max_iter", 500)

        preds = pipeline.predict(X_test)

        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds, average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test, preds, average="weighted"))
        mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_NaiveBayes(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1) 
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.set_experiment("Classification")

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name="Naive_Bayes_Gaussian"):
        mlflow.log_param("model_type", "GaussianNB")
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_metric("F1", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("data.csv", artifact_path="data")
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

def classification_OVR(experiment_type, run_type, dataset_level):
    df_processed = pd.read_csv("data.csv")

    df = df_processed.copy()
    
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = LogisticRegression(max_iter=1000)

    model = OneVsRestClassifier(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.set_experiment("Classification")

    input_example = X_train.iloc[:5]

    with mlflow.start_run(run_name="One_vs_Rest_Classifier"):
        mlflow.log_param("model_type", "OneVsRestClassifier")
        mlflow.log_param("database_level", dataset_level)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision_weighted", precision)
        mlflow.log_metric("Recall_weighted", recall)
        mlflow.log_metric("F1_weighted", f1)
        mlflow.log_artifact("data.csv", artifact_path="data")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        return {
            "run_id": mlflow.active_run().info.run_id,
        }

"""
Complete ML Training Module with Hyperparameter Support
Supports: Regression, Classification, and Clustering models
Backend: scikit-learn
All models use user-provided hyperparameters
With proper MLflow run management for sequential training
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import mlflow
import mlflow.sklearn
import atexit


# ============================================================================
# MLFLOW RUN MANAGEMENT UTILITIES
# ============================================================================

def _ensure_run_closed():
    """
    Ensure any active MLflow run is properly closed.
    
    This is critical for running multiple models sequentially to ensure
    each model gets its own isolated run with proper metrics logging.
    """
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
            return True
    except Exception as e:
        print(f"Warning: Error closing MLflow run: {e}")
    return False


def _cleanup_mlflow_on_exit():
    """Register cleanup function to ensure MLflow runs are closed on exit"""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except:
        pass


# Register cleanup on exit
atexit.register(_cleanup_mlflow_on_exit)


# ============================================================================
# MODEL FACTORY FUNCTIONS - Create models with hyperparameters
# ============================================================================

class ModelFactory:
    """Factory class to create models with hyperparameters"""

    @staticmethod
    def create_linear_regression(hyperparameters: Dict[str, Any]):
        """Create Linear Regression (Ridge/Lasso/ElasticNet)"""
        regParam = hyperparameters.get("regParam", 0.0)
        elasticNetParam = hyperparameters.get("elasticNetParam", 0.0)
        maxIter = hyperparameters.get("maxIter", 400)
        tol = hyperparameters.get("tol", 1e-6)
        fitIntercept = hyperparameters.get("fitIntercept", True)
        solver = hyperparameters.get("solver", "auto")

        solver_map = {"auto": "auto", "normal": "svd", "l-bfgs": "lbfgs"}
        mapped_solver = solver_map.get(solver, "auto")

        if elasticNetParam == 0.0:
            return Ridge(alpha=regParam, fit_intercept=fitIntercept, solver=mapped_solver, max_iter=maxIter, tol=tol)
        elif elasticNetParam == 1.0:
            return Lasso(alpha=regParam, fit_intercept=fitIntercept, max_iter=maxIter, tol=tol)
        else:
            return ElasticNet(alpha=regParam, l1_ratio=elasticNetParam, fit_intercept=fitIntercept, max_iter=maxIter, tol=tol)

    @staticmethod
    def create_decision_tree_regressor(hyperparameters: Dict[str, Any]):
        """Create Decision Tree Regressor"""
        return DecisionTreeRegressor(
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            random_state=hyperparameters.get("seed", 42),
            splitter=hyperparameters.get("splitter", "best")
        )

    @staticmethod
    def create_random_forest_regressor(hyperparameters: Dict[str, Any]):
        """Create Random Forest Regressor"""
        feature_subset_map = {
            "auto": "sqrt", "sqrt": "sqrt", "log2": "log2", "all": None, "onethird": None
        }
        mapped_feature_subset = feature_subset_map.get(
            hyperparameters.get("featureSubsetStrategy", "auto"), "sqrt"
        )

        return RandomForestRegressor(
            n_estimators=hyperparameters.get("numTrees", 20),
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            min_samples_split=hyperparameters.get("minInstancesPerNode", 2),
            max_features=mapped_feature_subset,
            max_samples=hyperparameters.get("subsamplingRate", 1.0),
            random_state=hyperparameters.get("seed", 42),
            n_jobs=-1
        )

    @staticmethod
    def create_gradient_boosting_regressor(hyperparameters: Dict[str, Any]):
        """Create Gradient Boosting Regressor"""
        return GradientBoostingRegressor(
            n_estimators=hyperparameters.get("maxIter", 20),
            max_depth=hyperparameters.get("maxDepth", 5),
            learning_rate=hyperparameters.get("stepSize", 0.1),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            min_samples_split=hyperparameters.get("minInstancesPerNode", 2),
            subsample=hyperparameters.get("subsamplingRate", 1.0),
            loss=hyperparameters.get("lossType", "squared"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_logistic_regression(hyperparameters: Dict[str, Any]):
        """Create Logistic Regression"""
        regParam = hyperparameters.get("regParam", 0.0)
        elasticNetParam = hyperparameters.get("elasticNetParam", 0.0)
        maxIter = hyperparameters.get("maxIter", 100)
        tol = hyperparameters.get("tol", 1e-6)
        fitIntercept = hyperparameters.get("fitIntercept", True)

        # Determine penalty type based on elasticNetParam
        if elasticNetParam == 0.0:
            penalty = "l2"
        elif elasticNetParam == 1.0:
            penalty = "l1"
        else:
            penalty = "elasticnet"

        return LogisticRegression(
            C=1.0 / (regParam + 1e-10) if regParam > 0 else 1.0,
            penalty=penalty,
            fit_intercept=fitIntercept,
            max_iter=maxIter,
            tol=tol,
            random_state=hyperparameters.get("seed", 42),
            solver="saga" if penalty == "elasticnet" else "lbfgs"
        )

    @staticmethod
    def create_decision_tree_classifier(hyperparameters: Dict[str, Any]):
        """Create Decision Tree Classifier"""
        return DecisionTreeClassifier(
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            criterion=hyperparameters.get("impurity", "gini"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_random_forest_classifier(hyperparameters: Dict[str, Any]):
        """Create Random Forest Classifier"""
        print(hyperparameters)
        feature_subset_map = {
            "auto": "sqrt", "sqrt": "sqrt", "log2": "log2", "all": None, "onethird": None
        }
        mapped_feature_subset = feature_subset_map.get(
            hyperparameters.get("featureSubsetStrategy", "auto"), "sqrt"
        )

        return RandomForestClassifier(
            n_estimators=hyperparameters.get("numTrees", 20),
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            min_samples_split=hyperparameters.get("minInstancesPerNode", 5),
            max_features=mapped_feature_subset,
            criterion=hyperparameters.get("impurity", "gini"),
            random_state=hyperparameters.get("seed", 42),
            n_jobs=-1
        )

    @staticmethod
    def create_gradient_boosting_classifier(hyperparameters: Dict[str, Any]):
        """Create Gradient Boosting Classifier"""
        return GradientBoostingClassifier(
            n_estimators=hyperparameters.get("maxIter", 20),
            max_depth=hyperparameters.get("maxDepth", 5),
            learning_rate=hyperparameters.get("stepSize", 0.1),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            min_samples_split=hyperparameters.get("minInstancesPerNode", 2),
            subsample=hyperparameters.get("subsamplingRate", 1.0),
            loss=hyperparameters.get("lossType", "log_loss"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_naive_bayes(hyperparameters: Dict[str, Any]):
        """Create Gaussian Naive Bayes"""
        return GaussianNB(var_smoothing=hyperparameters.get("smoothing", 1e-9))

    @staticmethod
    def create_linear_svc(hyperparameters: Dict[str, Any]):
        """Create Linear SVC"""
        return LinearSVC(
            C=hyperparameters.get("C", 1.0),
            max_iter=hyperparameters.get("maxIter", 1000),
            tol=hyperparameters.get("tol", 1e-3),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_mlp_classifier(hyperparameters: Dict[str, Any]):
        """Create MLP Classifier with Pipeline"""
        layers = hyperparameters.get("layers", (100, 50))
        
        # Convert string to tuple if needed
        if isinstance(layers, str):
            import json
            import ast
            try:
                # Try JSON format first: "[10, 15, 10, 2]"
                layers = tuple(json.loads(layers))
            except (json.JSONDecodeError, ValueError):
                try:
                    # Fall back to Python syntax: "(10, 15, 10, 2)"
                    layers = tuple(ast.literal_eval(layers))
                except (ValueError, SyntaxError):
                    # Default if all else fails
                    layers = (100, 50)
        
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=tuple(layers[1:-1]),
                activation=hyperparameters.get("activation", "relu"),
                solver=hyperparameters.get("solver", "adam"),
                max_iter=hyperparameters.get("maxIter", 500),
                tol=hyperparameters.get("tol", 1e-4),
                random_state=hyperparameters.get("seed", 42)
            ))
        ])

    @staticmethod
    def create_ovr_classifier(base_classifier_name: str, hyperparameters: Dict[str, Any]):
        """Create One-vs-Rest Classifier"""
        base_models = {
            "LogisticRegression": ModelFactory.create_logistic_regression,
            "DecisionTreeClassifier": ModelFactory.create_decision_tree_classifier,
            "RandomForestClassifier": ModelFactory.create_random_forest_classifier,
            "GradientBoostingClassifier": ModelFactory.create_gradient_boosting_classifier,
            "NaiveBayes": ModelFactory.create_naive_bayes,
            "LinearSVC": ModelFactory.create_linear_svc
        }

        if base_classifier_name not in base_models:
            raise ValueError(f"Unknown base classifier: {base_classifier_name}")

        base_model = base_models[base_classifier_name](hyperparameters)
        return OneVsRestClassifier(base_model)

    @staticmethod
    def create_kmeans(hyperparameters: Dict[str, Any]):
        """Create KMeans Clustering"""
        return KMeans(
            n_clusters=hyperparameters.get("k", 2),
            max_iter=hyperparameters.get("maxIter", 20),
            tol=hyperparameters.get("tol", 1e-4),
            n_init=hyperparameters.get("n_init", 10),
            random_state=hyperparameters.get("seed", 42),
            algorithm="lloyd" if hyperparameters.get("initMode", "k-means||") == "k-means||" else "auto"
        )

    @staticmethod
    def create_gaussian_mixture(hyperparameters: Dict[str, Any]):
        """Create Gaussian Mixture Model"""
        return GaussianMixture(
            n_components=hyperparameters.get("k", 2),
            max_iter=hyperparameters.get("maxIter", 100),
            tol=hyperparameters.get("tol", 1e-3),
            covariance_type=hyperparameters.get("covarianceType", "full"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_lda(hyperparameters: Dict[str, Any]):
        """Create Latent Dirichlet Allocation"""
        return LatentDirichletAllocation(
            n_components=hyperparameters.get("k", 10),
            max_iter=hyperparameters.get("maxIter", 20),
            learning_method=hyperparameters.get("optimizer", "online"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_bisecting_kmeans(hyperparameters: Dict[str, Any]):
        """Create custom Bisecting K-Means"""
        return BisectingKMeans(
            n_clusters=hyperparameters.get("k", 3),
            max_iter=hyperparameters.get("maxIter", 20),
            tol=hyperparameters.get("tol", 1e-4),
            n_init=hyperparameters.get("n_init", 10),
            random_state=hyperparameters.get("seed", 42)
        )


# ============================================================================
# HYPERPARAMETER VALIDATION
# ============================================================================

class HyperparameterValidator:
    """Validate hyperparameters before model training"""
    
    @staticmethod
    def convert_hyperparameters(params):
        """Convert string booleans to actual booleans"""
        converted = {}
        
        for key, value in params.items():
            # Handle string booleans from JSON
            if isinstance(value, str):
                if value.lower() == 'true':
                    converted[key] = True
                elif value.lower() == 'false':
                    converted[key] = False
                else:
                    # Try to convert to int/float if possible
                    try:
                        if '.' in str(value):
                            converted[key] = float(value)
                        else:
                            converted[key] = int(value)
                    except ValueError:
                        converted[key] = value
            else:
                converted[key] = value
        
        return converted

    @staticmethod
    def validate_linear_regression(hyperparameters: Dict[str, Any]):
        """Validate Linear Regression hyperparameters"""
        if "regParam" in hyperparameters and hyperparameters["regParam"] < 0:
            raise ValueError("regParam must be >= 0")
        if "elasticNetParam" in hyperparameters:
            param = hyperparameters["elasticNetParam"]
            if not (0 <= param <= 1):
                raise ValueError("elasticNetParam must be between 0 and 1")
        if "maxIter" in hyperparameters and hyperparameters["maxIter"] <= 0:
            raise ValueError("maxIter must be > 0")
        if "tol" in hyperparameters and hyperparameters["tol"] <= 0:
            raise ValueError("tol must be > 0")

    @staticmethod
    def validate_tree_hyperparameters(hyperparameters: Dict[str, Any]):
        """Validate tree-based model hyperparameters"""
        if "maxDepth" in hyperparameters and hyperparameters["maxDepth"] <= 0:
            raise ValueError("maxDepth must be > 0")
        if "minInstancesPerNode" in hyperparameters and hyperparameters["minInstancesPerNode"] < 1:
            raise ValueError("minInstancesPerNode must be >= 1")

    @staticmethod
    def validate_ensemble_hyperparameters(hyperparameters: Dict[str, Any]):
        """Validate ensemble model hyperparameters"""
        HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)
        if "numTrees" in hyperparameters and hyperparameters["numTrees"] <= 0:
            raise ValueError("numTrees must be > 0")
        if "subsamplingRate" in hyperparameters:
            rate = hyperparameters["subsamplingRate"]
            if not (0 < rate <= 1):
                raise ValueError("subsamplingRate must be between 0 and 1")

    @staticmethod
    def validate_clustering_hyperparameters(hyperparameters: Dict[str, Any]):
        """Validate clustering model hyperparameters"""
        if "k" in hyperparameters and hyperparameters["k"] <= 0:
            raise ValueError("k must be > 0")
        if "maxIter" in hyperparameters and hyperparameters["maxIter"] <= 0:
            raise ValueError("maxIter must be > 0")


# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def regression_run_linear(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Linear Regression (Ridge/Lasso/ElasticNet) with proper MLflow run management"""
    
    # ← KEY: Close any previous run before starting a new one
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxIter": 400, "regParam": 0.0, "elasticNetParam": 0.0, "tol": 1e-6, "fitIntercept": True}
    
    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_linear_regression(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    model = ModelFactory.create_linear_regression(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        # ← KEY: Use try/finally to ensure run is closed
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        # ← KEY: Always close the run
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def regression_decision_tree(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Decision Tree Regressor with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxDepth": 5, "minInstancesPerNode": 1, "seed": 42}
    
    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    model = ModelFactory.create_decision_tree_regressor(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "DecisionTreeRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def regression_random_forest(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Random Forest Regressor with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"numTrees": 20, "maxDepth": 5, "minInstancesPerNode": 1, "subsamplingRate": 1.0}
    
    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    model = ModelFactory.create_random_forest_regressor(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def regression_gbt_regressor(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Gradient Boosting Regressor with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxIter": 20, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 1.0}
    
    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    model = ModelFactory.create_gradient_boosting_regressor(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GradientBoostingRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classification_logistic_regression(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Logistic Regression Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0, "tol": 1e-6, "fitIntercept": True}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_linear_regression(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_logistic_regression(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_decision_tree(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Decision Tree Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxDepth": 5, "minInstancesPerNode": 1, "impurity": "gini", "seed": 42}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_decision_tree_classifier(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_random_forest(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Random Forest Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"numTrees": 20, "maxDepth": 5, "minInstancesPerNode": 1, "impurity": "gini"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_random_forest_classifier(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_gbt_classifier(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Gradient Boosting Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxIter": 20, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 1.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_gradient_boosting_classifier(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GradientBoostingClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_naive_bayes(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Gaussian Naive Bayes Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"smoothing": 1e-9}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_naive_bayes(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GaussianNB")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_linear_svc(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Linear SVC Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"C": 1.0, "maxIter": 1000, "tol": 1e-3, "seed": 42}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_linear_svc(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.decision_function(X_test)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LinearSVC")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_mlp_classifier(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Multilayer Perceptron Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"layers": (10, 15, 10, 2), "activation": "relu", "solver": "adam", "maxIter": 500}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    model = ModelFactory.create_mlp_classifier(hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "MLPClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def classification_ovr(experiment_type, run_type, dataset_level, metrics, targetColumn, base_classifier="LogisticRegression", hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """One-vs-Rest Classifier with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"maxIter": 1000, "regParam": 0.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    df = _load_and_preprocess_classification_data(data_path)
    X = df.drop(targetColumn, axis=1)
    y = df[targetColumn]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    model = ModelFactory.create_ovr_classifier(base_classifier, hyperparameters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X_train.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", f"OneVsRest_{base_classifier}")
            mlflow.log_param("base_classifier", base_classifier)
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

def clustering_kmeans(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """K-Means Clustering with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "tol": 1e-4, "n_init": 10}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ModelFactory.create_kmeans(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "KMeans")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_bisecting_kmeans(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Bisecting K-Means Clustering with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "tol": 1e-4, "n_init": 10}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ModelFactory.create_bisecting_kmeans(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "BisectingKMeans")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_gaussian_mixture(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Gaussian Mixture Model Clustering with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 100, "tol": 0.01, "covarianceType": "full"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ModelFactory.create_gaussian_mixture(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GaussianMixture")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_lda(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """Latent Dirichlet Allocation Clustering with proper MLflow run management"""
    _ensure_run_closed()
    
    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "optimizer": "online"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X = df.select_dtypes(include=[np.number])
    X_scaled = X.clip(lower=0)

    model = ModelFactory.create_lda(hyperparameters)
    topic_distributions = model.fit_transform(X_scaled)
    labels = topic_distributions.argmax(axis=1)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = X.iloc[:5]

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LDA")
            mlflow.log_param("dataset_level", dataset_level)
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _load_and_preprocess_classification_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess classification data"""
    df = pd.read_csv(data_path)
    df = df.copy()

    # Encode categorical columns if they exist
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Target':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))

    return df


def _compute_regression_metrics(y_test, y_pred, metrics: List[str]) -> Dict[str, float]:
    """Compute regression metrics"""
    metrics_dict = {}

    if "MAE" in metrics:
        metrics_dict["MAE"] = mean_absolute_error(y_test, y_pred)
    if "MSE" in metrics:
        metrics_dict["MSE"] = mean_squared_error(y_test, y_pred)
    if "RMSE" in metrics:
        mse = metrics_dict.get("MSE") or mean_squared_error(y_test, y_pred)
        metrics_dict["RMSE"] = np.sqrt(mse)
    if "R2 Score" in metrics:
        metrics_dict["R2 Score"] = r2_score(y_test, y_pred)

    return metrics_dict


def _compute_classification_metrics(y_test, y_pred, y_proba, metrics: List[str]) -> Dict[str, float]:
    """Compute classification metrics"""
    metrics_dict = {}

    if "Accuracy" in metrics:
        metrics_dict["Accuracy"] = accuracy_score(y_test, y_pred)
    if "Precision" in metrics:
        metrics_dict["Precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    if "Recall" in metrics:
        metrics_dict["Recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    if "F1 Score" in metrics:
        metrics_dict["F1 Score"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    if "AUC ROC" in metrics:
        try:
            metrics_dict["AUC ROC"] = roc_auc_score(y_test, y_proba)
        except:
            metrics_dict["AUC ROC"] = 0.0

    return metrics_dict


def _compute_clustering_metrics(X, labels, metrics: List[str]) -> Dict[str, float]:
    """Compute clustering metrics"""
    metrics_dict = {}

    if "Silhouette Score" in metrics:
        metrics_dict["Silhouette Score"] = silhouette_score(X, labels)
    if "Davies Bouldin Score" in metrics:
        metrics_dict["Davies Bouldin Score"] = davies_bouldin_score(X, labels)
    if "Calinski Harabasz Score" in metrics:
        metrics_dict["Calinski Harabasz Score"] = calinski_harabasz_score(X, labels)

    return metrics_dict


def _log_hyperparameters(hyperparameters: Dict[str, Any]):
    """Log hyperparameters to MLflow"""
    for param_name, param_value in hyperparameters.items():
        # Convert lists/tuples to strings for logging
        if isinstance(param_value, (list, tuple)):
            param_value = str(param_value)
        mlflow.log_param(f"hyperparameter_{param_name}", param_value)


def _log_metrics(metrics_dict: Dict[str, float]):
    """Log metrics to MLflow"""
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, float(metric_value))
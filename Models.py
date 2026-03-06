"""
Complete ML Training Module with Hyperparameter Support
Supports: Regression, Classification, and Clustering models
Backend: scikit-learn
All models use user-provided hyperparameters
With proper MLflow run management for sequential training
Feature Engineering: One Hot Encoding for categorical variables
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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


atexit.register(_cleanup_mlflow_on_exit)


# ============================================================================
# ONE HOT ENCODING UTILITIES
# ============================================================================

def _build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies One Hot Encoding to categorical
    columns and optionally StandardScaler to numeric columns.

    Args:
        X: Feature DataFrame
        scale_numeric: If True, apply StandardScaler to numeric columns
                       (recommended for SVC, MLP, Logistic Regression)

    Returns:
        ColumnTransformer instance (not yet fitted)
    """
    # Use explicit dtype list for pandas 4 compatibility (avoids DeprecationWarning)
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  → Numeric columns  ({len(numeric_cols)}): {numeric_cols}")
    print(f"  → Categorical cols ({len(categorical_cols)}): {categorical_cols}")

    transformers = []

    if numeric_cols:
        if scale_numeric:
            transformers.append(("num", StandardScaler(), numeric_cols))
        else:
            transformers.append(("num", "passthrough", numeric_cols))

    if categorical_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", ohe, categorical_cols))

    if not transformers:
        # Fallback: passthrough everything
        transformers.append(("passthrough", "passthrough", X.columns.tolist()))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_numeric: bool = False
) -> Tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Fit preprocessor on training data and transform both train and test sets.
    One Hot Encodes all categorical columns; optionally scales numeric columns.

    Returns:
        X_train_transformed, X_test_transformed, fitted_preprocessor
    """
    preprocessor = _build_preprocessor(X_train, scale_numeric=scale_numeric)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Log OHE feature info
    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if cat_cols:
        ohe = preprocessor.named_transformers_.get("cat")
        if ohe and hasattr(ohe, "get_feature_names_out"):
            ohe_feature_names = ohe.get_feature_names_out(cat_cols)
            print(f"  → OHE created {len(ohe_feature_names)} features from {len(cat_cols)} categorical columns")

    return X_train_transformed, X_test_transformed, preprocessor


def _load_and_preprocess_data(data_path: str, target_col: str, scale_numeric: bool = False):
    """
    Generic loader for supervised models (regression + classification).
    Applies One Hot Encoding to all categorical feature columns.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor, splitRatio-ready splits
    """
    df = pd.read_csv(data_path)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if it is categorical (classification use case)
    if y.dtype == object or str(y.dtype) in ("category", "string"):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        print(f"  → Target '{target_col}' label-encoded ({len(le.classes_)} classes)")

    return X, y


def _load_and_preprocess_classification_data(data_path: str) -> pd.DataFrame:
    """
    Legacy helper kept for backward compatibility.
    Loads data and label-encodes the target column only.
    The actual OHE of features is now handled per-model via _preprocess_features().
    """
    df = pd.read_csv(data_path)
    df = df.copy()
    df = df.fillna(df.mean(numeric_only=True))
    return df


# ============================================================================
# MODEL FACTORY FUNCTIONS
# ============================================================================

class ModelFactory:
    """Factory class to create models with hyperparameters"""

    @staticmethod
    def create_linear_regression(hyperparameters: Dict[str, Any]):
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
        return DecisionTreeRegressor(
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            random_state=hyperparameters.get("seed", 42),
            splitter=hyperparameters.get("splitter", "best")
        )

    @staticmethod
    def create_random_forest_regressor(hyperparameters: Dict[str, Any]):
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
        return GradientBoostingRegressor(
            n_estimators=hyperparameters.get("maxIter", 20),
            max_depth=hyperparameters.get("maxDepth", 5),
            learning_rate=hyperparameters.get("stepSize", 0.1),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            min_samples_split=hyperparameters.get("minInstancesPerNode", 2),
            subsample=hyperparameters.get("subsamplingRate", 1.0),
            loss=hyperparameters.get("lossType", "squared_error"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_logistic_regression(hyperparameters: Dict[str, Any]):
        regParam = hyperparameters.get("regParam", 0.0)
        elasticNetParam = hyperparameters.get("elasticNetParam", 0.0)
        maxIter = hyperparameters.get("maxIter", 100)
        tol = hyperparameters.get("tol", 1e-6)
        fitIntercept = hyperparameters.get("fitIntercept", True)

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
        return DecisionTreeClassifier(
            max_depth=hyperparameters.get("maxDepth", 5),
            min_samples_leaf=hyperparameters.get("minInstancesPerNode", 1),
            criterion=hyperparameters.get("impurity", "gini"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_random_forest_classifier(hyperparameters: Dict[str, Any]):
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
        return GaussianNB(var_smoothing=hyperparameters.get("smoothing", 1e-9))

    @staticmethod
    def create_linear_svc(hyperparameters: Dict[str, Any]):
        return LinearSVC(
            C=hyperparameters.get("C", 1.0),
            max_iter=hyperparameters.get("maxIter", 1000),
            tol=hyperparameters.get("tol", 1e-3),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_mlp_classifier(hyperparameters: Dict[str, Any]):
        """MLP — scaling is handled upstream via _preprocess_features(scale_numeric=True)"""
        layers = hyperparameters.get("layers", (100, 50))

        if isinstance(layers, str):
            import json, ast
            try:
                layers = tuple(json.loads(layers))
            except (json.JSONDecodeError, ValueError):
                try:
                    layers = tuple(ast.literal_eval(layers))
                except (ValueError, SyntaxError):
                    layers = (100, 50)

        return MLPClassifier(
            hidden_layer_sizes=tuple(layers[1:-1]) if len(layers) > 2 else tuple(layers),
            activation=hyperparameters.get("activation", "relu"),
            solver=hyperparameters.get("solver", "adam"),
            max_iter=hyperparameters.get("maxIter", 500),
            tol=hyperparameters.get("tol", 1e-4),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_ovr_classifier(base_classifier_name: str, hyperparameters: Dict[str, Any]):
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
        return KMeans(
            n_clusters=hyperparameters.get("k", 2),
            max_iter=hyperparameters.get("maxIter", 20),
            tol=hyperparameters.get("tol", 1e-4),
            n_init=hyperparameters.get("n_init", 10),
            random_state=hyperparameters.get("seed", 42),
            algorithm="lloyd"
        )

    @staticmethod
    def create_gaussian_mixture(hyperparameters: Dict[str, Any]):
        return GaussianMixture(
            n_components=hyperparameters.get("k", 2),
            max_iter=hyperparameters.get("maxIter", 100),
            tol=hyperparameters.get("tol", 1e-3),
            covariance_type=hyperparameters.get("covarianceType", "full"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_lda(hyperparameters: Dict[str, Any]):
        return LatentDirichletAllocation(
            n_components=hyperparameters.get("k", 10),
            max_iter=hyperparameters.get("maxIter", 20),
            learning_method=hyperparameters.get("optimizer", "online"),
            random_state=hyperparameters.get("seed", 42)
        )

    @staticmethod
    def create_bisecting_kmeans(hyperparameters: Dict[str, Any]):
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

    @staticmethod
    def convert_hyperparameters(params):
        converted = {}
        for key, value in params.items():
            if isinstance(value, str):
                if value.lower() == 'true':
                    converted[key] = True
                elif value.lower() == 'false':
                    converted[key] = False
                else:
                    try:
                        converted[key] = float(value) if '.' in str(value) else int(value)
                    except ValueError:
                        converted[key] = value
            else:
                converted[key] = value
        return converted

    @staticmethod
    def validate_linear_regression(hyperparameters):
        if "regParam" in hyperparameters and hyperparameters["regParam"] < 0:
            raise ValueError("regParam must be >= 0")
        if "elasticNetParam" in hyperparameters:
            param = hyperparameters["elasticNetParam"]
            if not (0 <= param <= 1):
                raise ValueError("elasticNetParam must be between 0 and 1")
        if "maxIter" in hyperparameters and hyperparameters["maxIter"] <= 0:
            raise ValueError("maxIter must be > 0")

    @staticmethod
    def validate_tree_hyperparameters(hyperparameters):
        if "maxDepth" in hyperparameters and hyperparameters["maxDepth"] <= 0:
            raise ValueError("maxDepth must be > 0")
        if "minInstancesPerNode" in hyperparameters and hyperparameters["minInstancesPerNode"] < 1:
            raise ValueError("minInstancesPerNode must be >= 1")

    @staticmethod
    def validate_ensemble_hyperparameters(hyperparameters):
        HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)
        if "numTrees" in hyperparameters and hyperparameters["numTrees"] <= 0:
            raise ValueError("numTrees must be > 0")
        if "subsamplingRate" in hyperparameters:
            rate = hyperparameters["subsamplingRate"]
            if not (0 < rate <= 1):
                raise ValueError("subsamplingRate must be between 0 and 1")

    @staticmethod
    def validate_clustering_hyperparameters(hyperparameters):
        if "k" in hyperparameters and hyperparameters["k"] <= 0:
            raise ValueError("k must be > 0")
        if "maxIter" in hyperparameters and hyperparameters["maxIter"] <= 0:
            raise ValueError("maxIter must be > 0")


# ============================================================================
# REGRESSION FUNCTIONS  (all now use OHE via _preprocess_features)
# ============================================================================

def regression_run_linear(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxIter": 400, "regParam": 0.0, "elasticNetParam": 0.0, "tol": 1e-6, "fitIntercept": True}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_linear_regression(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_linear_regression(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            mlflow.log_param("Target Column", targetColumn)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def regression_decision_tree(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxDepth": 5, "minInstancesPerNode": 1, "seed": 42}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_decision_tree_regressor(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "DecisionTreeRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"numTrees": 20, "maxDepth": 5, "minInstancesPerNode": 1, "subsamplingRate": 1.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_random_forest_regressor(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxIter": 20, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 1.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_gradient_boosting_regressor(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    metrics_dict = _compute_regression_metrics(y_test, y_pred, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GradientBoostingRegressor")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
# CLASSIFICATION FUNCTIONS  (all now use OHE via _preprocess_features)
# ============================================================================

def classification_logistic_regression(experiment_type, run_type, dataset_level, metrics, targetColumn, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxIter": 100, "regParam": 0.0, "elasticNetParam": 0.0, "tol": 1e-6, "fitIntercept": True}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_linear_regression(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING + scaling (important for Logistic Regression)
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=True)

    model = ModelFactory.create_logistic_regression(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_t)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxDepth": 5, "minInstancesPerNode": 1, "impurity": "gini", "seed": 42}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_tree_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_decision_tree_classifier(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "DecisionTreeClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"numTrees": 20, "maxDepth": 5, "minInstancesPerNode": 1, "impurity": "gini"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_random_forest_classifier(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxIter": 20, "maxDepth": 5, "stepSize": 0.1, "subsamplingRate": 1.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_ensemble_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_gradient_boosting_classifier(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GradientBoostingClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"smoothing": 1e-9}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING applied here
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=False)

    model = ModelFactory.create_naive_bayes(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GaussianNB")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"C": 1.0, "maxIter": 1000, "tol": 1e-3, "seed": 42}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING + scaling (important for SVC)
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=True)

    model = ModelFactory.create_linear_svc(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.decision_function(X_test_t)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LinearSVC")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"layers": (100, 50), "activation": "relu", "solver": "adam", "maxIter": 500}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42)

    # ← ONE HOT ENCODING + scaling (critical for MLP)
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=True)

    model = ModelFactory.create_mlp_classifier(hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "MLPClassifier")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"maxIter": 1000, "regParam": 0.0}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)

    X, y = _load_and_preprocess_data(data_path, targetColumn)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitRatio, random_state=42, stratify=y)

    # ← ONE HOT ENCODING + scaling
    X_train_t, X_test_t, preprocessor = _preprocess_features(X_train, X_test, scale_numeric=True)

    model = ModelFactory.create_ovr_classifier(base_classifier, hyperparameters)
    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test_t)

    metrics_dict = _compute_classification_metrics(y_test, y_pred, y_proba, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_train_t[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", f"OneVsRest_{base_classifier}")
            mlflow.log_param("base_classifier", base_classifier)
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_train.select_dtypes(include=["object","string","category"]).columns.tolist()))
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
# CLUSTERING FUNCTIONS  (OHE + scaling applied before clustering)
# ============================================================================

def _prepare_clustering_features(data_path: str) -> np.ndarray:
    """
    Load CSV, apply One Hot Encoding to categorical columns,
    then StandardScale all features for clustering.
    Returns transformed numpy array ready for clustering algorithms.
    """
    df = pd.read_csv(data_path)

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"  → Clustering OHE: {len(categorical_cols)} categorical cols, {len(numeric_cols)} numeric cols")

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    if not transformers:
        return df.values

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_transformed = preprocessor.fit_transform(df)

    if categorical_cols:
        ohe = preprocessor.named_transformers_.get("cat")
        if ohe and hasattr(ohe, "get_feature_names_out"):
            ohe_names = ohe.get_feature_names_out(categorical_cols)
            print(f"  → OHE expanded {len(categorical_cols)} categorical cols → {len(ohe_names)} binary features")

    return X_transformed


def clustering_kmeans(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "tol": 1e-4, "n_init": 10}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    # ← ONE HOT ENCODING + scaling applied here
    X_scaled = _prepare_clustering_features(data_path)
    X_df = pd.read_csv(data_path)

    model = ModelFactory.create_kmeans(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_scaled[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "KMeans")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_df.select_dtypes(include=["object","string","category"]).columns.tolist()))
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_bisecting_kmeans(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "tol": 1e-4, "n_init": 10}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    # ← ONE HOT ENCODING + scaling applied here
    X_scaled = _prepare_clustering_features(data_path)
    X_df = pd.read_csv(data_path)

    model = ModelFactory.create_bisecting_kmeans(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_scaled[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "BisectingKMeans")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_df.select_dtypes(include=["object","string","category"]).columns.tolist()))
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_gaussian_mixture(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 100, "tol": 0.01, "covarianceType": "full"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    # ← ONE HOT ENCODING + scaling applied here
    X_scaled = _prepare_clustering_features(data_path)
    X_df = pd.read_csv(data_path)

    model = ModelFactory.create_gaussian_mixture(hyperparameters)
    labels = model.fit_predict(X_scaled)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_scaled[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "GaussianMixture")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding+StandardScaler")
            mlflow.log_param("ohe_categorical_cols", str(X_df.select_dtypes(include=["object","string","category"]).columns.tolist()))
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


def clustering_lda(experiment_type, run_type, dataset_level, metrics, hyperparameters=None, splitRatio=0.2, data_path="data.csv"):
    """
    LDA requires non-negative inputs, so we apply OHE to categorical columns
    and clip numeric columns to >= 0 (no StandardScaler here as LDA needs counts).
    """
    _ensure_run_closed()

    if hyperparameters is None:
        hyperparameters = {"k": 3, "maxIter": 20, "optimizer": "online"}

    hyperparameters = HyperparameterValidator.convert_hyperparameters(hyperparameters)
    HyperparameterValidator.validate_clustering_hyperparameters(hyperparameters)

    df = pd.read_csv(data_path)
    X_df = df.copy()

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if numeric_cols:
        # Clip negatives — LDA requires non-negative values
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        X_scaled = preprocessor.fit_transform(df)
    else:
        X_scaled = df.values

    # Ensure non-negative for LDA
    X_scaled = np.clip(X_scaled, 0, None)

    model = ModelFactory.create_lda(hyperparameters)
    topic_distributions = model.fit_transform(X_scaled)
    labels = topic_distributions.argmax(axis=1)

    metrics_dict = _compute_clustering_metrics(X_scaled, labels, metrics)

    mlflow.set_experiment(experiment_type)
    input_example = pd.DataFrame(X_scaled[:5])

    run_id = None
    try:
        with mlflow.start_run(run_name=run_type) as run:
            mlflow.log_param("model_type", "LDA")
            mlflow.log_param("dataset_level", dataset_level)
            mlflow.log_param("feature_engineering", "OneHotEncoding (non-negative clipped)")
            mlflow.log_param("ohe_categorical_cols", str(X_df.select_dtypes(include=["object","string","category"]).columns.tolist()))
            _log_hyperparameters(hyperparameters)
            _log_metrics(metrics_dict)
            mlflow.log_artifact(data_path, artifact_path="data")
            mlflow.sklearn.log_model(model, name="model", input_example=input_example)
            run_id = run.info.run_id
    finally:
        _ensure_run_closed()

    return {"run_id": run_id, "metrics": metrics_dict}


# ============================================================================
# METRIC COMPUTATION UTILITIES
# ============================================================================

def _compute_regression_metrics(y_test, y_pred, metrics: List[str]) -> Dict[str, float]:
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
    metrics_dict = {}
    if "Silhouette Score" in metrics:
        metrics_dict["Silhouette Score"] = silhouette_score(X, labels)
    if "Davies Bouldin Score" in metrics:
        metrics_dict["Davies Bouldin Score"] = davies_bouldin_score(X, labels)
    if "Calinski Harabasz Score" in metrics:
        metrics_dict["Calinski Harabasz Score"] = calinski_harabasz_score(X, labels)
    return metrics_dict


def _log_hyperparameters(hyperparameters: Dict[str, Any]):
    for param_name, param_value in hyperparameters.items():
        if isinstance(param_value, (list, tuple)):
            param_value = str(param_value)
        mlflow.log_param(f"hyperparameter_{param_name}", param_value)


def _log_metrics(metrics_dict: Dict[str, float]):
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, float(metric_value))
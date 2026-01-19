# ==================================
# Gradient Boosting Regressor (CSV)
# ==================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.data

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data.csv")

X = df.drop("MEDV", axis=1)   # CHANGE MEDV column name if needed
y = df["MEDV"]

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
mlflow.set_experiment("Regression")

input_example = X_train.iloc[:5]

with mlflow.start_run(run_name="Gradient_Boosting_Regressor"):
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

print("Gradient Boosting Regressor training & logging complete.")

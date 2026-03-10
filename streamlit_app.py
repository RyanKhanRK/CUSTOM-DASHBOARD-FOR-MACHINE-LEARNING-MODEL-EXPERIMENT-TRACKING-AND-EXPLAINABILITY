import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os
import tempfile
import matplotlib.pyplot as plt
import shap
import Models  # Your existing Models.py file

# Set page configuration
st.set_page_config(page_title="Low-Code ML Dashboard", page_icon="🚀", layout="wide")

# Connect to local MLflow tracking server
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ==========================================
# HYPERPARAMETER DEFINITIONS (From hyperparameters.js)
# ==========================================
HYPERPARAMETERS = {
    'Regression': {
        'LinearRegression': {
            'maxIter': {'type': 'int', 'min': 50, 'max': 300, 'default': 100},
            'regParam': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.0},
            'elasticNetParam': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.0},
            'fitIntercept': {'type': 'bool', 'default': True}
        },
        'RandomForest': {
            'numTrees': {'type': 'int', 'min': 10, 'max': 200, 'default': 20},
            'maxDepth': {'type': 'int', 'min': 2, 'max': 30, 'default': 5},
            'minInstancesPerNode': {'type': 'int', 'min': 1, 'max': 20, 'default': 1}
        },
        'DecisionTree': {
            'maxDepth': {'type': 'int', 'min': 2, 'max': 30, 'default': 5},
            'minInstancesPerNode': {'type': 'int', 'min': 1, 'max': 20, 'default': 1}
        },
        'GBTRegressor': {
            'maxIter': {'type': 'int', 'min': 20, 'max': 200, 'default': 20},
            'maxDepth': {'type': 'int', 'min': 3, 'max': 15, 'default': 5},
            'stepSize': {'type': 'float', 'min': 0.01, 'max': 0.5, 'default': 0.1}
        }
    },
    'Classification': {
        'LogisticRegression': {
            'maxIter': {'type': 'int', 'min': 50, 'max': 300, 'default': 100},
            'regParam': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.0},
            'elasticNetParam': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.0}
        },
        'RandomForest': {
            'numTrees': {'type': 'int', 'min': 10, 'max': 200, 'default': 20},
            'maxDepth': {'type': 'int', 'min': 2, 'max': 30, 'default': 5}
        },
        'DecisionTree': {
            'maxDepth': {'type': 'int', 'min': 2, 'max': 30, 'default': 5}
        },
        'GBTClassifier': {
            'maxIter': {'type': 'int', 'min': 20, 'max': 200, 'default': 20},
            'maxDepth': {'type': 'int', 'min': 3, 'max': 15, 'default': 5}
        },
        'NaiveBayes': {
            'smoothing': {'type': 'float', 'min': 1e-10, 'max': 1e-5, 'default': 1e-9}
        }
    },
    'Clustering': {
        'KMeans': {
            'k': {'type': 'int', 'min': 2, 'max': 20, 'default': 3},
            'maxIter': {'type': 'int', 'min': 10, 'max': 100, 'default': 20}
        },
        'GaussianMixture': {
            'k': {'type': 'int', 'min': 2, 'max': 20, 'default': 3},
            'maxIter': {'type': 'int', 'min': 10, 'max': 200, 'default': 100}
        }
    }
}

AVAILABLE_METRICS = {
    'Regression': ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    'Classification': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Clustering': ['Silhouette Score', 'Davies Bouldin Score', 'Calinski Harabasz Score']
}

# ==========================================
# HELPER FUNCTIONS (From model_server.py)
# ==========================================
def preprocess_dataframe(df, prep_option, split_type):
    # Missing Values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        if prep_option == 'remove':
            df = df.dropna()
        elif prep_option == 'fill':
            num_cols = df.select_dtypes(include=['number']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            cat_cols = df.select_dtypes(exclude=['number']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Outlier Removal (IQR)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        outlier_mask = outlier_mask | col_outliers
        
    total_outlier_cells = outlier_mask.sum()
    if total_outlier_cells > 100:
        df = df[~outlier_mask].reset_index(drop=True)

    # Shuffling
    if split_type == 'random':
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
    return df

def determine_dataset_level(df):
    has_numeric = any(df[c].dtype in ['int64', 'float64'] for c in df.columns)
    has_string = any(df[c].dtype == 'object' for c in df.columns)
    return "LEVEL 1" if has_numeric and not has_string else "LEVEL 2"

# ==========================================
# PAGE VIEWS
# ==========================================
def batch_training_page():
    st.title("🚀 Batch Model Training")
    st.markdown("Select models, configure metrics, and run experiments simultaneously.")

    # File Upload
    uploaded_file = st.file_uploader("Step 1: Upload Dataset (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
        
        # Determine Problem Type
        model_type = st.radio("Step 2: Select Problem Type", ["Regression", "Classification", "Clustering"], horizontal=True)
        
        # Target Column
        target_col = None
        if model_type in ["Regression", "Classification"]:
            target_col = st.selectbox("Select Target Column", df.columns)
            
        # Preprocessing Options
        st.subheader("Step 3: Preprocessing & Splitting")
        col1, col2, col3 = st.columns(3)
        with col1:
            prep_option = st.radio("Missing Value Handling", ["remove", "fill"], format_func=lambda x: "Remove Rows" if x == 'remove' else "Fill with Mean/Mode")
        with col2:
            split_type = st.radio("Data Split Method", ["random", "sequential"])
        with col3:
            split_ratio = st.slider("Test Split Ratio", 0.05, 0.50, 0.20, 0.05)

        # Model & Metric Selection
        st.subheader("Step 4: Select Models & Metrics")
        selected_models = st.multiselect("Select Models to Train", list(HYPERPARAMETERS[model_type].keys()))
        selected_metrics = st.multiselect("Select Metrics to Track", AVAILABLE_METRICS[model_type], default=AVAILABLE_METRICS[model_type])

        # Hyperparameters
        if selected_models:
            st.subheader("Step 5: Configure Hyperparameters")
            hyperparam_configs = {}
            for model in selected_models:
                with st.expander(f"⚙️ {model} Hyperparameters"):
                    hyperparam_configs[model] = {}
                    for param, config in HYPERPARAMETERS[model_type][model].items():
                        if config['type'] == 'int':
                            hyperparam_configs[model][param] = st.slider(param, config['min'], config['max'], config['default'], key=f"{model}_{param}")
                        elif config['type'] == 'float':
                            hyperparam_configs[model][param] = st.slider(param, float(config['min']), float(config['max']), float(config['default']), key=f"{model}_{param}")
                        elif config['type'] == 'bool':
                            hyperparam_configs[model][param] = st.checkbox(param, value=config['default'], key=f"{model}_{param}")

        # Execution
        if st.button("🚀 Run Models", type="primary", use_container_width=True):
            if not selected_models:
                st.error("Please select at least one model.")
                return

            with st.spinner("Preprocessing data and training models..."):
                # Preprocess and save to temp file for Models.py
                processed_df = preprocess_dataframe(df, prep_option, split_type)
                dataset_level = determine_dataset_level(processed_df)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    processed_df.to_csv(tmp.name, index=False)
                    temp_path = tmp.name

                # Run Models based on selection
                for model in selected_models:
                    st.write(f"Training {model}...")
                    h_params = hyperparam_configs[model]
                    
                    try:
                        if model_type == "Regression":
                            if model == "LinearRegression":
                                Models.regression_run_linear(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "RandomForest":
                                Models.regression_random_forest(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "DecisionTree":
                                Models.regression_decision_tree(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "GBTRegressor":
                                Models.regression_gbt_regressor(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                                
                        elif model_type == "Classification":
                            if model == "LogisticRegression":
                                Models.classification_logistic_regression(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "RandomForest":
                                Models.classification_random_forest(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "DecisionTree":
                                Models.classification_decision_tree(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "GBTClassifier":
                                Models.classification_gbt_classifier(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                            elif model == "NaiveBayes":
                                Models.classification_naive_bayes(model_type, model, dataset_level, selected_metrics, target_col, h_params, split_ratio, temp_path)
                                
                        elif model_type == "Clustering":
                            if model == "KMeans":
                                Models.clustering_kmeans(model_type, model, dataset_level, selected_metrics, h_params, split_ratio, temp_path)
                            elif model == "GaussianMixture":
                                Models.clustering_gaussian_mixture(model_type, model, dataset_level, selected_metrics, h_params, split_ratio, temp_path)
                    
                    except Exception as e:
                        st.error(f"Error training {model}: {str(e)}")
                
                os.remove(temp_path)
                st.success("All selected models trained and logged to MLflow successfully!")

def experiments_page():
    st.title("📊 MLflow Experiments & Runs")
    
    try:
        experiments = mlflow.search_experiments()
        if not experiments:
            st.info("No experiments found. Run some models first!")
            return
            
        exp_names = {exp.experiment_id: exp.name for exp in experiments}
        selected_exp_id = st.selectbox("Select Experiment", options=list(exp_names.keys()), format_func=lambda x: exp_names[x])
        
        runs = mlflow.search_runs(experiment_ids=[selected_exp_id])
        
        if not runs.empty:
            st.dataframe(runs[['run_id', 'status', 'start_time', 'tags.mlflow.runName'] + [c for c in runs.columns if c.startswith('metrics.')]], use_container_width=True)
        else:
            st.info("No runs found for this experiment.")
            
    except Exception as e:
        st.error(f"Could not connect to MLflow. Make sure the server is running on {MLFLOW_TRACKING_URI}.")

def shap_analysis_page():
    st.title("🔍 SHAP Analysis Dashboard")
    st.markdown("Upload a dataset to generate SHAP feature importance visualizations on the fly.")
    
    uploaded_file = st.file_uploader("Upload Data for SHAP (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Basic mock SHAP generation based on your JS logic (normalized deviation from mean)
        # Note: In a real app, you would load the actual trained model here.
        st.write("Generating approximate SHAP values based on data variance for visualization...")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'species' in numeric_cols: numeric_cols.remove('species') # example exclusion
        
        if len(numeric_cols) > 0:
            # Generate summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create dummy SHAP values based on standard deviation
            shap_values = np.zeros((len(df), len(numeric_cols)))
            for i, col in enumerate(numeric_cols):
                mean = df[col].mean()
                std = df[col].std()
                shap_values[:, i] = ((df[col] - mean) / std) * 0.1 if std > 0 else 0
                
            shap.summary_plot(shap_values, df[numeric_cols], show=False)
            st.pyplot(fig)
            
            # Bar chart
            st.subheader("Feature Importance (Bar)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, df[numeric_cols], plot_type="bar", show=False)
            st.pyplot(fig2)
        else:
            st.warning("No numeric columns found for SHAP analysis.")

# ==========================================
# MAIN APP ROUTING
# ==========================================
def main():
    st.sidebar.title("Low-Code ML Dashboard")
    page = st.sidebar.radio("Navigation", ["Batch Training", "Experiments", "SHAP Analysis"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("Make sure MLflow is running in the background:\n\n`mlflow server --host 0.0.0.0 --port 5000`")

    if page == "Batch Training":
        batch_training_page()
    elif page == "Experiments":
        experiments_page()
    elif page == "SHAP Analysis":
        shap_analysis_page()

if __name__ == "__main__":
    main()
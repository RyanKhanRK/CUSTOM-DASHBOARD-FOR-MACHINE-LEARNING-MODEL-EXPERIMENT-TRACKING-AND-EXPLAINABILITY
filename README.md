# LOW-CODE MACHINE LEARNING EXPERIMENT TRACKING & EXPLAINABILITY DASHBOARD

This project implements a lightweight, web-based dashboard integrated with MLflow to support automated machine learning experimentation, experiment tracking, run comparison, and model explainability.

The system is developed for academic and research purposes to demonstrate an end-to-end low-code machine learning workflow, including data ingestion, preprocessing, model execution, evaluation, and explainability within a unified platform.

---

## Project Overview

This project presents a low-code machine learning experiment tracking and explainability dashboard built on top of MLflow.

It extends MLflow’s default capabilities by introducing:

- Dataset ingestion and structural analysis
- User-controlled data preprocessing
- Automated experiment and run creation using MLflow APIs
- Automated model training, evaluation, and logging
- Dataset-aware run comparison (same dataset only)
- Integrated SHAP-based explainability
- A custom web-based dashboard for experiment analysis

The system is designed for academic and research use, focusing on reproducibility, transparency, and reduced manual interaction with MLflow’s default UI.

---

## Implemented Features

## Dataset Ingestion and Preprocessing

1. Supports uploading arbitrary CSV datasets
2. Automatically analyzes dataset structure and feature types
3. Allows users to configure preprocessing options, including:
 - Removing rows with missing values
 - Replacing missing values using feature mean
4. Applies preprocessing consistently across all experiment runs

## Automated Experiment and Run Handling

- Automatically creates MLflow experiments for uploaded datasets
- Automatically generates runs for selected machine learning models
- Handles training, evaluation, and logging programmatically
- Logs parameters, metrics, and artifacts in a structured manner

## Experiment Run Comparison

- Retrieves experiment runs logged in MLflow
- Displays evaluation metrics for each run
- Enables side-by-side comparison between runs
- Identifies the best-performing run based on selected metrics
- Restricts comparison to runs originating from the same dataset

### Model Explainability (SHAP)

- Provides SHAP-based feature contribution analysis
- Visualizes feature importance and influence on predictions
- Supports both global and local interpretability
- Integrated directly into the web dashboard

> **Note:**  
> Run comparison and SHAP explainability are implemented as custom web-based features extending MLflow’s default functionality.

---

## Technology Stack

### Backend / Machine Learning

- Python
- MLflow
- Scikit-learn

### Frontend

- HTML
- CSS
- JavaScript

### Explainability

- SHAP-based feature contribution analysis

### Utilities

- Python HTTP server
- CORS proxy for API communication

---

## Project Structure

├── cors_proxy.py
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
├── mlruns/          # MLflow experiment data
├── shap/            # SHAP-related logic and visualizations
└── README.md

---

## How to Run the Project

### Prerequisites
- Python 3.x
- Virtual environment (recommended)
- MLflow installed

---

### Execution Steps

### 0. Create and Activate a Virtual Environment
Before running the project, create and activate a Python virtual environment to isolate the project dependencies.

Create the virtual environment:
```bash
python3 -m venv venv
```

Activate the virtual environment:
On macOS / Linux:
```bash
source venv/bin/activate
```

On Windows:
```bash
venv\Scripts\activate
```


#### 1. Start the CORS Proxy
```bash
python cors_proxy.py
```

Once activated, install the required dependencies (if applicable):
```bash
pip install -r requirements.txt
```

#### 2. Start the Frontend Server
```bash
python3 -m http.server 8000
```

#### 3. Start the MLflow Tracking Server
```bash
mlflow server --host 0.0.0.0 --port 5000
```

#### 4. Start the Backend Server
```bash
python3 model_server.py
```

## Access the Dashboard
Open a web browser and navigate to:
```bash
http://localhost:8000/
```

## System Workflow
1. Users upload a dataset through the web dashboard
2. The system analyzes dataset structure and feature types
3. Users configure preprocessing options
4. Experiments and runs are automatically created in MLflow
5. Models are trained, evaluated, and logged
6. The dashboard retrieves experiment data via MLflow APIs
7. Users can:
 - Compare multiple experiment runs
 - Identify the best-performing model
 - View SHAP-based explainability visualizations

## Limitations
 - MLflow Model Registry is not implemented
 - Load testing was not conducted
 - SHAP values are calculated using an approximation approach rather than exact game-theoretic Shapley values
 - The system is intended for academic evaluation and demonstration purposes only

## Future Improvements
- Integration with MLflow Model Registry
- Advanced preprocessing pipelines
- Automated fairness and bias evaluation
- Load and performance testing
- Containerized deployment (Docker)
- Enhanced explainability visualizations

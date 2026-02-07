from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import subprocess
import tempfile
import pandas as pd
import shutil
from pathlib import Path
import json
from Models import regression_run_linear, regression_gbt_regressor, regression_decision_tree, regression_random_forest, clustering_bisecting_kmeans, clustering_kmeans, clustering_gaussian_mixture, clustering_lda, classification_decision_tree, classification_random_forest, classification_logistic_regression, classification_gbt_classifier, classification_linear_svc, classification_mlp_classifier, classification_naive_bayes, classification_ovr

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/run-batch-models', methods=['POST'])
def run_multiple_models():
    file = request.files['file']
    models = json.loads(request.form.get('models', '{}'))
    selectedModelType = request.form.get('selectedModelType')
    hyperparameters = json.loads(request.form.get('hyperparameters', '{}'))
    preprocessingOption = request.form.get('preprocessingOption')
    splitRatio = float(request.form.get('splitRatio', 0.2))
    splitType = request.form.get('splitType', "random")
    targetColumn = request.form.get('targetColumn')

    if not selectedModelType:
        return jsonify({'success': False, 'error': 'No model type specified'}), 400
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File must be CSV format'}), 400
    
    if not preprocessingOption or preprocessingOption not in ['remove', 'fill']:
        return jsonify({'success': False, 'error': 'Invalid preprocessing option. Must be "remove" or "fill"'}), 400
    
    if not splitType or splitType not in ['random', 'sequential']:
        return jsonify({'success': False, 'error': 'Invalid split type. Must be "random" or "sequential"'}), 400
        
    if selectedModelType in ['Regression', 'Classification'] and not targetColumn:
        return jsonify({'success': False, 'error': 'No target column specified'}), 400
    
    run_dir = tempfile.mkdtemp()
    
    file_path = os.path.join('', 'data.csv')
    file.save(file_path)

    df = pd.read_csv(file_path)

    null_count = df.isnull().sum().sum()
    
    if null_count > 0:
        if preprocessingOption == 'remove':
            df = df.dropna()
            print(f"Removed rows with null values. New shape: {df.shape}")
        elif preprocessingOption == 'fill':
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                df[col].fillna(df[col].mean(), inplace=True)
            
            non_numeric_columns = df.select_dtypes(exclude=['number']).columns
            for col in non_numeric_columns:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
            
            print(f"Filled null values with mean/mode. Shape: {df.shape}")

    if splitType == 'random':
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Data shuffled randomly for random split")
    elif splitType == 'sequential':
        df = df.reset_index(drop=True)
        print(f"Data kept in original order for sequential split")
    
    # Save the preprocessed data back to CSV
    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")
    # ===== END OF PREPROCESSING SECTION =====

    dataset_level = determine_dataset_level(df)

    if selectedModelType=="Regression":
        for key, value in models.items():
            run_type = key.split('_')[1]
            metrics = value['metrics']
            if run_type=="LinearRegression":
                result = regression_run_linear(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="GBTRegressor":
                result = regression_gbt_regressor(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="RandomForest":
                result = regression_random_forest(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="DecisionTree":
                result = regression_decision_tree(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)

    elif selectedModelType=="Clustering":
        for key, value in models.items():
            run_type = key.split('_')[1]
            metrics = value['metrics']
            if run_type=="BisectingKMeans":
                result = clustering_bisecting_kmeans(selectedModelType, run_type, dataset_level, metrics, hyperparameters[key], splitRatio)
            if run_type=="GaussianMixture":
                result = clustering_gaussian_mixture(selectedModelType, run_type, dataset_level, metrics, hyperparameters[key], splitRatio)
            if run_type=="KMeans":
                result = clustering_kmeans(selectedModelType, run_type, dataset_level, metrics, hyperparameters[key], splitRatio)
            if run_type=="LDA":
                result = clustering_lda(selectedModelType, "Latent_Dirichlet_Allocation", dataset_level, metrics, hyperparameters[key], splitRatio)

    elif selectedModelType=="Classification":
        for key, value in models.items():
            run_type = key.split('_')[1]
            metrics = value['metrics']
            if run_type=="LogisticRegression":
                result = classification_logistic_regression(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="DecisionTree":
                result = classification_decision_tree(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="RandomForest":
                result = classification_random_forest(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="GBTClassifier":
                result = classification_gbt_classifier(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="LinearSVC":
                result = classification_linear_svc(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="MLPC":
                result = classification_mlp_classifier(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="NaiveBayes":
                result = classification_naive_bayes(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key], splitRatio)
            if run_type=="OVR":
                result = classification_ovr(selectedModelType, run_type, dataset_level, metrics, targetColumn, hyperparameters[key]["baseClassifier"], hyperparameters[key], splitRatio)
    
    return jsonify({
        'success': True,
        'message': 'Model training completed successfully',
    }), 200

@app.route('/run-model', methods=['POST'])
def run_model():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    model_type = request.form.get('model_type')
    experiment_type = model_type.split('_')[0]
    run_type = model_type.split('_')[1]
    
    if not model_type:
        return jsonify({'success': False, 'error': 'No model type specified'}), 400
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File must be CSV format'}), 400
    
    run_dir = tempfile.mkdtemp()
    
    file_path = os.path.join('', 'data.csv')
    file.save(file_path)

    df = pd.read_csv(file_path)
    dataset_level = determine_dataset_level(df)
    
    if experiment_type=="Regression" and run_type=="LinearRegression":
        result = run_linear(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="GBTRegressor":
        result = gbt_regressor(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="RandomForest":
        result = regression_random_forest(experiment_type, run_type, dataset_level)
    if experiment_type=="Regression" and run_type=="DecisionTree":
        result = regression_decision_tree(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="BisectingKMeans":
        result = clustering_bisecting_kmeans(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="GaussianMixture":
        result = clustering_gaussian_mixture(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="KMeans":
        result = clustering_kmeans(experiment_type, run_type, dataset_level)
    if experiment_type=="Clustering" and run_type=="LDA":
        result = clustering_lda(experiment_type, "Latent_Dirichlet_Allocation", dataset_level)
    if experiment_type=="Classification" and run_type=="LogisticRegression":
        result = classification_LogisticRegression(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="DecisionTree":
        result = decision_tree(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="RandomForest":
        result = classification_random_forest(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="GBTClassifier":
        result = classification_gbt_classifier(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="LinearSVC":
        result = classification_linear_svc(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="MLPC":
        result = classification_mlp_classifier(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="NaiveBayes":
        result = classification_naive_bayes(experiment_type, run_type, dataset_level)
    if experiment_type=="Classification" and run_type=="OVR":
        result = classification_ovr(experiment_type, run_type, dataset_level)
    
    return jsonify({
        'success': True,
        'message': 'Model training completed successfully',
    }), 200

def determine_dataset_level(df):
    has_numeric = False
    has_string = False
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'int32', 'float64', 'float32']:
            has_numeric = True
        elif df[column].dtype == 'object':
            if df[column].astype(str).str.isalpha().any():
                has_string = True
    
    if has_numeric and not has_string:
        return "LEVEL 1"
    elif has_numeric and has_string:
        return "LEVEL 2"
    else:
        return "LEVEL 2"

if __name__ == '__main__':
    print("🚀 Flask Model Training Server starting on http://localhost:5001")
    print("Make sure MLflow is running on http://localhost:8080")
    app.run(debug=True, port=5001)
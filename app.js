const experimentsData = {
  "experiments": [
    {"id": "run_001", "name": "LogisticRegression_Iris_1", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Iris", "start_time": "2025-08-20 08:47:30", "duration": "87s", "parameters": {"C": 0.71}, "metrics": {"accuracy": 0.7623, "precision": 0.778, "recall": 0.7375, "f1_score": 0.7584, "auc_roc": 0.7538}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_002", "name": "DecisionTree_Diabetes_2", "status": "FINISHED", "user": "Bob", "model_type": "DecisionTree", "dataset": "Diabetes", "start_time": "2025-08-26 11:23:14", "duration": "183s", "parameters": {"max_depth": 8}, "metrics": {"accuracy": 0.8324, "precision": 0.8156, "recall": 0.8654, "f1_score": 0.8226, "auc_roc": 0.8551}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png"]},
    {"id": "run_003", "name": "RandomForest_Titanic_3", "status": "FINISHED", "user": "Charlie", "model_type": "RandomForest", "dataset": "Titanic", "start_time": "2025-08-28 15:39:58", "duration": "244s", "parameters": {"learning_rate": 0.131, "n_estimators": 156, "max_depth": 9}, "metrics": {"accuracy": 0.9018, "precision": 0.9095, "recall": 0.8723, "f1_score": 0.899, "auc_roc": 0.9156}, "fairness_metrics": {"demographic_parity_difference": 0.0742, "equalized_odds_difference": -0.0231, "equal_opportunity_difference": 0.0856}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png", "fairness_report.html"]},
    {"id": "run_004", "name": "XGBoost_Wine Quality_4", "status": "FINISHED", "user": "Diana", "model_type": "XGBoost", "dataset": "Wine Quality", "start_time": "2025-09-01 09:15:42", "duration": "156s", "parameters": {"learning_rate": 0.087, "n_estimators": 312, "max_depth": 11}, "metrics": {"accuracy": 0.8913, "precision": 0.8734, "recall": 0.9156, "f1_score": 0.8876, "auc_roc": 0.8945}, "fairness_metrics": {"demographic_parity_difference": -0.1102, "equalized_odds_difference": 0.0398, "equal_opportunity_difference": -0.0234}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_005", "name": "SVM_Iris_5", "status": "FINISHED", "user": "Eva", "model_type": "SVM", "dataset": "Iris", "start_time": "2025-09-03 14:28:17", "duration": "91s", "parameters": {"C": 4.23, "kernel": "rbf"}, "metrics": {"accuracy": 0.8241, "precision": 0.8456, "recall": 0.7923, "f1_score": 0.8167, "auc_roc": 0.8376}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_006", "name": "LogisticRegression_Diabetes_6", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Diabetes", "start_time": "2025-09-05 12:10:33", "duration": "76s", "parameters": {"C": 2.89}, "metrics": {"accuracy": 0.7923, "precision": 0.8134, "recall": 0.7656, "f1_score": 0.7845, "auc_roc": 0.8123}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png"]},
    {"id": "run_007", "name": "RandomForest_Wine Quality_7", "status": "FINISHED", "user": "Bob", "model_type": "RandomForest", "dataset": "Wine Quality", "start_time": "2025-09-07 16:45:22", "duration": "203s", "parameters": {"learning_rate": 0.234, "n_estimators": 287, "max_depth": 6}, "metrics": {"accuracy": 0.8567, "precision": 0.8789, "recall": 0.8345, "f1_score": 0.8523, "auc_roc": 0.8678}, "fairness_metrics": {"demographic_parity_difference": 0.1234, "equalized_odds_difference": -0.0567, "equal_opportunity_difference": 0.0789}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png", "fairness_report.html"]},
    {"id": "run_016", "name": "RandomForest_Diabetes_16", "status": "FINISHED", "user": "Ryan", "model_type": "RandomForest", "dataset": "Diabetes", "start_time": "2025-09-17 20:08:29", "duration": "211s", "parameters": {"learning_rate": 0.089, "n_estimators": 356, "max_depth": 10}, "metrics": {"accuracy": 0.8567, "precision": 0.8723, "recall": 0.8412, "f1_score": 0.8556, "auc_roc": 0.8678}, "fairness_metrics": {}, "artifacts": ["model.pkl", "feature_importance.png", "confusion_matrix.png", "shap_summary.png"]},
    {"id": "run_018", "name": "LogisticRegression_Titanic_18", "status": "RUNNING", "user": "Charlie", "model_type": "LogisticRegression", "dataset": "Titanic", "start_time": "2025-09-17 23:47:15", "duration": "N/A", "parameters": {"C": 3.45}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []},
    {"id": "run_019", "name": "DecisionTree_Wine Quality_19", "status": "FAILED", "user": "Diana", "model_type": "DecisionTree", "dataset": "Wine Quality", "start_time": "2025-09-17 23:58:42", "duration": "N/A", "parameters": {"max_depth": 3}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []},
    {"id": "run_020", "name": "SVM_Diabetes_20", "status": "RUNNING", "user": "Eva", "model_type": "SVM", "dataset": "Diabetes", "start_time": "2025-09-17 23:59:18", "duration": "N/A", "parameters": {"C": 5.67, "kernel": "rbf"}, "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null, "auc_roc": null}, "fairness_metrics": {}, "artifacts": []}
  ],
  "feature_importance": {
    "run_001": {"feature_importance": {"petal_width": 0.5636, "sepal_length": 0.3756, "sepal_width": 0.0335, "petal_length": 0.0273}, "top_features": [["petal_width", 0.5636], ["sepal_length", 0.3756], ["sepal_width", 0.0335], ["petal_length", 0.0273]], "shap_values": {"petal_width": 0.4234, "sepal_length": -0.2145, "sepal_width": 0.1876, "petal_length": 0.3456}},
    "run_002": {"feature_importance": {"glucose": 0.6234, "bmi": 0.1892, "age": 0.0956, "blood_pressure": 0.0535, "skin_thickness": 0.0246, "insulin": 0.0093, "pregnancies": 0.0044}, "top_features": [["glucose", 0.6234], ["bmi", 0.1892], ["age", 0.0956], ["blood_pressure", 0.0535], ["skin_thickness", 0.0246]], "shap_values": {"glucose": 0.5123, "bmi": -0.1456, "age": 0.2134, "blood_pressure": -0.0876, "skin_thickness": 0.1234}},
    "run_003": {"feature_importance": {"Sex": 0.4567, "Fare": 0.2341, "Age": 0.1456, "Pclass": 0.0823, "SibSp": 0.0456, "Parch": 0.0234, "Embarked": 0.0123}, "top_features": [["Sex", 0.4567], ["Fare", 0.2341], ["Age", 0.1456], ["Pclass", 0.0823], ["SibSp", 0.0456]], "shap_values": {"Sex": 0.3876, "Fare": -0.1987, "Age": 0.2567, "Pclass": -0.1234, "SibSp": 0.0987}},
    "run_004": {"feature_importance": {"alcohol": 0.3456, "volatile_acidity": 0.2123, "citric_acid": 0.1789, "density": 0.1234, "sulphates": 0.0789, "pH": 0.0456, "residual_sugar": 0.0153}, "top_features": [["alcohol", 0.3456], ["volatile_acidity", 0.2123], ["citric_acid", 0.1789], ["density", 0.1234], ["sulphates", 0.0789]], "shap_values": {"alcohol": 0.2876, "volatile_acidity": -0.1654, "citric_acid": 0.1987, "density": -0.0987, "sulphates": 0.1456}},
    "run_005": {"feature_importance": {"petal_length": 0.4123, "petal_width": 0.3456, "sepal_length": 0.1789, "sepal_width": 0.0632}, "top_features": [["petal_length", 0.4123], ["petal_width", 0.3456], ["sepal_length", 0.1789], ["sepal_width", 0.0632]], "shap_values": {"petal_length": 0.3654, "petal_width": -0.2876, "sepal_length": 0.1987, "sepal_width": 0.0876}}
  },
  "shap_data": {
    "base_value": 0.5234,
    "sample_predictions": [
      {"features": {"petal_width": 2.4, "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}, "shap_values": {"petal_width": 0.4234, "sepal_length": -0.2145, "sepal_width": 0.1876, "petal_length": 0.3456}, "prediction": 0.8765},
      {"features": {"glucose": 148, "bmi": 33.6, "age": 50, "blood_pressure": 72, "skin_thickness": 35}, "shap_values": {"glucose": 0.5123, "bmi": -0.1456, "age": 0.2134, "blood_pressure": -0.0876, "skin_thickness": 0.1234}, "prediction": 0.7234}
    ]
  }
};

const experimentRunsData = {
  "609376690833434304": [ // Default experiment
    {"run_id": "run_001", "run_name": "LogisticRegression_Iris_1", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Iris", "start_time": "2025-08-20 08:47:30", "duration": "87s", "metrics": {"accuracy": 0.7623, "precision": 0.778, "recall": 0.7375, "f1_score": 0.7584}},
    {"run_id": "run_002", "run_name": "DecisionTree_Diabetes_2", "status": "FINISHED", "user": "Bob", "model_type": "DecisionTree", "dataset": "Diabetes", "start_time": "2025-08-26 11:23:14", "duration": "183s", "metrics": {"accuracy": 0.8324, "precision": 0.8156, "recall": 0.8654, "f1_score": 0.8226}},
    {"run_id": "run_003", "run_name": "RandomForest_Titanic_3", "status": "FINISHED", "user": "Charlie", "model_type": "RandomForest", "dataset": "Titanic", "start_time": "2025-08-28 15:39:58", "duration": "244s", "metrics": {"accuracy": 0.9018, "precision": 0.9095, "recall": 0.8723, "f1_score": 0.899}}
  ],
  "1": [
    {"run_id": "run_004", "run_name": "XGBoost_Wine Quality_4", "status": "FINISHED", "user": "Diana", "model_type": "XGBoost", "dataset": "Wine Quality", "start_time": "2025-09-01 09:15:42", "duration": "156s", "metrics": {"accuracy": 0.8913, "precision": 0.8734, "recall": 0.9156, "f1_score": 0.8876}},
    {"run_id": "run_005", "run_name": "SVM_Iris_5", "status": "FINISHED", "user": "Eva", "model_type": "SVM", "dataset": "Iris", "start_time": "2025-09-03 14:28:17", "duration": "91s", "metrics": {"accuracy": 0.8241, "precision": 0.8456, "recall": 0.7923, "f1_score": 0.8167}}
  ],
  "2": [
    {"run_id": "run_006", "run_name": "LogisticRegression_Diabetes_6", "status": "FINISHED", "user": "Ryan", "model_type": "LogisticRegression", "dataset": "Diabetes", "start_time": "2025-09-05 12:10:33", "duration": "76s", "metrics": {"accuracy": 0.7923, "precision": 0.8134, "recall": 0.7656, "f1_score": 0.7845}},
    {"run_id": "run_007", "run_name": "RandomForest_Wine Quality_7", "status": "RUNNING", "user": "Bob", "model_type": "RandomForest", "dataset": "Wine Quality", "start_time": "2025-09-07 16:45:22", "duration": "N/A", "metrics": {"accuracy": null, "precision": null, "recall": null, "f1_score": null}}
  ]
};

// Global state
let currentView = 'experiments';
let filteredExperiments = [...experimentsData.experiments];
let selectedExperiments = [];
let compareMode = true;
let charts = {};
let currentModel = null;
let uploadedModelFile = null;
let selectedModelType = null;
let uploadedShapFile = null;
let uploadedData = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  initializeNavigation();
  initializeExperiments();
  setupEventListeners();
  initializeModelRunSection();
  initializeFileUpload();
  window.showView = showView;
  window.showExperimentDetail = showExperimentDetail;
  window.showComparison = showComparison;
  window.clearSelection = clearSelection;
  window.handleExperimentSelection = handleExperimentSelection;
  window.updateCompareMode = updateCompareMode;
  window.loadModel = loadModel;
  window.generateShapAnalysis = generateShapAnalysis;
  window.switchFeatureTab = switchFeatureTab;
  window.runModel = runModel;
  window.initializeArchitecture = initializeArchitecture;
  
  showView('experiments');
});

// Navigation
function initializeNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const newLink = link.cloneNode(true);
    link.parentNode.replaceChild(newLink, link);
  });
  
  // Add fresh listeners
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      
      const view = this.getAttribute('data-view');
      
      if (view) {
        showView(view);
        if (view === 'architecture') {
          setTimeout(() => {
            initializeArchitecture();
          }, 100);
        }
      } else {
        console.error('No view attribute found on navigation link');
      }
    });
  });
}

// Initialize model upload and type selection handlers
function initializeModelRunSection() {
  const modelFilter = document.getElementById('model-filter');
  const modelFileInput = document.getElementById('model-file-input');
  const fileUploadArea = document.getElementById('file-upload-area');
  
  if (modelFilter) {
    modelFilter.addEventListener('change', function(e) {
      selectedModelType = e.target.value;
      updateRunModelButtonState();
    });
  }
  
  // Handle file upload area click
  if (fileUploadArea) {
    fileUploadArea.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      if (modelFileInput) {
        modelFileInput.click();
      }
    });
  }
  
  // Handle file input change
  if (modelFileInput) {
    modelFileInput.addEventListener('change', function(e) {
      if (e.target.files && e.target.files.length > 0) {
        handleModelFileUpload(e.target.files[0]);
      }
    });
  }

  // Handle drag and drop
  if (fileUploadArea) {
    fileUploadArea.addEventListener('dragover', function(e) {
      e.preventDefault();
      e.stopPropagation();
      fileUploadArea.classList.add('dragover');
    });
    
    fileUploadArea.addEventListener('dragleave', function(e) {
      e.preventDefault();
      e.stopPropagation();
      fileUploadArea.classList.remove('dragover');
    });
    
    fileUploadArea.addEventListener('drop', function(e) {
      e.preventDefault();
      e.stopPropagation();
      fileUploadArea.classList.remove('dragover');
      
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleModelFileUpload(e.dataTransfer.files[0]);
      }
    });
  }
}

function handleModelTypeSelection(e) {
  selectedModelType = e.target.value;
  updateRunModelButtonState();
}

function handleModelFileSelect(e) {
  if (e.target.files.length > 0) {
    handleModelFileUpload(e.target.files[0]);
  }
}

function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
}

function handleModelFileDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleModelFileUpload(files[0]);
  }
}

function handleModelFileUpload(file) {
  if (!file.name.endsWith('.csv')) {
    alert('❌ Please upload a CSV file');
    return;
  }
  
  uploadedModelFile = file;
  updateFileUploadUI(file.name);
  updateRunModelButtonState();
}

function updateFileUploadUI(fileName) {
  const fileUploadArea = document.getElementById('file-upload-area');
  if (!fileUploadArea) {
    console.error('File upload area not found');
    return;
  }
  
  fileUploadArea.innerHTML = `
    <div class="upload-icon">✅</div>
    <div class="upload-text">
      <p class="upload-title">File uploaded successfully!</p>
      <p class="upload-subtitle">${fileName}</p>
      <p class="upload-subtitle" style="font-size: 12px; margin-top: 8px; color: #666;">Click to change file</p>
    </div>
    <input type="file" id="model-file-input" accept=".csv" class="file-input hidden" style="display: none;">
  `;
  
  // Re-attach file input listener
  const newFileInput = document.getElementById('model-file-input');
  if (newFileInput) {
    newFileInput.addEventListener('change', function(e) {
      if (e.target.files && e.target.files.length > 0) {
        handleModelFileUpload(e.target.files[0]);
      }
    });
  }
  
  // Re-attach click handler
  const newFileUploadArea = document.getElementById('file-upload-area');
  if (newFileUploadArea) {
    newFileUploadArea.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      const input = document.getElementById('model-file-input');
      if (input) input.click();
    });
  }
}

function updateRunModelButtonState() {
  const runModelBtn = document.querySelector('.model-run-section .btn--primary');
  
  if (runModelBtn) {
    if (uploadedModelFile && selectedModelType) {
      runModelBtn.disabled = false;
      runModelBtn.style.opacity = '1';
      runModelBtn.style.cursor = 'pointer';
    } else {
      runModelBtn.disabled = true;
      runModelBtn.style.opacity = '0.5';
      runModelBtn.style.cursor = 'not-allowed';
    }
  } else {
    console.error('Run model button not found!');
  }
}

// Main function to run the model
async function runModel() {
  if (!uploadedModelFile || !selectedModelType) {
    alert('Please upload a CSV file and select a model type');
    return;
  }
  
  showLoadingModal('Preparing model training...');
  
  try {
    const formData = new FormData();
    formData.append('file', uploadedModelFile, 'data.csv');
    formData.append('model_type', selectedModelType);
    
    // Step 2: Send to backend to execute Python script
    const response = await fetch('http://localhost:5001/run-model', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    hideLoadingModal();
    
    if (result.success) {
      showSuccessMessage(`✅ Model training completed! Run ID: ${result.run_id}`);
      
      // Reset the form
      resetModelRunForm();
      
      // Optionally refresh experiments list
      setTimeout(() => {
        renderExperimentsTable();
      }, 1000);
    } else {
      throw new Error(result.error || 'Unknown error occurred');
    }
    
  } catch (error) {
    hideLoadingModal();
    console.error('Error running model:', error);
    alert(`Error: ${error.message}`);
  }
}

function resetModelRunForm() {
  uploadedModelFile = null;
  selectedModelType = null;
  
  const modelFilter = document.getElementById('model-filter');
  if (modelFilter) modelFilter.value = '';
  
  const fileUploadArea = document.getElementById('file-upload-area');
  if (fileUploadArea) {
    fileUploadArea.innerHTML = `
      <div class="upload-icon">📤</div>
      <div class="upload-text">
        <p class="upload-title">Upload CSV file for Model Run</p>
        <p class="upload-subtitle">Drag and drop your file here or click to browse</p>
      </div>
    `;
  }
  
  const modelFileInput = document.getElementById('model-file-input');
  if (modelFileInput) modelFileInput.value = '';
  
  updateRunModelButtonState();
}

function handleNavClick(e) {
  e.preventDefault();
  e.stopPropagation();
  
  const view = e.currentTarget.getAttribute('data-view');
  
  if (view) {
    showView(view);
  } else {
    console.error('No view attribute found on navigation link');
  }
}

function showView(viewName) {
  const allViews = document.querySelectorAll('.view');
  allViews.forEach(view => {
    view.classList.remove('active');
    view.style.display = 'none'; // Force hide
  });
  
  // Show ONLY the target view
  const targetView = document.getElementById(`${viewName}-view`);
  if (targetView) {
    targetView.classList.add('active');
    targetView.style.display = 'block'; // Force show
  } else {
    console.error(`Could not find view element: ${viewName}-view`);
  }
  
  // Update navigation active state
  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
  });
  
  const activeNavLink = document.querySelector(`.nav-link[data-view="${viewName}"]`);
  if (activeNavLink) {
    activeNavLink.classList.add('active');
  }
}

// SHAP Analysis functionality
function initializeShap() {
  setupShapEventListeners();
  populateRunSelector();
  generateDefaultShapVisualizations();
}

function setupShapEventListeners() {
  
  // File upload
  const fileInput = document.getElementById('file-input');
  const fileUploadArea = document.getElementById('file-upload-area');
  
  if (fileUploadArea && fileInput) {
    // Remove existing listeners first
    fileUploadArea.removeEventListener('click', handleUploadClick);
    fileUploadArea.removeEventListener('dragover', handleDragOver);
    fileUploadArea.removeEventListener('drop', handleFileDrop);
    fileUploadArea.removeEventListener('dragleave', handleDragLeave);
    fileInput.removeEventListener('change', handleFileSelect);
    
    // Add new listeners
    fileUploadArea.addEventListener('click', handleUploadClick);
    fileUploadArea.addEventListener('dragover', handleDragOver);
    fileUploadArea.addEventListener('drop', handleFileDrop);
    fileUploadArea.addEventListener('dragleave', handleDragLeave);
    fileInput.addEventListener('change', handleFileSelect);
  } else {
    console.error('File upload elements not found');
  }

  // Experiment name input
  const experimentNameInput = document.getElementById('experiment-name-input');
  if (experimentNameInput) {
    experimentNameInput.removeEventListener('input', updateRunSelector);
    experimentNameInput.addEventListener('input', updateRunSelector);
  }
}

function handleUploadClick(e) {
  e.preventDefault();
  const fileInput = document.getElementById('model-file-input');
  if (fileInput) {
    fileInput.click();
  }
}

function handleDragOver(e) {
  e.preventDefault();
  e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
}

function handleFileDrop(e) {
  e.preventDefault();
  e.currentTarget.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileUpload(files[0]);
  }
}

function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
}

function handleFile(file) {
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a CSV file');
    return;
  }
  
  // Mock file processing
  const reader = new FileReader();
  reader.onload = function(e) {
    const csv = e.target.result;
    const lines = csv.split('\n');
    const headers = lines[0].split(',');
    const rows = lines.slice(1, 6).map(line => line.split(',')); // Show first 5 rows
    
    displayDataPreview(headers, rows);
    uploadedData = { headers, rows: lines.slice(1).map(line => line.split(',')) };
  };
  reader.readAsText(file);
}

function displayDataPreview(headers, rows) {
  const dataPreview = document.getElementById('data-preview');
  const table = document.getElementById('data-preview-table');
  
  if (dataPreview && table) {
    dataPreview.classList.remove('hidden');
    
    table.innerHTML = `
      <thead>
        <tr>
          ${headers.map(header => `<th>${header.trim()}</th>`).join('')}
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            ${row.map(cell => `<td>${cell.trim()}</td>`).join('')}
          </tr>
        `).join('')}
      </tbody>
    `;
  }
}

function updateRunSelector() {
  const experimentName = document.getElementById('experiment-name-input')?.value;
  const runSelector = document.getElementById('run-selector');
  
  if (!runSelector) return;
  
  if (!experimentName) {
    runSelector.innerHTML = '<option value="">Select a run...</option>';
    return;
  }
  
  // Mock filtering based on experiment name
  const matchingRuns = experimentsData.experiments.filter(exp => 
    exp.name.toLowerCase().includes(experimentName.toLowerCase()) ||
    exp.dataset.toLowerCase().includes(experimentName.toLowerCase()) ||
    exp.model_type.toLowerCase().includes(experimentName.toLowerCase())
  );
  
  runSelector.innerHTML = '<option value="">Select a run...</option>' +
    matchingRuns.map(exp => 
      `<option value="${exp.id}">${exp.name} (${exp.id})</option>`
    ).join('');
}

function populateRunSelector() {
  const runSelector = document.getElementById('run-selector');
  if (!runSelector) return;
  
  runSelector.innerHTML = '<option value="">Select a run...</option>' +
    experimentsData.experiments
      .filter(exp => exp.status === 'FINISHED')
      .map(exp => `<option value="${exp.id}">${exp.name} (${exp.id})</option>`)
      .join('');
}

function loadModel() {
  const runId = document.getElementById('run-selector')?.value;
  const modelStatus = document.getElementById('model-status');
  
  if (!runId) {
    alert('Please select a run first');
    return;
  }
  
  // Mock model loading
  currentModel = experimentsData.experiments.find(exp => exp.id === runId);
  
  if (modelStatus) {
    modelStatus.classList.remove('hidden');
    modelStatus.innerHTML = `
      <div class="status-indicator">
        <span class="status-icon">✅</span>
        <span class="status-text">Model loaded successfully: ${currentModel.name}</span>
      </div>
    `;
  }
}

function generateShapAnalysis() {
  if (!currentModel) {
    alert('Please load a model first');
    return;
  }
  
  if (!uploadedShapFile) {
    alert('Please upload sample data first');
    return;
  }
  
  const analysisType = document.getElementById('shap-analysis-type')?.value || 'summary';
  
  // Generate SHAP visualizations using the uploaded file
  setTimeout(() => {
    generateShapSummaryChart(uploadedShapFile);
    generateShapDecisionChart(uploadedShapFile);
    generateShapForceChart(uploadedShapFile);
    generateShapValuesTable(uploadedShapFile);
  }, 500);
}

function generateDefaultShapVisualizations() {
  setTimeout(() => {
    // Check if a file has been uploaded
    if (uploadedShapFile) {
      // Use the uploaded file
      generateShapSummaryChart(uploadedShapFile);
      generateShapDecisionChart(uploadedShapFile);
      generateShapForceChart(uploadedShapFile);
      generateShapValuesTable(uploadedShapFile);
    } else {
      console.log('No file uploaded yet. Please upload a CSV file.');
    }
  }, 500);
}

// SHAP Summary Chart - Standard SHAP Summary Plot
async function generateShapSummaryChart(file, containerId = 'shap-summary-chart') {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  try {
    let fileData;
    
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric feature columns (exclude 'species')
    const features = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Calculate SHAP values (using normalized deviation from mean as proxy)
    const shapData = {};
    features.forEach(feature => {
      const values = data.map(row => parseFloat(row[feature])).filter(v => !isNaN(v));
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
      
      shapData[feature] = data.map(row => {
        const val = parseFloat(row[feature]);
        const shapValue = std > 0 ? (val - mean) / std * 0.1 : 0; // Scaled SHAP value
        const featureValue = val;
        const species = row.species ? row.species.toLowerCase() : 'setosa';
        
        return { shapValue, featureValue, species };
      });
    });
    
    // Calculate mean absolute SHAP value for sorting
    const featureImportance = features.map(f => {
      const meanAbsShap = shapData[f].reduce((sum, d) => sum + Math.abs(d.shapValue), 0) / shapData[f].length;
      return { feature: f, importance: meanAbsShap };
    });
    
    // Sort by importance (descending)
    featureImportance.sort((a, b) => b.importance - a.importance);
    const sortedFeatures = featureImportance.map(f => f.feature);
    
    // Color mapping for species (representing feature value - high to low)
    const getColor = (featureValue, feature) => {
      const values = data.map(row => parseFloat(row[feature])).filter(v => !isNaN(v));
      const min = Math.min(...values);
      const max = Math.max(...values);
      const normalized = (featureValue - min) / (max - min);
      
      // Color gradient: low (blue) to high (pink/red)
      if (normalized > 0.66) return '#E91E63'; // High - Pink
      if (normalized > 0.33) return '#9C27B0'; // Medium - Purple
      return '#2196F3'; // Low - Blue
    };
    
    // Create scatter datasets for each feature with vertical jitter
    const datasets = [];
    sortedFeatures.forEach((feature, idx) => {
      const points = shapData[feature].map(d => {
        // Add random vertical jitter to prevent overlap (-0.3 to +0.3)
        const jitter = (Math.random() - 0.5) * 0.6;
        
        return {
          x: d.shapValue,
          y: idx + jitter,
          backgroundColor: getColor(d.featureValue, feature),
          borderColor: getColor(d.featureValue, feature)
        };
      });
      
      datasets.push({
        label: feature,
        data: points,
        backgroundColor: points.map(p => p.backgroundColor),
        borderColor: points.map(p => p.borderColor),
        pointRadius: 5,
        pointHoverRadius: 7,
        showLine: false
      });
    });
    
    // Destroy existing chart
    if (charts.shapSummary) {
      charts.shapSummary.destroy();
    }
    
    // Get or create canvas
    let canvas = container;
    if (container.tagName !== 'CANVAS') {
      canvas = container.querySelector('canvas') || document.createElement('canvas');
      if (!canvas.parentElement) {
        container.innerHTML = '';
        container.appendChild(canvas);
      }
    }
    
    // Create the chart
    charts.shapSummary = new Chart(canvas, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: 'SHAP value (impact on model output)',
              font: { size: 12 }
            },
            grid: { color: '#e0e0e0' }
          },
          y: {
            type: 'linear',
            min: -0.5,
            max: sortedFeatures.length - 0.5,
            ticks: {
              stepSize: 1,
              callback: function(value) {
                const idx = Math.round(value);
                return sortedFeatures[idx] || '';
              },
              font: { size: 11 }
            },
            grid: { display: false }
          }
        },
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'SHAP Summary Plot',
            font: { size: 16, weight: 'bold' }
          },
          tooltip: {
            callbacks: {
              title: function(context) {
                const idx = Math.round(context[0].parsed.y);
                return sortedFeatures[idx] || '';
              },
              label: function(context) {
                return `SHAP value: ${context.parsed.x.toFixed(4)}`;
              }
            }
          }
        }
      }
    });
    
    return charts.shapSummary;
    
  } catch (error) {
    console.error('Error generating SHAP summary chart:', error);
    throw error;
  }
}

async function generateShapDecisionChart(file, ctxId = 'shap-decision-chart') {
  const ctx = document.getElementById(ctxId);
  if (!ctx) {
    console.error(`${ctxId} element not found`);
    return;
  }
  
  // Ensure canvas is visible and has proper display
  ctx.style.display = 'block';
  ctx.style.width = '100%';
  ctx.style.height = '100%';
  
  if (ctx.offsetWidth === 0 || ctx.offsetHeight === 0) {
    console.error('Canvas has zero dimensions even after styling!');
    console.error('Width:', ctx.offsetWidth, 'Height:', ctx.offsetHeight);
    // Wait a tick for layout
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species' or other categorical columns)
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Map species to class numbers
    const speciesMap = {
      'setosa': 0,
      'versicolor': 1,
      'virginica': 2
    };
    
    // Calculate mean absolute SHAP values per feature to sort them
    const featureMeans = {};
    numericColumns.forEach(feature => {
      const values = data.map(row => Math.abs(parseFloat(row[feature]) || 0));
      featureMeans[feature] = values.reduce((sum, v) => sum + v, 0) / values.length;
    });
    
    // Sort features by importance (descending)
    const sortedFeatures = numericColumns.sort((a, b) => featureMeans[b] - featureMeans[a]);
    
    // Create decision plot using canvas
    const canvas = ctx;
    // Use explicit fallback sizes if offsetWidth/Height are 0
    const containerWidth = canvas.offsetWidth || canvas.parentElement?.offsetWidth || 800;
    const containerHeight = canvas.offsetHeight || canvas.parentElement?.offsetHeight || 600;
    
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    
    const context = canvas.getContext('2d');
    
    if (!context) {
      console.error('Could not get 2d context!');
      return;
    }
    
    // Clear canvas
    context.clearRect(0, 0, canvas.width, canvas.height);
    
    // Chart dimensions
    const margin = { top: 40, right: 150, bottom: 60, left: 120 };
    const width = canvas.width - margin.left - margin.right;
    const height = canvas.height - margin.top - margin.bottom;
    
    // Calculate cumulative SHAP values for each sample
    const samples = data.map(row => {
      const species = row.species ? row.species.toLowerCase() : 'setosa';
      const classNum = speciesMap[species] || 0;
      
      let cumulative = 1.0; // Base value
      const path = [{ feature: 'base', value: cumulative, class: classNum }];
      
      sortedFeatures.forEach(feature => {
        const shapValue = parseFloat(row[feature]) || 0;
        cumulative += shapValue;
        path.push({ feature, value: cumulative, class: classNum });
      });
      
      return { path, class: classNum };
    });
    
    // Determine value range
    const allValues = samples.flatMap(s => s.path.map(p => p.value));
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    
    // Scales
    const xScale = (val) => margin.left + ((val - minValue) / (maxValue - minValue)) * width;
    const yScale = (idx) => margin.top + (idx / sortedFeatures.length) * height;
    
    // Colors for classes
    const classColors = ['#1E88E5', '#9C27B0', '#E91E63'];
    
    // Draw horizontal grid lines and feature labels
    context.strokeStyle = '#e0e0e0';
    context.lineWidth = 1;
    context.setLineDash([2, 2]);
    context.font = '14px Arial';
    context.fillStyle = '#e0e0e0';
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    
    sortedFeatures.forEach((feature, idx) => {
      const y = yScale(idx);
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(canvas.width - margin.right, y);
      context.stroke();
      
      context.fillText(feature, margin.left - 10, y);
    });
    
    context.setLineDash([]);
    
    // Draw x-axis
    context.strokeStyle = '#e0e0e0';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(margin.left, canvas.height - margin.bottom);
    context.lineTo(canvas.width - margin.right, canvas.height - margin.bottom);
    context.stroke();
    
    // X-axis label
    context.font = '13px Arial';
    context.fillStyle = '#e0e0e0';
    context.textAlign = 'center';
    context.fillText('Model output value', margin.left + width / 2, canvas.height - margin.bottom + 35);
    
    // X-axis ticks
    context.font = '12px Arial';
    context.textBaseline = 'top';
    for (let i = 0; i <= 8; i++) {
      const val = minValue + (i / 8) * (maxValue - minValue);
      const x = xScale(val);
      context.fillText(val.toFixed(2), x, canvas.height - margin.bottom + 5);
    }
    
    // Color scale bar at top
    const barHeight = 20;
    const barY = 10;
    const gradient = context.createLinearGradient(margin.left, 0, canvas.width - margin.right, 0);
    gradient.addColorStop(0, classColors[0]);
    gradient.addColorStop(0.5, classColors[1]);
    gradient.addColorStop(1, classColors[2]);
    
    context.fillStyle = gradient;
    context.fillRect(margin.left, barY, width, barHeight);
    
    // Draw sample paths
    samples.forEach(sample => {
      const color = classColors[sample.class];
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.globalAlpha = 0.6;
      
      context.beginPath();
      sample.path.forEach((point, idx) => {
        const x = xScale(point.value);
        const y = idx === 0 ? yScale(-0.5) : yScale(idx - 1);
        
        if (idx === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      });
      context.stroke();
    });
    
    context.globalAlpha = 1.0;
    
    // Store for cleanup
    charts.shapDecision = { 
      destroy: () => {
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    };
    return charts.shapDecision;
    
  } catch (error) {
    console.error('Error generating SHAP decision chart:', error);
    throw error;
  }
}

async function generateShapForceChart(file, containerId = 'shap-force-chart') {
  let container = document.getElementById(containerId);
  if (!container) {
    console.error('Force chart container not found');
    return;
  }
  
  // If it's a canvas, we need to replace it with a div
  if (container.tagName === 'CANVAS') {
    const parent = container.parentElement;
    const newContainer = document.createElement('div');
    newContainer.id = 'shap-force-chart';
    newContainer.style.width = '100%';
    newContainer.style.height = '100%';
    parent.replaceChild(newContainer, container);
    container = newContainer;
  }
  
  try {
    let fileData;
    
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species')
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Calculate means
    const means = {};
    numericColumns.forEach(col => {
      const vals = data.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
      means[col] = vals.reduce((sum, v) => sum + v, 0) / vals.length;
    });
    
    // Use first data row
    const firstRow = data[0];
    const baseValue = 1.0; // Starting base value
    
    // Calculate SHAP contributions
    const contributions = numericColumns.map(col => {
      const value = parseFloat(firstRow[col]) || 0;
      const contribution = (value - means[col]) * 0.1; // Scale factor
      return { feature: col, value: contribution, featureValue: value };
    });
    
    // Sort by contribution magnitude
    contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    
    // Build cumulative values for waterfall
    let cumulative = baseValue;
    const positions = [{ x: baseValue, label: `base value`, isBase: true }];
    
    contributions.forEach(contrib => {
      const start = cumulative;
      cumulative += contrib.value;
      positions.push({
        x: cumulative,
        label: `${contrib.feature} = ${contrib.featureValue.toFixed(1)}`,
        start: start,
        end: cumulative,
        value: contrib.value,
        isPositive: contrib.value >= 0
      });
    });
    
    const finalValue = cumulative;
    
    // Create custom HTML visualization
    container.innerHTML = '';
    container.style.position = 'relative';
    container.style.height = '180px';
    container.style.padding = '50px 30px 30px 30px';
    container.style.backgroundColor = '#f8f9fa';
    container.style.borderRadius = '8px';
    
    // Create SVG
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.style.position = 'absolute';
    svg.style.top = '50px';
    svg.style.left = '30px';
    svg.style.right = '30px';
    svg.style.bottom = '30px';
    
    // Calculate scale with better spacing
    const minVal = Math.min(baseValue, finalValue, ...positions.map(p => p.x));
    const maxVal = Math.max(baseValue, finalValue, ...positions.map(p => p.x));
    const range = maxVal - minVal || 1;
    const padding = range * 0.15;
    const scale = (val) => {
      const normalized = (val - minVal + padding) / (range + 2 * padding);
      return 5 + normalized * 90; // Use 5-95% of width for better spacing
    };
    
    // Draw axis
    const axisY = 70;
    const axisLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    axisLine.setAttribute('x1', '5%');
    axisLine.setAttribute('x2', '95%');
    axisLine.setAttribute('y1', axisY);
    axisLine.setAttribute('y2', axisY);
    axisLine.setAttribute('stroke', '#aaa');
    axisLine.setAttribute('stroke-width', '1.5');
    svg.appendChild(axisLine);
    
    // Draw fewer, cleaner axis ticks
    const numTicks = 6;
    for (let i = 0; i <= numTicks; i++) {
      const val = minVal - padding + (range + 2 * padding) * (i / numTicks);
      const x = 5 + (90 * i / numTicks);
      
      const tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tick.setAttribute('x1', `${x}%`);
      tick.setAttribute('x2', `${x}%`);
      tick.setAttribute('y1', axisY);
      tick.setAttribute('y2', axisY + 4);
      tick.setAttribute('stroke', '#aaa');
      tick.setAttribute('stroke-width', '1');
      svg.appendChild(tick);
      
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', `${x}%`);
      text.setAttribute('y', axisY + 16);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#666');
      text.setAttribute('font-size', '11');
      text.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
      text.textContent = val.toFixed(2);
      svg.appendChild(text);
    }
    
    // Draw force plot arrows with improved styling
    let prevX = scale(baseValue);
    
    positions.forEach((pos, idx) => {
      if (idx === 0) {
        // Base value marker - cleaner style
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', `${prevX}%`);
        label.setAttribute('y', axisY - 45);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', '#555');
        label.setAttribute('font-size', '12');
        label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
        label.textContent = 'base value';
        svg.appendChild(label);
        
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        marker.setAttribute('x1', `${prevX}%`);
        marker.setAttribute('x2', `${prevX}%`);
        marker.setAttribute('y1', axisY - 38);
        marker.setAttribute('y2', axisY);
        marker.setAttribute('stroke', '#888');
        marker.setAttribute('stroke-width', '2');
        marker.setAttribute('stroke-dasharray', '2,2');
        svg.appendChild(marker);
      } else {
        const currentX = scale(pos.x);
        const color = pos.isPositive ? '#ff0d57' : '#1e88e5';
        const width = Math.abs(currentX - prevX);
        
        // Only show arrow if width is significant
        if (width > 2) {
          // Arrow body with gradient
          const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
          const arrowHeight = 22;
          const arrowHeadWidth = 6;
          
          const points = pos.isPositive
            ? `${prevX},${axisY - arrowHeight} ${currentX - arrowHeadWidth},${axisY - arrowHeight} ${currentX},${axisY - arrowHeight / 2} ${currentX - arrowHeadWidth},${axisY} ${prevX},${axisY}`
            : `${prevX},${axisY - arrowHeight} ${currentX + arrowHeadWidth},${axisY - arrowHeight} ${currentX},${axisY - arrowHeight / 2} ${currentX + arrowHeadWidth},${axisY} ${prevX},${axisY}`;
          
          arrow.setAttribute('points', points);
          arrow.setAttribute('fill', color);
          arrow.setAttribute('opacity', '0.85');
          arrow.setAttribute('stroke', color);
          arrow.setAttribute('stroke-width', '0.5');
          svg.appendChild(arrow);
          
          // Feature label - only if there's enough space
          if (width > 8) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', `${(prevX + currentX) / 2}%`);
            label.setAttribute('y', axisY - arrowHeight - 3);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('fill', '#444');
            label.setAttribute('font-size', '10');
            label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
            label.textContent = pos.label;
            svg.appendChild(label);
          }
        }
        
        prevX = currentX;
      }
    });
    
    // Final prediction marker with better styling
    const finalX = scale(finalValue);
    
    const finalLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    finalLabel.setAttribute('x', `${finalX}%`);
    finalLabel.setAttribute('y', axisY - 45);
    finalLabel.setAttribute('text-anchor', 'middle');
    finalLabel.setAttribute('fill', '#222');
    finalLabel.setAttribute('font-size', '13');
    finalLabel.setAttribute('font-weight', 'bold');
    finalLabel.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
    finalLabel.textContent = `f(x) = ${finalValue.toFixed(2)}`;
    svg.appendChild(finalLabel);
    
    const finalMarker = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    finalMarker.setAttribute('x1', `${finalX}%`);
    finalMarker.setAttribute('x2', `${finalX}%`);
    finalMarker.setAttribute('y1', axisY - 38);
    finalMarker.setAttribute('y2', axisY);
    finalMarker.setAttribute('stroke', '#222');
    finalMarker.setAttribute('stroke-width', '2.5');
    svg.appendChild(finalMarker);
    
    // Legend with improved styling
    const legend = document.createElement('div');
    legend.style.position = 'absolute';
    legend.style.top = '15px';
    legend.style.right = '30px';
    legend.style.fontSize = '12px';
    legend.style.fontFamily = 'system-ui, -apple-system, sans-serif';
    legend.innerHTML = `
      <div style="display: flex; gap: 12px; align-items: center; background: white; padding: 6px 12px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div><span style="color: #ff0d57; font-weight: 600;">higher</span></div>
        <div style="display: flex; align-items: center; gap: 6px;">
          <div style="width: 24px; height: 10px; background: linear-gradient(to right, #1e88e5, #ff0d57); border-radius: 2px;"></div>
          <span style="color: #555; font-weight: 500;">f(x)</span>
        </div>
        <div><span style="color: #1e88e5; font-weight: 600;">lower</span></div>
      </div>
    `;
    
    container.appendChild(svg);
    container.appendChild(legend);
    
    return true;
    
  } catch (error) {
    console.error('Error generating SHAP force chart:', error);
    throw error;
  }
}

async function generateShapBarChart(file, ctxId = 'shap-bar-chart') {
  const ctx = document.getElementById(ctxId);
  if (!ctx) return;
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Get numeric columns (excluding 'species' or other categorical columns)
    const numericColumns = headers.filter(h => {
      const firstValue = data[0][h];
      return typeof firstValue === 'number' || (!isNaN(parseFloat(firstValue)) && h.toLowerCase() !== 'species');
    });
    
    // Map species to class numbers
    const speciesMap = {
      'setosa': 0,
      'versicolor': 1,
      'virginica': 2
    };
    
    // Calculate mean absolute SHAP values per feature per class
    const featureClassValues = {};
    
    numericColumns.forEach(feature => {
      featureClassValues[feature] = { 0: [], 1: [], 2: [] };
      
      data.forEach(row => {
        const species = row.species ? row.species.toLowerCase() : 'setosa';
        const classNum = speciesMap[species] || 0;
        const value = parseFloat(row[feature]);
        
        if (!isNaN(value)) {
          featureClassValues[feature][classNum].push(Math.abs(value));
        }
      });
    });
    
    // Calculate mean for each feature-class combination
    const featureMeans = {};
    numericColumns.forEach(feature => {
      featureMeans[feature] = {};
      for (let c = 0; c < 3; c++) {
        const values = featureClassValues[feature][c];
        const mean = values.length > 0 
          ? values.reduce((sum, v) => sum + v, 0) / values.length 
          : 0;
        featureMeans[feature][c] = mean;
      }
    });
    
    // Sort features by total impact (sum across all classes)
    const sortedFeatures = numericColumns.sort((a, b) => {
      const sumA = featureMeans[a][0] + featureMeans[a][1] + featureMeans[a][2];
      const sumB = featureMeans[b][0] + featureMeans[b][1] + featureMeans[b][2];
      return sumB - sumA; // Descending order
    });
    
    // Create datasets for each class
    const datasets = [
      {
        label: 'Class 0',
        data: sortedFeatures.map(f => featureMeans[f][0]),
        backgroundColor: '#1E88E5', // Blue
        borderColor: '#1E88E5',
        borderWidth: 0
      },
      {
        label: 'Class 1',
        data: sortedFeatures.map(f => featureMeans[f][1]),
        backgroundColor: '#E91E63', // Pink/Magenta
        borderColor: '#E91E63',
        borderWidth: 0
      },
      {
        label: 'Class 2',
        data: sortedFeatures.map(f => featureMeans[f][2]),
        backgroundColor: '#9E9D24', // Olive/Yellow-green
        borderColor: '#9E9D24',
        borderWidth: 0
      }
    ];
    
    // Destroy existing chart if it exists
    if (charts.shapBar) {
      charts.shapBar.destroy();
    }
    
    // Create stacked horizontal bar chart
    charts.shapBar = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: sortedFeatures,
        datasets: datasets
      },
      options: {
        indexAxis: 'y', // Horizontal bars
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            beginAtZero: true,
            title: {
              display: true,
              text: 'mean(|SHAP value|) (average impact on model output magnitude)',
              font: { size: 12 }
            },
            grid: {
              color: '#e0e0e0'
            }
          },
          y: {
            stacked: true,
            grid: {
              display: false
            },
            ticks: {
              font: { size: 13 }
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'right',
            labels: {
              font: { size: 12 },
              boxWidth: 40,
              padding: 15
            }
          },
          title: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.dataset.label || '';
                const value = context.parsed.x.toFixed(4);
                return `${label}: ${value}`;
              }
            }
          }
        }
      }
    });
    
    return charts.shapBar;
    
  } catch (error) {
    console.error('Error generating SHAP bar chart:', error);
    throw error;
  }
}

async function generateShapValuesTable(file, tableId = 'shap-values-table') {
  const table = document.getElementById(tableId);
  if (!table) return;
  
  try {
    let fileData;
    
    // Handle different input types
    if (file instanceof File) {
      fileData = await file.text();
    } else if (typeof file === 'string') {
      const response = await fetch(file);
      fileData = await response.text();
    } else {
      throw new Error('Invalid file input. Expected File object or file path string.');
    }
    
    // Parse CSV using PapaParse
    const parsed = Papa.parse(fileData, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      transformHeader: header => header.trim()
    });
    
    if (parsed.errors.length > 0) {
      console.error('CSV parsing errors:', parsed.errors);
      throw new Error('Failed to parse CSV file');
    }
    
    const data = parsed.data;
    const headers = Object.keys(data[0] || {});
    
    // Check if this is pre-formatted SHAP data or raw data
    const hasShapColumns = headers.some(h => 
      (h.toLowerCase() === 'feature' || h.toLowerCase() === 'shap_value')
    );
    
    let shapData;
    
    if (hasShapColumns) {
      // Pre-formatted SHAP data with feature, shap_value, feature_value columns
      shapData = data.map(row => {
        const shapValue = parseFloat(row.shap_value || row['SHAP Value'] || row.shap || 0);
        const featureValue = parseFloat(row.feature_value || row['Feature Value'] || row.value || 0);
        
        // Determine contribution category
        let contribution;
        const absValue = Math.abs(shapValue);
        if (shapValue >= 0.3) {
          contribution = 'High Positive';
        } else if (shapValue > 0) {
          contribution = 'Positive';
        } else if (shapValue <= -0.3) {
          contribution = 'High Negative';
        } else {
          contribution = 'Negative';
        }
        
        return {
          feature: row.feature || row.Feature || '',
          shap_value: shapValue,
          feature_value: featureValue,
          contribution: row.contribution || contribution
        };
      });
    } else {
      // Raw data format - calculate SHAP-like values from first row
      const numericColumns = headers.filter(h => {
        const firstValue = data[0][h];
        return typeof firstValue === 'number' || !isNaN(parseFloat(firstValue));
      });
      
      // Calculate means and standard deviations
      const stats = {};
      numericColumns.forEach(col => {
        const vals = data.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
        const mean = vals.reduce((sum, v) => sum + v, 0) / vals.length;
        const variance = vals.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / vals.length;
        const stdDev = Math.sqrt(variance);
        stats[col] = { mean, stdDev };
      });
      
      // Use first row to generate SHAP-like values
      const firstRow = data[0];
      
      shapData = numericColumns.map(col => {
        const featureValue = parseFloat(firstRow[col]) || 0;
        const { mean, stdDev } = stats[col];
        
        // Calculate normalized deviation as SHAP value
        const shapValue = stdDev > 0 ? (featureValue - mean) / stdDev : 0;
        
        // Determine contribution category
        let contribution;
        if (shapValue >= 0.5) {
          contribution = 'High Positive';
        } else if (shapValue > 0) {
          contribution = 'Positive';
        } else if (shapValue <= -0.5) {
          contribution = 'High Negative';
        } else {
          contribution = 'Negative';
        }
        
        return {
          feature: col,
          shap_value: shapValue,
          feature_value: featureValue,
          contribution: contribution
        };
      });
      
      // Sort by absolute SHAP value (descending)
      shapData.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    }
    
    // Find or create tbody
    let tbody = table.querySelector('tbody');
    if (!tbody) {
      tbody = document.createElement('tbody');
      table.appendChild(tbody);
    }
    
    // Find max absolute SHAP value for scaling the bars
    const maxAbsShap = Math.max(...shapData.map(row => Math.abs(row.shap_value)));
    
    // Populate table
    tbody.innerHTML = shapData.map(row => `
      <tr>
        <td>${row.feature}</td>
        <td class="${row.shap_value >= 0 ? 'shap-value-positive' : 'shap-value-negative'}">
          ${row.shap_value.toFixed(4)}
        </td>
        <td>${row.feature_value.toFixed(2)}</td>
        <td>
          ${row.contribution}
          <div class="contribution-bar">
            <div class="${row.shap_value >= 0 ? 'contribution-positive' : 'contribution-negative'}" 
                 style="width: ${maxAbsShap > 0 ? (Math.abs(row.shap_value) / maxAbsShap * 100) : 0}%"></div>
          </div>
        </td>
      </tr>
    `).join('');
    
    return shapData;
    
  } catch (error) {
    console.error('Error generating SHAP values table:', error);
    throw error;
  }
}

// Architecture functionality
function initializeArchitecture() {
  renderArchitectureDiagram();
  setTimeout(() => {
    createDataFlowChart();
    createTechStackChart();
  }, 100);
}

function renderArchitectureDiagram() {
  const diagramContainer = document.getElementById('architecture-diagram');
  if (!diagramContainer) {
    return;
  }
  
  diagramContainer.innerHTML = `
    <div class="flowchart-diagram">
      <!-- Top Layer: User Configuration -->
      <div class="flow-row">
        <div class="flow-node training">
          <div class="node-header">
            <span class="node-icon">⚙️</span>
            <span class="node-title">① Select Models (Single Type)</span>
          </div>
          <div class="node-body">
            <div class="node-detail">Regression / Classification / Clustering</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-head">▼</div>
      </div>
      
      <div class="flow-row">
        <div class="flow-node mlflow">
          <div class="node-header">
            <span class="node-icon">📊</span>
            <span class="node-title">② Select Metrics (Artifacts)</span>
          </div>
          <div class="node-body">
            <div class="node-detail">Per-model metric selection</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-head">▼</div>
      </div>
      
      <div class="flow-row">
        <div class="flow-node api">
          <div class="node-header">
            <span class="node-icon">🎛️</span>
            <span class="node-title">③ Configure Hyperparameters</span>
          </div>
          <div class="node-body">
            <div class="node-detail">Model-specific parameters</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-head">▼</div>
      </div>
      
      <div class="flow-row">
        <div class="flow-node storage">
          <div class="node-header">
            <span class="node-icon">📁</span>
            <span class="node-title">④ Upload Dataset & Configure</span>
          </div>
          <div class="node-body">
            <div class="node-detail">• Upload CSV</div>
            <div class="node-detail">• Preprocessing (remove/fill nulls)</div>
            <div class="node-detail">• Split ratio & method</div>
            <div class="node-detail">• Target column selection</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-label">POST /run-batch-models</div>
        <div class="arrow-head">▼</div>
      </div>
      
      <!-- Model Execution Layer -->
      <div class="flow-row">
        <div class="flow-node ui">
          <div class="node-header">
            <span class="node-icon">🚀</span>
            <span class="node-title">⑤ Multi-Model Execution</span>
          </div>
          <div class="node-body">
            <div class="node-detail">Sequential training of all selected models</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-label">Each model creates separate run</div>
        <div class="arrow-head">▼</div>
      </div>
      
      <!-- MLflow Logging -->
      <div class="flow-row">
        <div class="flow-node training">
          <div class="node-header">
            <span class="node-icon">🎯</span>
            <span class="node-title">⑥ MLflow Tracking</span>
          </div>
          <div class="node-body">
            <div class="node-detail">• Log hyperparameters</div>
            <div class="node-detail">• Log metrics</div>
            <div class="node-detail">• Save model artifacts</div>
            <div class="node-detail">• Store SHAP plots</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-head">▼</div>
      </div>
      
      <!-- SHAP Analysis -->
      <div class="flow-row">
        <div class="flow-node mlflow">
          <div class="node-header">
            <span class="node-icon">🔬</span>
            <span class="node-title">⑦ SHAP Analysis</span>
          </div>
          <div class="node-body">
            <div class="node-detail">Automatic for every run</div>
          </div>
        </div>
      </div>
      
      <div class="flow-arrow">
        <div class="arrow-line"></div>
        <div class="arrow-label">View results</div>
        <div class="arrow-head">▼</div>
      </div>
      
      <!-- Comparison Feature -->
      <div class="flow-row">
        <div class="flow-node user">
          <div class="node-header">
            <span class="node-icon">📈</span>
            <span class="node-title">⑧ Experiment Detail & Comparison</span>
          </div>
          <div class="node-body">
            <div class="node-detail">• Compare all runs side-by-side</div>
            <div class="node-detail">• View metrics & parameters</div>
            <div class="node-detail">• Access SHAP analysis</div>
            <div class="node-detail">• Download results</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function createDataFlowChart() {
  const ctx = document.getElementById('dataFlowChart');
  if (!ctx) return;
  
  if (charts.dataFlow) {
    charts.dataFlow.destroy();
  }
  
  charts.dataFlow = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Model Training', 'Feature Analysis', 'SHAP Computation', 'Fairness Evaluation', 'Result Storage', 'UI Rendering'],
      datasets: [{
        data: [22, 18, 15, 13, 20, 12],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom'
        },
        title: {
          display: true,
          text: 'Enhanced Data Processing Distribution'
        }
      }
    }
  });
}

function createTechStackChart() {
  const ctx = document.getElementById('techStackChart');
  if (!ctx) return;
  
  if (charts.techStack) {
    charts.techStack.destroy();
  }
  
  charts.techStack = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Frontend', 'Backend', 'ML/AI', 'Explainability', 'Storage', 'Analytics'],
      datasets: [{
        label: 'Technology Components',
        data: [8, 12, 15, 10, 10, 6],
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Number of Components'
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

function renderRecentExperiments() {
  const tbody = document.getElementById('recent-experiments-tbody');
  if (!tbody) return;
  
  const recentExperiments = experimentsData.experiments.slice(0, 5);
  
  tbody.innerHTML = recentExperiments.map(exp => `
    <tr>
      <td>
        <a href="#" class="experiment-name" onclick="showExperimentDetail('${exp.id}'); return false;">${exp.name}</a>
      </td>
      <td><span class="status-badge ${exp.status.toLowerCase()}">${exp.status}</span></td>
      <td>${exp.model_type}</td>
      <td>${exp.dataset}</td>
      <td>${exp.metrics.accuracy ? (exp.metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
      <td>${exp.duration}</td>
      <td>${exp.user}</td>
    </tr>
  `).join('');
}

function createSuccessRateChart() {
  const ctx = document.getElementById('successRateChart');
  if (!ctx) return;
  
  // Generate sample data for success rate over time
  const dates = ['Sep 10', 'Sep 11', 'Sep 12', 'Sep 13', 'Sep 14', 'Sep 15', 'Sep 16', 'Sep 17'];
  const successRates = [82, 85, 87, 83, 89, 91, 88, 85];
  
  if (charts.successRate) {
    charts.successRate.destroy();
  }
  
  charts.successRate = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Success Rate (%)',
        data: successRates,
        borderColor: '#1FB8CD',
        backgroundColor: 'rgba(31, 184, 205, 0.1)',
        borderWidth: 3,
        fill: true,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

function createPerformanceChart() {
  const ctx = document.getElementById('performanceChart');
  if (!ctx) return;
  
  const finishedExperiments = experimentsData.experiments.filter(exp => exp.status === 'FINISHED');
  const modelTypes = [...new Set(finishedExperiments.map(exp => exp.model_type))];
  const avgAccuracy = modelTypes.map(model => {
    const modelExps = finishedExperiments.filter(exp => exp.model_type === model);
    const avgAcc = modelExps.reduce((sum, exp) => sum + exp.metrics.accuracy, 0) / modelExps.length;
    return avgAcc * 100;
  });
  
  if (charts.performance) {
    charts.performance.destroy();
  }
  
  charts.performance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: modelTypes,
      datasets: [{
        label: 'Average Accuracy (%)',
        data: avgAccuracy,
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        }
      }
    }
  });
}

// Experiments functionality
function initializeExperiments() {
  filteredExperiments = [...experimentsData.experiments];
  showView("experiments")
  renderExperimentsTable();
}

async function getAllExperimentsFromAPI() {
  const MLFLOW_TRACKING_URI = 'http://localhost:8080';
  
  try {
    const response = await fetch(`${MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/search`, {
      method: 'POST',  // ← This is the key change!
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ max_results: 1000 })
    });
    const data = await response.json();
    return data.experiments || [];
  } catch (error) {
    console.error('Error fetching experiments:', error.message);
    return [];
  }
}

async function getExperimentDetailsFromAPI(experimentId) {
  const MLFLOW_TRACKING_URI = 'http://localhost:8080';
  
  try {
    const response = await fetch(
      `${MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/search`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: [experimentId] })
      }
    );
    const data = await response.json();
    return data.runs || [];
  } catch (error) {
    console.error('Error fetching experiments:', error.message);
    return [];
  }
}

async function getRunDetailsFromAPI(runId) {
  const MLFLOW_TRACKING_URI = 'http://localhost:8080';
  try {
    const url = new URL(`${MLFLOW_TRACKING_URI}/api/2.0/mlflow/runs/get`);
    url.searchParams.set('run_id', runId);
    const response = await fetch(url.toString(), { method: 'GET' });
    const data = await response.json();
    return data.run || null;
  } catch (error) {
    console.error('Error fetching run details:', error.message);
    return null;
  }
}

async function getRunArtifactsFromAPI(runId, path = '') {
  const MLFLOW_TRACKING_URI = 'http://localhost:8080';
  try {
    const url = new URL(`${MLFLOW_TRACKING_URI}/api/2.0/mlflow/artifacts/list`);
    url.searchParams.set('run_id', runId);
    if (path) url.searchParams.set('path', path);

    const response = await fetch(url.toString(), { method: 'GET' });
    const data = await response.json();
    return data.files || [];
  } catch (error) {
    console.error('Error fetching artifacts:', error.message);
    return [];
  }
}

let currentRunId = null;
let runMetricsChart = null;

async function showRunDetail(runId) {
  const run = await getRunDetailsFromAPI(runId);
  if (!run) {
    console.error('Run not found:', runId);
    return;
  }
  currentRunId = runId;

  const runName = run.info.run_name || run.info.run_id;
  const titleEl = document.getElementById('run-detail-title');
  if (titleEl) titleEl.textContent = `Run: ${runName}`;

  const toPairs = (objOrArr) => Array.isArray(objOrArr) ? objOrArr.map(i => [i.key, i.value]) : Object.entries(objOrArr || {});
  const paramsPairs = toPairs(run.data.params);
  const metricsPairs = toPairs(run.data.metrics);
  const tagsPairs = toPairs(run.data.tags);

  // Overview tab
  const overviewEl = document.getElementById('run-overview-tab');
  if (overviewEl) {
    overviewEl.innerHTML = `
      <div class="card" style="margin-top: var(--space-24);">
        <div class="card__body">
          <h3>About this run</h3>
          <div class="info-grid">
            <div><strong>Experiment ID:</strong> ${run.info.experiment_id}</div>
            <div><strong>Status:</strong> <span class="status-badge ${run.info.status.toLowerCase()}">${run.info.status}</span></div>
            <div><strong>Run ID:</strong> ${run.info.run_id}</div>
            <div><strong>Duration:</strong> ${run.info.end_time ? (((run.info.end_time - run.info.start_time)/1000).toFixed(1) + 's') : 'Running'}</div>
            <div><strong>Source:</strong> ${run.data.tags?.find?.(t => t.key === 'mlflow.source.name')?.value || '--'}</div>
            <div><strong>Created by:</strong> ${run.info.user_id || '--'}</div>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: var(--space-24);">
        <div class="card__body">
          <h3>Metrics</h3>
          <div style="margin-bottom: var(--space-12);">
            <input id="run-metrics-search" class="form-control" placeholder="Search metrics" oninput="filterRunTableRows('run-metrics-table', this.value)">
          </div>
          ${metricsPairs.length === 0 ? '<p>No metrics logged</p>' : `
          <div class="table-container">
            <table id="run-metrics-table" class="experiments-table">
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>
                ${metricsPairs.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}
              </tbody>
            </table>
          </div>`}
        </div>
      </div>

      <div class="card" style="margin-top: var(--space-24);">
        <div class="card__body">
          <h3>Parameters</h3>
          <div style="margin-bottom: var(--space-12);">
            <input id="run-params-search" class="form-control" placeholder="Search parameters" oninput="filterRunTableRows('run-params-table', this.value)">
          </div>
          ${paramsPairs.length === 0 ? '<p>No parameters logged</p>' : `
          <div class="table-container">
            <table id="run-params-table" class="experiments-table">
              <thead><tr><th>Key</th><th>Value</th></tr></thead>
              <tbody>
                ${paramsPairs.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}
              </tbody>
            </table>
          </div>`}
        </div>
      </div>
    `;
  }

  // Model metrics tab with chart
  const metricsEl = document.getElementById('run-metrics-tab');
  if (metricsEl) {
    const numericMetrics = metricsPairs
      .map(([k,v]) => [k, Number(v)])
      .filter(([_,v]) => !Number.isNaN(v));

    metricsEl.innerHTML = `
      <div class="card">
        <div class="card__body">
          <div style="display:flex; align-items:center; gap: var(--space-12);">
            <h3 style="margin:0;">Model metrics</h3>
            <label style="flex:1;"><input id="run-metrics-search-2" class="form-control" placeholder="Search metric charts" oninput="filterRunTableRows('run-metrics-table-2', this.value)"></label>
          </div>
          ${numericMetrics.length === 0 ? '<p>No numeric metrics available</p>' : `
          <div class="chart-container" style="position: relative; height: 300px; margin-top: var(--space-16);">
            <canvas id="runMetricsChart"></canvas>
          </div>`}
          <div class="table-container" style="margin-top: var(--space-16);">
            <table id="run-metrics-table-2" class="experiments-table">
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>
                ${metricsPairs.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    const chartEl = document.getElementById('runMetricsChart');
    if (chartEl && numericMetrics.length > 0) {
      if (runMetricsChart) runMetricsChart.destroy();
      runMetricsChart = new Chart(chartEl, {
        type: 'bar',
        data: {
          labels: numericMetrics.map(([k]) => k),
          datasets: [{
            label: 'Metric value',
            data: numericMetrics.map(([_,v]) => v),
            backgroundColor: '#1FB8CD',
            borderColor: '#1FB8CD',
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: true } },
          plugins: { legend: { display: false } }
        }
      });
    }
  }

  // Artifacts tab initial load
  await renderRunArtifacts(runId, '');

  showView('run-detail');
  switchRunTab('overview');
}

function filterRunTableRows(tableId, query) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  if (!tbody) return;
  const q = (query || '').toLowerCase();
  tbody.querySelectorAll('tr').forEach(tr => {
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

async function renderRunArtifacts(runId, path = '') {
  const artifactsEl = document.getElementById('run-artifacts-tab');
  if (!artifactsEl) return;

  const files = await getRunArtifactsFromAPI(runId, path);
  const parentPath = path.split('/').slice(0, -1).join('/');
  const hasParent = !!path;

  const PROXY_BASE = 'http://localhost:8080';
  const previewUrlBase = `${PROXY_BASE}/get-artifact`;

  artifactsEl.innerHTML = `
    <div class="card">
      <div class="card__body">
        <div style="display:flex; gap: var(--space-24);">
          <div style="flex: 0 0 300px;">
            <h3>Artifacts ${path ? `— ${path}` : ''}</h3>
            <div class="artifact-actions" style="margin-bottom: var(--space-12);">
              ${hasParent ? `<button class="btn btn--sm btn--outline" onclick="loadArtifactPath('${runId}', '${parentPath}')">⬆️ Up</button>` : ''}
            </div>
            ${files.length === 0 ? '<p>No artifacts found</p>' : `
              <ul>
                ${files.map(file => `
                  <li style="margin-bottom: 8px;">
                    ${file.is_dir
                      ? `<button class="btn btn--sm btn--outline" onclick="loadArtifactPath('${runId}', '${file.path}')">${file.path.split('/').pop()}/</button>`
                      : `<button class="btn btn--sm btn--primary" onclick="previewArtifact('${runId}', '${file.path}')">${file.path.split('/').pop()}</button>
                         <a class="btn btn--sm btn--outline" href="${previewUrlBase}?run_uuid=${runId}&path=${encodeURIComponent(file.path)}" target="_blank" rel="noopener">Download</a>`
                    }
                  </li>
                `).join('')}
              </ul>
            `}
          </div>
          <div style="flex: 1;">
            <h3>Preview</h3>
            <div id="artifact-preview" style="border: 1px solid var(--color-border); height: 360px;">
              <div style="display:flex; align-items:center; justify-content:center; height:100%; color: var(--color-text-secondary);">
                Select a file to preview
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function loadArtifactPath(runId, path) {
  renderRunArtifacts(runId, path);
}

function previewArtifact(runId, path) {
  const previewEl = document.getElementById('artifact-preview');
  const PROXY_BASE = 'http://localhost:8080';
  const url = `${PROXY_BASE}/get-artifact?run_uuid=${runId}&path=${encodeURIComponent(path)}`;
  if (previewEl) {
    previewEl.innerHTML = `<iframe src="${url}" style="width:100%; height:100%; border:none;"></iframe>`;
  }
}

function switchRunTab(tabName) {
  ['run-overview-tab','run-metrics-tab','run-artifacts-tab','run-shap-tab'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });

  const target = document.getElementById(
    tabName === 'overview' ? 'run-overview-tab' :
    tabName === 'metrics' ? 'run-metrics-tab' :
    tabName === 'artifacts' ? 'run-artifacts-tab' :
    'run-shap-tab'
  );
  if (target) target.classList.add('active');

  document.querySelectorAll('#run-detail-view .feature-tab').forEach(btn => btn.classList.remove('active'));
  const activeBtn = Array.from(document.querySelectorAll('#run-detail-view .feature-tab'))
    .find(b => b.textContent.toLowerCase().includes(tabName));
  if (activeBtn) activeBtn.classList.add('active');

  if (tabName === 'artifacts' && currentRunId) {
    renderRunArtifacts(currentRunId);
  } else if (tabName === 'shap' && currentRunId) {
    renderRunShapTab(currentRunId);
  }
}

// Find a CSV artifact path, searching up to 3 levels deep
async function findCsvArtifactPath(runId, path = '', depth = 0) {
  const entries = await getRunArtifactsFromAPI(runId, path);
  const csv = entries.find(e => !e.is_dir && e.path.toLowerCase().endsWith('.csv'));
  if (csv) return csv.path;
  if (depth >= 3) return null;
  for (const dir of entries.filter(e => e.is_dir)) {
    const found = await findCsvArtifactPath(runId, dir.path, depth + 1);
    if (found) return found;
  }
  return null;
}

async function renderRunShapTab(runId) {
  const statusEl = document.getElementById('run-shap-status');
  try {
    if (statusEl) statusEl.textContent = 'Searching for CSV artifacts...';
    const csvPath = await findCsvArtifactPath(runId, '');
    if (!csvPath) {
      if (statusEl) statusEl.textContent = 'No CSV artifact found in this run.';
      return;
    }

    const csvUrl = await chooseWorkingArtifactUrl(runId, csvPath);
    if (statusEl) statusEl.textContent = `Using artifact: ${csvPath}`;

    await generateShapSummaryChart(csvUrl, 'run-shap-summary-chart');
    await generateShapDecisionChart(csvUrl, 'run-shap-decision-chart');
    await generateShapBarChart(csvUrl, 'run-shap-bar-chart');
    await generateShapForceChart(csvUrl, 'run-shap-force-chart');
    await generateShapValuesTable(csvUrl, 'run-shap-values-table');

    if (statusEl) statusEl.textContent = 'SHAP analysis rendered successfully.';
  } catch (err) {
    console.error('Failed to render SHAP tab:', err);
    if (statusEl) statusEl.textContent = `Failed to render SHAP analysis: ${err.message}`;
  }
}

async function chooseWorkingArtifactUrl(runId, csvPath) {
  const PROXY_BASE = 'http://localhost:8080';
  const url = `${PROXY_BASE}/artifact-content?run_id=${runId}&path=${encodeURIComponent(csvPath)}`;
  // simple reachability check (optional)
  try {
    const r = await fetch(url, { method: 'GET' });
    if (r.ok) return url;
  } catch (_) { /* ignore */ }
  throw new Error('Artifact content route unreachable');
}

async function renderExperimentsTable() {
  const tbody = document.getElementById('experiments-tbody');
  if (!tbody) return;
  
  let experimentsData = await getAllExperimentsFromAPI()
  tbody.innerHTML = experimentsData.map((exp, i) => `
    <tr data-exp='${encodeURIComponent(JSON.stringify(exp))}'>
      <td>
        <a href="#" class="experiment-name">${exp.name}</a>
      </td>
      <td>
        <span class="status-badge ${exp.lifecycle_stage.toLowerCase()}">
          ${exp.lifecycle_stage}
        </span>
      </td>
      <td>
        <button class="btn btn--sm btn--outline view-btn">View</button>
      </td>
    </tr>
  `).join('');

  tbody.querySelectorAll('.experiment-name, .view-btn').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
  
      const tr = e.target.closest('tr');
      const exp = JSON.parse(decodeURIComponent(tr.dataset.exp));
  
      showExperimentDetail(exp);
    });
  });
}

function showComparisonAndGenerateReport() {
  if (selectedExperiments.length < 2) {
    alert('Please select at least 2 experiments to compare.');
    return;
  }
  
  renderComparisonView();
  showView('comparison');
}

function setupEventListeners() {
  // Search functionality
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', handleSearch);
  }
  
  // Filter functionality
  ['status-filter', 'model-filter', 'dataset-filter'].forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener('change', handleFilters);
    }
  });
}

function handleSearch(e) {
  applyFilters();
}

function handleFilters() {
  applyFilters();
}

function applyFilters() {
  const searchTerm = document.getElementById('search-input')?.value.toLowerCase() || '';
  const statusFilter = document.getElementById('status-filter')?.value || '';
  const modelFilter = document.getElementById('model-filter')?.value || '';
  const datasetFilter = document.getElementById('dataset-filter')?.value || '';
  
  filteredExperiments = experimentsData.experiments.filter(exp => {
    const matchesSearch = exp.name.toLowerCase().includes(searchTerm) || 
                         exp.user.toLowerCase().includes(searchTerm);
    const matchesStatus = !statusFilter || exp.status === statusFilter;
    const matchesModel = !modelFilter || exp.model_type === modelFilter;
    const matchesDataset = !datasetFilter || exp.dataset === datasetFilter;
    
    return matchesSearch && matchesStatus && matchesModel && matchesDataset;
  });
  
  renderExperimentsTable();
}

// Compare mode functionality
function updateCompareMode() {
  const compareBanner = document.getElementById('compare-banner');
  const compareHeader = document.getElementById('compare-header');
  
  if (compareMode) {
    compareBanner?.classList.remove('hidden');
    compareHeader?.classList.remove('hidden');
  } else {
    compareBanner?.classList.add('hidden');
    compareHeader?.classList.add('hidden');
    selectedExperiments = [];
  }
  
  renderExperimentsTable();
  updateSelectedCount();
}

function handleExperimentSelection(checkbox) {
  if (checkbox.checked) {
    if (selectedExperiments.length < 4) {
      selectedExperiments.push(checkbox.value);
    } else {
      checkbox.checked = false;
      alert('You can compare up to 4 experiments at once.');
    }
  } else {
    selectedExperiments = selectedExperiments.filter(id => id !== checkbox.value);
  }
  updateSelectedCount();
}

function updateSelectedCount() {
  const countElement = document.getElementById('selected-count');
  if (countElement) {
    countElement.textContent = `${selectedExperiments.length} selected`;
  }
}

function clearSelection() {
  selectedExperiments = [];
  document.querySelectorAll('.compare-checkbox').forEach(cb => cb.checked = false);
  updateSelectedCount();
}

function showComparison() {
  if (selectedExperiments.length < 2) {
    alert('Please select at least 2 experiments to compare.');
    return;
  }
  
  renderComparisonView();
  showView('comparison');
}

// Feature analysis tabs functionality
function switchFeatureTab(tabName, experimentId) {
  // Hide all tab contents for this experiment
  document.querySelectorAll(`#experiment-${experimentId} .feature-tab-content`).forEach(content => {
    content.classList.remove('active');
  });
  
  // Show selected tab content
  const targetContent = document.getElementById(`${tabName}-content-${experimentId}`);
  if (targetContent) {
    targetContent.classList.add('active');
  }
  
  // Update tab buttons
  document.querySelectorAll(`#experiment-${experimentId} .feature-tab`).forEach(tab => {
    tab.classList.remove('active');
  });
  
  const targetTab = document.querySelector(`#experiment-${experimentId} .feature-tab[onclick*="${tabName}"]`);
  if (targetTab) {
    targetTab.classList.add('active');
  }
}

// Experiment detail view
// Add this to your existing showExperimentDetail function

function switchExperimentDetailTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('#experiment-detail-view .feature-tab-content').forEach(el => {
        el.classList.remove('active');
    });

    // Show selected tab content
    const tabContent = document.getElementById(`${tabName}-tab-content`);
    if (tabContent) {
        tabContent.classList.add('active');
        
        // If comparison tab is clicked, render the comparison
        if (tabName === 'comparison') {
            renderExperimentRunsComparison(window.currentExperimentId, window.currentExperimentRuns);
        }
    }

    // Update tab button active state
    document.querySelectorAll('#experiment-detail-view .feature-tab').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const activeBtn = Array.from(document.querySelectorAll('#experiment-detail-view .feature-tab'))
        .find(b => {
            const text = b.textContent.toLowerCase().trim();
            return text === tabName.toLowerCase();
        });
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
}

// Update showExperimentDetail to populate the runs tab
async function showExperimentDetail(experiment) {
    const runs = await getExperimentDetailsFromAPI(experiment.experiment_id);

    if (!runs) {
        console.error('Experiment not found:', experiment.experiment_id);
        return;
    }
    
    document.getElementById('experiment-detail-title').textContent = experiment.name;
    
    const content = document.getElementById('experiment-detail-content');
    content.innerHTML = `
        <div class="detail-section" style="margin-top: var(--space-32);">
            <h3>Runs (${runs.length})</h3>
            ${runs.length === 0 ? `
                <p style="color: var(--color-text-secondary); padding: var(--space-24); text-align: center;">
                    No runs found for this experiment
                </p>
            ` : `
                <div class="card">
                    <div class="card__body">
                        <div class="table-container">
                            <table class="experiments-table">
                                <thead>
                                    <tr>
                                        <th>Run Name</th>
                                        <th>Source</th>
                                        <th>Status</th>
                                        <th>Model Type</th>
                                        <th>Dataset</th>
                                        <th>User</th>
                                        <th>Accuracy</th>
                                        <th>Duration</th>
                                        <th>Start Time</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${runs.map((run) => {
                                        const metricsMap = Object.fromEntries(
                                            (run.data.metrics || []).map(m => [m.key, m.value])
                                        );
                                        return `
                                        <tr>
                                            <td>
                                                <a href="#" class="experiment-name" onclick="showRunDetail('${run.info.run_id}'); return false;">
                                                    ${run.info.run_name}
                                                </a>
                                            </td>
                                            <td>${run.data.tags && run.data.tags[2] ? run.data.tags[2].value : '--'}</td>
                                            <td>
                                                <span class="status-badge ${run.info.status.toLowerCase()}">${run.info.status}</span>
                                            </td>
                                            <td>${run.data.params && run.data.params[0] ? run.data.params[0].value : '--'}</td>
                                            <td>${run.dataset ? run.dataset : '--'}</td>
                                            <td>${run.info.user_id}</td>
                                            <td>${metricsMap.accuracy ? (metricsMap.accuracy * 100).toFixed(1) + '%' : 'N/A'}</td>
                                            <td>${((run.info.end_time-run.info.start_time) / 1000).toFixed(1)}s</td>
                                            <td>${new Date(run.info.start_time).toLocaleString()}</td>
                                            <td>
                                                <button class="btn btn--sm btn--outline" onclick="showRunDetail('${run.info.run_id}')">
                                                    View Details
                                                </button>
                                            </td>
                                        </tr>
                                    `}).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `}
        </div>
    `;
    
    showView('experiment-detail');

    // Store experiment ID and runs for comparison
    window.currentExperimentId = experiment.experiment_id;
    window.currentExperimentRuns = runs;
}

// Function to render comparison of all runs in an experiment
function renderExperimentRunsComparison(experimentId, runs) {
    const container = document.getElementById('experiment-comparison-content');
    if (!container || !runs) {
        console.error('Container or runs not found');
        return;
    }

    const finishedRuns = runs.filter(r => r.info.status === 'FINISHED');

    if (finishedRuns.length < 2) {
        container.innerHTML = `
            <div class="card">
                <div class="card__body">
                    <p style="text-align: center; color: var(--color-text-secondary);">
                        Need at least 2 finished runs to compare. Current finished runs: ${finishedRuns.length}
                    </p>
                </div>
            </div>
        `;
        return;
    }

    // Extract metrics
    const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
    
    // Helper to get metric value
    const getMetricValue = (run, metricName) => {
        const metric = run.data.metrics?.find(m => m.key === metricName);
        return metric ? metric.value : null;
    };

    // Helper to get param value
    const getParamValue = (run, paramName) => {
        const param = run.data.params?.find(p => p.key === paramName);
        return param ? param.value : null;
    };

    // Build metrics comparison table
    let metricsHTML = '<div class="table-container"><table class="experiments-table"><thead><tr><th style="min-width: 150px;">Metric</th>';
    metricsHTML += finishedRuns.map(run => `<th style="text-align: center;">${run.info.run_name}</th>`).join('');
    metricsHTML += '</tr></thead><tbody>';

    metrics.forEach(metric => {
        metricsHTML += '<tr>';
        metricsHTML += `<td style="font-weight: 600;">${metric.replace(/_/g, ' ').toUpperCase()}</td>`;
        
        const metricValues = finishedRuns.map(run => getMetricValue(run, metric));
        const validValues = metricValues.filter(v => v !== null);
        const bestValue = validValues.length > 0 ? Math.max(...validValues) : null;
        
        metricsHTML += metricValues.map(val => {
            const isBest = val !== null && val === bestValue;
            const displayVal = val ? (val * 100).toFixed(2) + '%' : 'N/A';
            const bgColor = isBest ? 'background-color: rgba(31, 184, 205, 0.2);' : '';
            return `<td style="text-align: center; padding: 12px; ${bgColor}${isBest ? 'font-weight: bold;' : ''}">${displayVal}</td>`;
        }).join('');
        
        metricsHTML += '</tr>';
    });

    metricsHTML += '</tbody></table></div>';

    // Get all unique parameters
    const allParams = new Set();
    finishedRuns.forEach(run => {
        (run.data.params || []).forEach(p => allParams.add(p.key));
    });

    let paramsHTML = '';
    if (allParams.size > 0) {
        paramsHTML = '<div class="table-container"><table class="experiments-table"><thead><tr><th style="min-width: 150px;">Parameter</th>';
        paramsHTML += finishedRuns.map(run => `<th style="text-align: center;">${run.info.run_name}</th>`).join('');
        paramsHTML += '</tr></thead><tbody>';

        Array.from(allParams).forEach(paramKey => {
            paramsHTML += '<tr>';
            paramsHTML += `<td style="font-weight: 600;">${paramKey}</td>`;
            paramsHTML += finishedRuns.map(run => {
                const param = run.data.params?.find(p => p.key === paramKey);
                return `<td style="text-align: center; padding: 12px;">${param ? param.value : '-'}</td>`;
            }).join('');
            paramsHTML += '</tr>';
        });

        paramsHTML += '</tbody></table></div>';
    }

    // Build overview cards
    let overviewHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">';
    
    finishedRuns.forEach(run => {
        const metricsMap = Object.fromEntries(
            (run.data.metrics || []).map(m => [m.key, m.value])
        );
        const duration = ((run.info.end_time - run.info.start_time) / 1000).toFixed(1);
        
        overviewHTML += `
            <div class="card" style="cursor: pointer; transition: box-shadow 0.2s;" onclick="showRunDetail('${run.info.run_id}')">
                <div class="card__body">
                    <h4 style="margin-top: 0; color: #1FB8CD;">${run.info.run_name}</h4>
                    <div style="font-size: 13px;">
                        <p style="margin: 6px 0;"><strong>Status:</strong> <span class="status-badge ${run.info.status.toLowerCase()}">${run.info.status}</span></p>
                        <p style="margin: 6px 0;"><strong>Accuracy:</strong> ${metricsMap.accuracy ? (metricsMap.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
                        <p style="margin: 6px 0;"><strong>Duration:</strong> ${duration}s</p>
                        <p style="margin: 6px 0;"><strong>User:</strong> ${run.info.user_id || 'N/A'}</p>
                    </div>
                    <button class="btn btn--sm btn--primary" style="margin-top: 12px;">View Full Details</button>
                </div>
            </div>
        `;
    });
    
    overviewHTML += '</div>';

    // Create comparison chart
    const chartId = `experiment-comparison-chart-${experimentId}`;

    let chartHTML = `
        <div class="card" style="margin-top: 20px;">
            <div class="card__body">
                <h3>📊 Metrics Radar Chart</h3>
                <div style="position: relative; height: 400px; margin-top: 20px;">
                    <canvas id="${chartId}"></canvas>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = `
        ${overviewHTML}
        
        <div class="card">
            <div class="card__body">
                <h3>📋 Metrics Comparison</h3>
                ${metricsHTML}
                <p style="font-size: 12px; color: #666; margin-top: 12px;"><em>Highlighted cells show the best value for each metric</em></p>
            </div>
        </div>

        ${allParams.size > 0 ? `
            <div class="card" style="margin-top: 20px;">
                <div class="card__body">
                    <h3>⚙️ Parameters Comparison</h3>
                    ${paramsHTML}
                </div>
            </div>
        ` : ''}

        ${chartHTML}
    `;

    // Create radar chart after a small delay to ensure canvas is rendered
    setTimeout(() => {
        createExperimentRunsComparisonChart(chartId, finishedRuns, metrics);
    }, 100);
}

// New function to switch between tabs in experiment detail
function switchExperimentTab(tabName, experimentId) {
  // Hide all tabs
  document.querySelectorAll(`#runs-tab-${experimentId}, #comparison-tab-${experimentId}`).forEach(el => {
    el.classList.remove('active');
  });

  // Show selected tab
  if (tabName === 'runs') {
    const runsTab = document.getElementById(`runs-tab-${experimentId}`);
    if (runsTab) runsTab.classList.add('active');
  } else if (tabName === 'comparison') {
    const comparisonTab = document.getElementById(`comparison-tab-${experimentId}`);
    if (comparisonTab) {
      comparisonTab.classList.add('active');
      // Render comparison when tab is clicked
      renderExperimentRunsComparison(experimentId, window.currentExperimentRuns);
    }
  }

  // Update tab button active state
  document.querySelectorAll(`#experiment-detail-view .feature-tab`).forEach(btn => {
    btn.classList.remove('active');
  });
  
  const activeBtn = Array.from(document.querySelectorAll(`#experiment-detail-view .feature-tab`))
    .find(b => b.textContent.toLowerCase().includes(tabName));
  if (activeBtn) activeBtn.classList.add('active');
}

// Function to render comparison of all runs in an experiment
function renderExperimentRunsComparison(experimentId, runs) {
  const container = document.getElementById(`comparison-content-${experimentId}`);
  if (!container) return;

  const finishedRuns = runs.filter(r => r.info.status === 'FINISHED');

  if (finishedRuns.length < 2) {
    container.innerHTML = `
      <div class="card">
        <div class="card__body">
          <p style="text-align: center; color: var(--color-text-secondary);">
            Need at least 2 finished runs to compare.
          </p>
        </div>
      </div>
    `;
    return;
  }

  // Extract metrics and parameters
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  
  // Helper to get metric value
  const getMetricValue = (run, metricName) => {
    const metric = run.data.metrics?.find(m => m.key === metricName);
    return metric ? metric.value : null;
  };

  // Helper to get param value
  const getParamValue = (run, paramName) => {
    const param = run.data.params?.find(p => p.key === paramName);
    return param ? param.value : null;
  };

  // Build metrics comparison table
  let metricsHTML = '<div class="table-container"><table class="experiments-table"><thead><tr><th style="min-width: 150px;">Metric</th>';
  metricsHTML += finishedRuns.map(run => `<th style="text-align: center;">${run.info.run_name}</th>`).join('');
  metricsHTML += '</tr></thead><tbody>';

  metrics.forEach(metric => {
    metricsHTML += '<tr>';
    metricsHTML += `<td style="font-weight: 600;">${metric.replace(/_/g, ' ').toUpperCase()}</td>`;
    
    const metricValues = finishedRuns.map(run => getMetricValue(run, metric));
    const validValues = metricValues.filter(v => v !== null);
    const bestValue = validValues.length > 0 ? Math.max(...validValues) : null;
    
    metricsHTML += metricValues.map(val => {
      const isBest = val !== null && val === bestValue;
      const displayVal = val ? (val * 100).toFixed(2) + '%' : 'N/A';
      const bgColor = isBest ? 'background-color: rgba(31, 184, 205, 0.2);' : '';
      return `<td style="text-align: center; padding: 12px; ${bgColor}${isBest ? 'font-weight: bold;' : ''}">${displayVal}</td>`;
    }).join('');
    
    metricsHTML += '</tr>';
  });

  metricsHTML += '</tbody></table></div>';

  // Get all unique parameters
  const allParams = new Set();
  finishedRuns.forEach(run => {
    (run.data.params || []).forEach(p => allParams.add(p.key));
  });

  let paramsHTML = allParams.size > 0 ? '<div class="table-container"><table class="experiments-table"><thead><tr><th style="min-width: 150px;">Parameter</th>' : '';
  paramsHTML += finishedRuns.map(run => `<th style="text-align: center;">${run.info.run_name}</th>`).join('');
  paramsHTML += '</tr></thead><tbody>';

  Array.from(allParams).forEach(paramKey => {
    paramsHTML += '<tr>';
    paramsHTML += `<td style="font-weight: 600;">${paramKey}</td>`;
    paramsHTML += finishedRuns.map(run => {
      const param = run.data.params?.find(p => p.key === paramKey);
      return `<td style="text-align: center; padding: 12px;">${param ? param.value : '-'}</td>`;
    }).join('');
    paramsHTML += '</tr>';
  });

  paramsHTML += allParams.size > 0 ? '</tbody></table></div>' : '';

  // Build overview cards
  let overviewHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">';
  
  finishedRuns.forEach(run => {
    const metricsMap = Object.fromEntries(
      (run.data.metrics || []).map(m => [m.key, m.value])
    );
    const duration = ((run.info.end_time - run.info.start_time) / 1000).toFixed(1);
    
    overviewHTML += `
      <div class="card" style="cursor: pointer; transition: box-shadow 0.2s;" onclick="showRunDetail('${run.info.run_id}')">
        <div class="card__body">
          <h4 style="margin-top: 0; color: #1FB8CD;">${run.info.run_name}</h4>
          <div style="font-size: 13px;">
            <p style="margin: 6px 0;"><strong>Status:</strong> <span class="status-badge ${run.info.status.toLowerCase()}">${run.info.status}</span></p>
            <p style="margin: 6px 0;"><strong>Accuracy:</strong> ${metricsMap.accuracy ? (metricsMap.accuracy * 100).toFixed(2) + '%' : 'N/A'}</p>
            <p style="margin: 6px 0;"><strong>Duration:</strong> ${duration}s</p>
            <p style="margin: 6px 0;"><strong>User:</strong> ${run.info.user_id || 'N/A'}</p>
          </div>
          <button class="btn btn--sm btn--primary" style="margin-top: 12px;">View Full Details</button>
        </div>
      </div>
    `;
  });
  
  overviewHTML += '</div>';

  // Create comparison chart
  const chartId = `experiment-comparison-chart-${experimentId}`;
  const chartContainerId = `experiment-comparison-container-${experimentId}`;

  let chartHTML = `
    <div class="card" style="margin-top: 20px;">
      <div class="card__body">
        <h3>📊 Metrics Radar Chart</h3>
        <div style="position: relative; height: 400px; margin-top: 20px;">
          <canvas id="${chartId}"></canvas>
        </div>
      </div>
    </div>
  `;

  container.innerHTML = `
    ${overviewHTML}
    
    <div class="card">
      <div class="card__body">
        <h3>📋 Metrics Comparison</h3>
        ${metricsHTML}
        <p style="font-size: 12px; color: #666; margin-top: 12px;"><em>Highlighted cells show the best value for each metric</em></p>
      </div>
    </div>

    ${allParams.size > 0 ? `
      <div class="card" style="margin-top: 20px;">
        <div class="card__body">
          <h3>⚙️ Parameters Comparison</h3>
          ${paramsHTML}
        </div>
      </div>
    ` : ''}

    ${chartHTML}
  `;

  // Create radar chart after a small delay to ensure canvas is rendered
  setTimeout(() => {
    createExperimentRunsComparisonChart(chartId, finishedRuns, metrics);
  }, 100);
}

// Function to create radar chart for runs comparison
function createExperimentRunsComparisonChart(canvasId, runs, metrics) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) {
        console.error('Canvas not found:', canvasId);
        return;
    }

    // Helper to get metric value
    const getMetricValue = (run, metricName) => {
        const metric = run.data.metrics?.find(m => m.key === metricName);
        return metric ? metric.value * 100 : 0; // Convert to percentage
    };

    // Prepare datasets for each run
    const datasets = runs.map((run, index) => {
        const colors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F'];
        const color = colors[index % colors.length];

        return {
            label: run.info.run_name,
            data: metrics.map(metric => getMetricValue(run, metric)),
            borderColor: color,
            backgroundColor: color + '33', // Add transparency
            borderWidth: 2,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7,
            pointBackgroundColor: color,
            pointBorderColor: '#fff',
            pointBorderWidth: 2
        };
    });

    // Destroy existing chart if it exists
    if (window.experimentRunsCharts && window.experimentRunsCharts[canvasId]) {
        window.experimentRunsCharts[canvasId].destroy();
    }

    if (!window.experimentRunsCharts) {
        window.experimentRunsCharts = {};
    }

    // Create new chart
    window.experimentRunsCharts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: metrics.map(m => m.replace(/_/g, ' ').toUpperCase()),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 12 },
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.r.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

function createFeatureImportanceChart(experimentId) {
  const ctx = document.getElementById(`feature-importance-chart-${experimentId}`);
  if (!ctx) return;
  
  const featureData = experimentsData.feature_importance[experimentId];
  const features = featureData.top_features.map(f => f[0]);
  const importances = featureData.top_features.map(f => f[1] * 100);
  
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: features,
      datasets: [{
        label: 'Importance (%)',
        data: importances,
        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'],
        borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C', '#964325', '#944454', '#13343B'],
        borderWidth: 1
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `Importance: ${context.parsed.x.toFixed(2)}%`;
            }
          }
        }
      }
    }
  });
}

function createDetailShapCharts(experimentId) {
  const featureData = experimentsData.feature_importance[experimentId];
  if (!featureData.shap_values) return;
  
  // SHAP Summary Chart
  const summaryCtx = document.getElementById(`shap-summary-detail-${experimentId}`);
  if (summaryCtx) {
    const features = Object.keys(featureData.shap_values);
    const shapValues = Object.values(featureData.shap_values);
    
    new Chart(summaryCtx, {
      type: 'bar',
      data: {
        labels: features,
        datasets: [{
          label: 'SHAP Values',
          data: shapValues.map(Math.abs),
          backgroundColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
          borderColor: shapValues.map(val => val >= 0 ? '#1FB8CD' : '#B4413C'),
          borderWidth: 1
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: 'Mean |SHAP Value|' }
          }
        },
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'SHAP Feature Importance' }
        }
      }
    });
  }
  
  // SHAP Force Chart
  const forceCtx = document.getElementById(`shap-force-detail-${experimentId}`);
  if (forceCtx) {
    const features = Object.keys(featureData.shap_values);
    const shapValues = Object.values(featureData.shap_values);
    const baseValue = 0.5;
    
    // Create waterfall data
    const waterfallLabels = ['Base Value', ...features, 'Prediction'];
    const waterfallValues = [baseValue, ...shapValues, baseValue + shapValues.reduce((sum, val) => sum + val, 0)];
    
    new Chart(forceCtx, {
      type: 'bar',
      data: {
        labels: waterfallLabels,
        datasets: [{
          label: 'SHAP Force Plot',
          data: waterfallValues,
          backgroundColor: waterfallValues.map((val, idx) => {
            if (idx === 0 || idx === waterfallValues.length - 1) return '#ECEBD5';
            return val >= 0 ? '#1FB8CD' : '#B4413C';
          }),
          borderColor: waterfallValues.map((val, idx) => {
            if (idx === 0 || idx === waterfallValues.length - 1) return '#ECEBD5';
            return val >= 0 ? '#1FB8CD' : '#B4413C';
          }),
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: false, title: { display: true, text: 'Output' } }
        },
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'SHAP Force Plot - Prediction Breakdown' }
        }
      }
    });
  }
}

// Comparison view
// Fixed comparison view - fetches run data and compares metrics
async function renderComparisonView() {
  const allExperiments = await getAllExperimentsFromAPI();
  
  // Get selected experiments
  const selectedExps = allExperiments.filter(exp => 
    selectedExperiments.includes(exp.experiment_id)
  );

  if (selectedExps.length < 2) {
    alert('Please select at least 2 experiments to compare.');
    return;
  }

  // Fetch runs for each selected experiment
  const expWithRuns = await Promise.all(
    selectedExps.map(async (exp) => {
      const runs = await getExperimentDetailsFromAPI(exp.experiment_id);
      return { ...exp, runs: runs || [] };
    })
  );

  // Extract best run from each experiment (by status and metrics)
  const bestRuns = expWithRuns.map(exp => {
    const finishedRuns = exp.runs.filter(r => r.info.status === 'FINISHED');
    if (finishedRuns.length === 0) return null;
    
    // Find run with best accuracy
    return finishedRuns.reduce((best, current) => {
      const currentAcc = current.data.metrics?.find(m => m.key === 'accuracy')?.value || 0;
      const bestAcc = best.data.metrics?.find(m => m.key === 'accuracy')?.value || 0;
      return currentAcc > bestAcc ? current : best;
    });
  }).filter(r => r !== null);

  // Helper function to get metric value
  const getMetricValue = (run, metricName) => {
    const metric = run.data.metrics?.find(m => m.key === metricName);
    return metric ? metric.value : null;
  };

  // Create metrics comparison table
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  
  let metricsHTML = '<div class="table-container"><table class="experiments-table"><thead><tr><th>Metric</th>';
  metricsHTML += bestRuns.map(run => `<th>${expWithRuns.find(e => e.runs.includes(run))?.name || 'Unknown'}</th>`).join('');
  metricsHTML += '</tr></thead><tbody>';

  metrics.forEach(metric => {
    metricsHTML += '<tr>';
    metricsHTML += `<td><strong>${metric.replace('_', ' ').toUpperCase()}</strong></td>`;
    
    const metricValues = bestRuns.map(run => getMetricValue(run, metric));
    const bestValue = Math.max(...metricValues.filter(v => v !== null));
    
    metricsHTML += metricValues.map(val => {
      const isBest = val === bestValue && val !== null;
      const displayVal = val ? (val * 100).toFixed(2) + '%' : 'N/A';
      const bgColor = isBest ? 'background-color: rgba(31, 184, 205, 0.2);' : '';
      return `<td style="${bgColor}${isBest ? 'font-weight: bold;' : ''}">${displayVal}</td>`;
    }).join('');
    
    metricsHTML += '</tr>';
  });

  metricsHTML += '</tbody></table></div>';

  // Get parameters from best runs
  let paramsHTML = '<div class="table-container"><table class="experiments-table"><thead><tr><th>Parameter</th>';
  paramsHTML += bestRuns.map(run => `<th>${expWithRuns.find(e => e.runs.includes(run))?.name || 'Unknown'}</th>`).join('');
  paramsHTML += '</tr></thead><tbody>';

  const allParams = new Set();
  bestRuns.forEach(run => {
    (run.data.params || []).forEach(p => allParams.add(p.key));
  });

  Array.from(allParams).forEach(paramKey => {
    paramsHTML += '<tr>';
    paramsHTML += `<td><strong>${paramKey}</strong></td>`;
    paramsHTML += bestRuns.map(run => {
      const param = run.data.params?.find(p => p.key === paramKey);
      return `<td>${param ? param.value : '-'}</td>`;
    }).join('');
    paramsHTML += '</tr>';
  });

  paramsHTML += '</tbody></table></div>';

  // Create run details
  let runsHTML = '';
  bestRuns.forEach((run, idx) => {
    const exp = expWithRuns.find(e => e.runs.includes(run));
    runsHTML += `
      <div class="card" style="margin-bottom: 20px;">
        <div class="card__body">
          <h3>${exp?.name}</h3>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
              <p><strong>Run ID:</strong> <code style="background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 11px;">${run.info.run_id}</code></p>
              <p><strong>Status:</strong> <span class="status-badge ${run.info.status.toLowerCase()}">${run.info.status}</span></p>
              <p><strong>Duration:</strong> ${run.info.end_time ? (((run.info.end_time - run.info.start_time) / 1000).toFixed(1) + 's') : 'Running'}</p>
            </div>
            <div>
              <p><strong>Started:</strong> ${new Date(run.info.start_time).toLocaleString()}</p>
              <p><strong>User:</strong> ${run.info.user_id || 'N/A'}</p>
              <button class="btn btn--sm btn--primary" onclick="showRunDetail('${run.info.run_id}')">View Full Details</button>
            </div>
          </div>
        </div>
      </div>
    `;
  });

  const content = document.getElementById('comparison-content');
  content.innerHTML = `
    <div class="comparison-overview" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
      ${expWithRuns.map(exp => `
        <div class="card">
          <div class="card__body">
            <h3 style="margin-top: 0; font-size: 16px;">${exp.name}</h3>
            <p style="margin: 8px 0; font-size: 13px;"><strong>Runs:</strong> ${exp.runs.length}</p>
            <p style="margin: 8px 0; font-size: 13px;"><strong>Status:</strong> <span class="status-badge ${exp.lifecycle_stage.toLowerCase()}">${exp.lifecycle_stage}</span></p>
          </div>
        </div>
      `).join('')}
    </div>

    <div class="card">
      <div class="card__body">
        <h3>📊 Metrics Comparison (Best Run per Experiment)</h3>
        ${metricsHTML}
        <p style="font-size: 12px; color: #666; margin-top: 12px;"><em>Highlighted cells show the best value for each metric</em></p>
      </div>
    </div>

    <div class="card" style="margin-top: 20px;">
      <div class="card__body">
        <h3>⚙️ Parameters Comparison</h3>
        ${paramsHTML}
      </div>
    </div>

    <div style="margin-top: 20px;">
      <h3 style="margin-bottom: 20px;">📋 Run Details</h3>
      ${runsHTML}
    </div>
  `;
}

// Updated showComparison function
async function showComparison() {
  if (selectedExperiments.length < 2) {
    alert('Please select at least 2 experiments to compare.');
    return;
  }
  
  await renderComparisonView();
  showView('comparison');
}

function renderComparisonMetrics(allExps, currentExp) {
  if (currentExp.status !== 'FINISHED') {
    return '<p>Experiment not finished</p>';
  }
  
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  return metrics.map(metric => {
    const value = currentExp.metrics[metric];
    const allValues = allExps
      .filter(exp => exp.status === 'FINISHED')
      .map(exp => exp.metrics[metric])
      .filter(v => v !== null);
    const isBest = allValues.length > 0 && value === Math.max(...allValues);
    
    return `
      <div class="comparison-metric ${isBest ? 'best' : ''}">
        <span class="comparison-metric-name">${metric.replace('_', ' ').toUpperCase()}</span>
        <span class="comparison-metric-value">${value ? (value * 100).toFixed(2) + '%' : 'N/A'}</span>
      </div>
    `;
  }).join('');
}

function createComparisonChart(experiments) {
  const ctx = document.getElementById('comparisonChart');
  if (!ctx) return;
  
  const finishedExps = experiments.filter(exp => exp.status === 'FINISHED');
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];
  
  const datasets = finishedExps.map((exp, index) => ({
    label: exp.name,
    data: metrics.map(metric => exp.metrics[metric] * 100),
    borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index % 4],
    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index % 4] + '20',
    borderWidth: 2,
    fill: false
  }));
  
  new Chart(ctx, {
    type: 'radar',
    data: {
      labels: metrics.map(m => m.replace('_', ' ').toUpperCase()),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        }
      }
    }
  });
}

// Fairness analysis
function initializeFairness() {
  setTimeout(() => createFairnessChart(), 100);
}

function createFairnessChart() {
  const ctx = document.getElementById('fairnessChart');
  if (!ctx) return;
  
  const experimentsWithFairness = experimentsData.experiments.filter(exp => 
    Object.keys(exp.fairness_metrics).length > 0 && exp.status === 'FINISHED'
  );
  
  if (experimentsWithFairness.length === 0) return;
  
  const metrics = ['demographic_parity_difference', 'equalized_odds_difference', 'equal_opportunity_difference'];
  
  const datasets = metrics.map((metric, index) => ({
    label: metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    data: experimentsWithFairness.map(exp => Math.abs(exp.fairness_metrics[metric] || 0)),
    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C'][index],
    borderColor: ['#1FB8CD', '#FFC185', '#B4413C'][index],
    borderWidth: 1
  }));
  
  if (charts.fairness) {
    charts.fairness.destroy();
  }
  
  charts.fairness = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: experimentsWithFairness.map(exp => exp.name.split('_')[0]),
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Absolute Difference'
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Fairness Metrics Comparison Across Models'
        }
      }
    }
  });
}


// file upload code
// Simple direct approach - no fancy stuff
function initializeFileUpload() {
  const fileUploadArea = document.getElementById('file-upload-area');
  const fileInput = document.getElementById('file-input');

  if (!fileUploadArea || !fileInput) return;

  fileUploadArea.addEventListener('click', () => fileInput.click());

  fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragging');
  });

  fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragging');
  });

  fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragging');

    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });
}


async function handleFileUpload(file) {
  const fileUploadArea = document.getElementById('file-upload-area');
  fileUploadArea.classList.add('loading');
  
  if (!file.name.endsWith('.csv')) {
    alert('Please upload a CSV file');
    return;
  }
  
  try {
    const fileUploadArea = document.getElementById('file-upload-area');
    
    // Show loading
    fileUploadArea.innerHTML = `
      <div class="upload-icon">⏳</div>
      <div class="upload-text">
        <p class="upload-title">Processing file...</p>
      </div>
    `;
    
    // STORE THE FILE GLOBALLY
    uploadedShapFile = file;

    await generateShapSummaryChart(file);
    await generateShapDecisionChart(file);
    await generateShapForceChart(file);
    await generateShapBarChart(file);
    await generateShapValuesTable(file);
    
    // Show success AFTER charts are generated
    fileUploadArea.innerHTML = `
      <div class="upload-icon">✅</div>
      <div class="upload-text">
        <p class="upload-title">File uploaded successfully!</p>
        <p class="upload-subtitle">${file.name}</p>
        <p class="upload-subtitle" style="color: var(--color-success); margin-top: 8px;">✓ All 5 charts generated</p>
      </div>
    `;
    fileUploadArea.classList.remove('loading');
    fileUploadArea.classList.add('uploaded');
    
  } catch (error) {
    console.error('=== ERROR IN FILE UPLOAD ===');
    console.error('Error:', error);
    console.error('Error stack:', error.stack);
    alert('Failed to process file: ' + error.message);
    
    fileUploadArea.innerHTML = `
      <div class="upload-icon">📤</div>
      <div class="upload-text">
        <p class="upload-title">Upload CSV file for SHAP analysis</p>
        <p class="upload-subtitle">Drag and drop your file here or click to browse</p>
      </div>
    `;
    initializeFileUpload();
  }
}

// IMPORTANT: Only call this once!
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeFileUpload);
} else {
  initializeFileUpload();
}


// Add this script tag to your HTML (in the <head> section):
// <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>

// ============================================
// PDF REPORT GENERATION FEATURE
// ============================================

/**
 * Generate PDF Report for Selected Experiments
 * Call this function when user clicks "Generate Report" button
 */
async function generatePDFReport() {
  if (selectedExperiments.length < 2) {
    alert('Please select at least 2 experiments to generate a report.');
    return;
  }

  // Show loading indicator
  showLoadingModal('Generating PDF Report... This may take a few seconds.');

  try {
    // Get selected experiments data
    const selectedExps = experimentsData.experiments.filter(exp => 
      selectedExperiments.includes(exp.id)
    );

    // Create report HTML
    const reportHTML = createReportHTML(selectedExps);

    // Generate PDF
    const element = document.createElement('div');
    element.innerHTML = reportHTML;

    const opt = {
      margin: 10,
      filename: `model_comparison_${new Date().getTime()}.pdf`,
      image: { type: 'jpeg', quality: 0.98 },
      html2canvas: { scale: 2 },
      jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
    };

    // Generate and download
    await html2pdf().set(opt).from(element).save();

    // Hide loading and show success
    hideLoadingModal();
    showSuccessMessage('PDF Report generated and downloaded successfully! 📄');

  } catch (error) {
    hideLoadingModal();
    console.error('Error generating PDF:', error);
    alert('Error generating PDF: ' + error.message);
  }
}

/**
 * Create HTML structure for the PDF report
 */
function createReportHTML(selectedExps) {
  const finishedExps = selectedExps.filter(exp => exp.status === 'FINISHED');
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'];

  // Find best performers for each metric
  const bestPerformers = {};
  metrics.forEach(metric => {
    const validValues = finishedExps
      .map(exp => ({ exp, value: exp.metrics[metric] }))
      .filter(item => item.value !== null);
    
    if (validValues.length > 0) {
      const best = validValues.reduce((max, item) => 
        item.value > max.value ? item : max
      );
      bestPerformers[metric] = best.exp.id;
    }
  });

  // Generate table rows for metrics
  const metricsTableRows = finishedExps.map(exp => {
    return `
      <tr style="border-bottom: 1px solid #ddd;">
        <td style="padding: 12px; font-weight: 600;">${exp.name}</td>
        ${metrics.map(metric => {
          const value = exp.metrics[metric];
          const isBest = bestPerformers[metric] === exp.id;
          const cellStyle = isBest 
            ? 'background-color: #d4edda; font-weight: 600; color: #155724;'
            : 'color: #333;';
          
          return `<td style="padding: 12px; text-align: center; ${cellStyle}">
            ${value ? (value * 100).toFixed(2) + '%' : 'N/A'}
          </td>`;
        }).join('')}
      </tr>
    `;
  }).join('');

  // Generate model details section
  const modelDetailsHTML = finishedExps.map(exp => {
    const fairnessMetrics = Object.keys(exp.fairness_metrics).length > 0;
    
    return `
      <div style="page-break-inside: avoid; margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 8px;">
        <h3 style="margin-top: 0; color: #1FB8CD; border-bottom: 2px solid #1FB8CD; padding-bottom: 10px;">
          ${exp.name}
        </h3>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
          <div>
            <p><strong>Status:</strong> <span style="background-color: ${getStatusColor(exp.status)}; padding: 4px 12px; border-radius: 20px; color: white;">${exp.status}</span></p>
            <p><strong>Model Type:</strong> ${exp.model_type}</p>
            <p><strong>Dataset:</strong> ${exp.dataset}</p>
            <p><strong>User:</strong> ${exp.user}</p>
          </div>
          <div>
            <p><strong>Duration:</strong> ${exp.duration}</p>
            <p><strong>Start Time:</strong> ${exp.start_time}</p>
            <p><strong>Run ID:</strong> <code style="background-color: #f5f5f5; padding: 2px 6px;">${exp.id}</code></p>
          </div>
        </div>

        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
          <h4 style="margin-top: 0; color: #333;">Parameters</h4>
          <ul style="margin: 0; padding-left: 20px;">
            ${Object.entries(exp.parameters).map(([key, value]) => 
              `<li>${key}: <strong>${value}</strong></li>`
            ).join('')}
          </ul>
        </div>

        ${fairnessMetrics ? `
          <div style="background-color: #fff3cd; padding: 15px; border-radius: 4px; border-left: 4px solid #ffc107;">
            <h4 style="margin-top: 0; color: #856404;">Fairness Metrics</h4>
            <ul style="margin: 0; padding-left: 20px;">
              ${Object.entries(exp.fairness_metrics).map(([key, value]) => 
                `<li>${key.replace(/_/g, ' ').toUpperCase()}: <strong>${value.toFixed(4)}</strong></li>`
              ).join('')}
            </ul>
          </div>
        ` : ''}

        <div style="margin-top: 15px;">
          <p><strong>Artifacts (${exp.artifacts.length}):</strong> ${exp.artifacts.join(', ')}</p>
        </div>
      </div>
    `;
  }).join('');

  // Generate recommendations section
  const recommendationsHTML = generateRecommendations(finishedExps);

  // Main report HTML
  return `
    <div style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
      
      <!-- Header -->
      <div style="text-align: center; border-bottom: 3px solid #1FB8CD; padding-bottom: 20px; margin-bottom: 30px;">
        <h1 style="margin: 0; color: #1FB8CD; font-size: 32px;">Model Comparison Report</h1>
        <p style="margin: 10px 0 0 0; color: #666; font-size: 14px;">
          Generated on ${new Date().toLocaleString()}
        </p>
      </div>

      <!-- Executive Summary -->
      <div style="background-color: #e3f2fd; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #1FB8CD;">
        <h2 style="margin-top: 0; color: #0d47a1;">Executive Summary</h2>
        <p><strong>Models Compared:</strong> ${finishedExps.length} finished experiments</p>
        <p><strong>Comparison Date:</strong> ${new Date().toLocaleDateString()}</p>
        <p><strong>Best Overall Model:</strong> <span style="color: #1FB8CD; font-weight: 600;">${findBestModel(finishedExps).name}</span> (Accuracy: ${(findBestModel(finishedExps).metrics.accuracy * 100).toFixed(2)}%)</p>
      </div>

      <!-- Metrics Comparison Table -->
      <div style="margin-bottom: 30px;">
        <h2 style="color: #1FB8CD; border-bottom: 2px solid #1FB8CD; padding-bottom: 10px;">Performance Metrics Comparison</h2>
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
          <thead style="background-color: #1FB8CD; color: white;">
            <tr>
              <th style="padding: 12px; text-align: left;">Model</th>
              ${metrics.map(metric => `<th style="padding: 12px; text-align: center;">${metric.replace(/_/g, ' ').toUpperCase()}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${metricsTableRows}
          </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">
          <em>Note: Green highlighted cells indicate the best performer for each metric.</em>
        </p>
      </div>

      <!-- Detailed Model Breakdown -->
      <div style="margin-bottom: 30px;">
        <h2 style="color: #1FB8CD; border-bottom: 2px solid #1FB8CD; padding-bottom: 10px;">Detailed Model Analysis</h2>
        ${modelDetailsHTML}
      </div>

      <!-- Recommendations -->
      <div style="background-color: #f5f5f5; padding: 20px; border-radius: 8px; page-break-inside: avoid;">
        <h2 style="margin-top: 0; color: #1FB8CD;">Recommendations</h2>
        ${recommendationsHTML}
      </div>

      <!-- Footer -->
      <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #999; font-size: 12px;">
        <p>This report was generated by Enhanced MLflow Tracking System</p>
        <p>For questions or feedback, contact your ML team lead.</p>
      </div>

    </div>
  `;
}

/**
 * Generate recommendations based on experiment results
 */
function generateRecommendations(experiments) {
  const recommendations = [];

  if (experiments.length === 0) {
    return '<p>No finished experiments to analyze.</p>';
  }

  // Find best model
  const best = findBestModel(experiments);
  recommendations.push({
    type: 'success',
    title: '✅ Best Performer Identified',
    text: `<strong>${best.name}</strong> shows the best overall accuracy at <strong>${(best.metrics.accuracy * 100).toFixed(2)}%</strong>. Consider this for production deployment after further validation.`
  });

  // Fastest model
  const fastest = experiments.reduce((prev, current) => {
    const prevDuration = parseInt(prev.duration);
    const currentDuration = parseInt(current.duration);
    return currentDuration < prevDuration ? current : prev;
  });
  recommendations.push({
    type: 'info',
    title: '⚡ Fastest Model',
    text: `<strong>${fastest.name}</strong> completed in ${fastest.duration}. Useful if inference speed is critical.`
  });

  // Fairness check
  const fairnessExps = experiments.filter(exp => Object.keys(exp.fairness_metrics).length > 0);
  if (fairnessExps.length > 0) {
    const avgFairness = fairnessExps.reduce((sum, exp) => {
      const fairnessValues = Object.values(exp.fairness_metrics).map(Math.abs);
      return sum + fairnessValues.reduce((a, b) => a + b, 0) / fairnessValues.length;
    }, 0) / fairnessExps.length;

    if (avgFairness > 0.1) {
      recommendations.push({
        type: 'warning',
        title: '⚠️ Fairness Concerns',
        text: `Average fairness metrics indicate potential bias (mean difference: ${avgFairness.toFixed(4)}). Review fairness analysis and consider mitigation strategies.`
      });
    } else {
      recommendations.push({
        type: 'success',
        title: '✨ Good Fairness Performance',
        text: `Models show relatively balanced fairness metrics across different demographic groups. This is a positive indicator.`
      });
    }
  }

  // Precision vs Recall tradeoff
  const precisionRecallDiff = experiments.map(exp => ({
    name: exp.name,
    diff: Math.abs(exp.metrics.precision - exp.metrics.recall)
  }));
  const mostBalanced = precisionRecallDiff.reduce((prev, current) => 
    current.diff < prev.diff ? current : prev
  );
  recommendations.push({
    type: 'info',
    title: '⚖️ Best Precision-Recall Balance',
    text: `<strong>${mostBalanced.name}</strong> has the most balanced precision-recall trade-off, suitable if both metrics are equally important.`
  });

  // Variability check
  const standardDeviation = calculateMetricStdDev(experiments, 'accuracy');
  if (standardDeviation > 0.05) {
    recommendations.push({
      type: 'warning',
      title: '📊 High Model Variability',
      text: `Accuracy varies significantly across models (std dev: ${(standardDeviation * 100).toFixed(2)}%). Ensure consistent data preprocessing and validation methodology.`
    });
  }

  // Render recommendations
  return recommendations.map(rec => {
    const bgColor = rec.type === 'success' ? '#d4edda' : rec.type === 'warning' ? '#fff3cd' : '#d1ecf1';
    const borderColor = rec.type === 'success' ? '#28a745' : rec.type === 'warning' ? '#ffc107' : '#17a2b8';

    return `
      <div style="background-color: ${bgColor}; border-left: 4px solid ${borderColor}; padding: 15px; margin-bottom: 15px; border-radius: 4px;">
        <h4 style="margin-top: 0; color: ${borderColor};">${rec.title}</h4>
        <p style="margin: 0;">${rec.text}</p>
      </div>
    `;
  }).join('');
}

/**
 * Find the best performing model
 */
function findBestModel(experiments) {
  return experiments.reduce((best, current) => 
    current.metrics.accuracy > best.metrics.accuracy ? current : best
  );
}

/**
 * Calculate standard deviation of a metric
 */
function calculateMetricStdDev(experiments, metric) {
  const values = experiments
    .map(exp => exp.metrics[metric])
    .filter(v => v !== null);
  
  if (values.length === 0) return 0;
  
  const mean = values.reduce((a, b) => a + b) / values.length;
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Get color for status badge
 */
function getStatusColor(status) {
  switch (status) {
    case 'FINISHED': return '#28a745';
    case 'RUNNING': return '#ffc107';
    case 'FAILED': return '#dc3545';
    default: return '#6c757d';
  }
}

/**
 * Show loading modal
 */
function showLoadingModal(message) {
  const modal = document.createElement('div');
  modal.id = 'loading-modal';
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
  `;
  modal.innerHTML = `
    <div style="background: white; padding: 30px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
      <div style="font-size: 40px; margin-bottom: 15px;">⏳</div>
      <p style="margin: 0; font-size: 16px; color: #333;">${message}</p>
      <div style="margin-top: 15px; display: flex; justify-content: center; gap: 5px;">
        <div style="width: 8px; height: 8px; background: #1FB8CD; border-radius: 50%; animation: pulse 1.4s infinite;"></div>
        <div style="width: 8px; height: 8px; background: #1FB8CD; border-radius: 50%; animation: pulse 1.4s infinite 0.2s;"></div>
        <div style="width: 8px; height: 8px; background: #1FB8CD; border-radius: 50%; animation: pulse 1.4s infinite 0.4s;"></div>
      </div>
    </div>
  `;
  
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0%, 100% { opacity: 0.3; }
      50% { opacity: 1; }
    }
  `;
  document.head.appendChild(style);
  document.body.appendChild(modal);
}

/**
 * Hide loading modal
 */
function hideLoadingModal() {
  const modal = document.getElementById('loading-modal');
  if (modal) {
    modal.remove();
  }
}

/**
 * Show success message
 */
function showSuccessMessage(message) {
  const toast = document.createElement('div');
  toast.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #d4edda;
    color: #155724;
    padding: 16px 20px;
    border-radius: 8px;
    border: 1px solid #c3e6cb;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    z-index: 10000;
    animation: slideIn 0.3s ease-out;
  `;
  toast.textContent = message;
  
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from {
        transform: translateX(400px);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }
  `;
  document.head.appendChild(style);
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = 'slideIn 0.3s ease-out reverse';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
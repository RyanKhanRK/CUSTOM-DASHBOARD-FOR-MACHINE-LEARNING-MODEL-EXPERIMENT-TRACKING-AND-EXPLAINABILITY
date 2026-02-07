// ============================================
// MULTI-MODEL & MULTI-METRIC SELECTION
// ============================================

// Model to metrics mapping
const MODEL_METRICS = {
    // Regression Models
    'Regression_LinearRegression': ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    'Regression_GBTRegressor': ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    'Regression_RandomForest': ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    'Regression_DecisionTree': ['MSE', 'RMSE', 'MAE', 'R2 Score'],
    
    // Classification Models
    'Classification_LogisticRegression': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_DecisionTree': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_RandomForest': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_GBTClassifier': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_LinearSVC': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_MLPC': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_NaiveBayes': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    'Classification_OVR': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC ROC'],
    
    // Clustering Models
    'Clustering_BisectingKMeans': ['Silhouette Score', 'Davies Bouldin Score', 'Calinski Harabasz Score'],
    'Clustering_GaussianMixture': ['Silhouette Score', 'Davies Bouldin Score', 'Calinski Harabasz Score'],
    'Clustering_LDA': ['Silhouette Score', 'Davies Bouldin Score', 'Calinski Harabasz Score'],
    'Clustering_KMeans': ['Silhouette Score', 'Davies Bouldin Score', 'Calinski Harabasz Score']
};

// State management
const multiModelState = {
    selectedModels: new Set(),
    selectedMetrics: {},  // { modelName: ['Metric1', 'Metric2', ...], ... }
    selectedHyperparameters: {},
    uploadedFile: null,
    selectedModelType: null,  // Track which model type is selected (Regression, Classification, Clustering)
    preprocessingOption: 'remove',
    splitRatio: 0.2,
    splitType: 'random',  // 'random' or 'sequential'
    targetColumn: null,  // NEW: Add this
    availableColumns: [],  // NEW: Add this
    modelsByCategory: {
        'Regression': [],
        'Classification': [],
        'Clustering': []
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeMultiModelUI();
    setupModelEventListeners();
    setupMetricsPanel();
    setupFileUpload();
});

// ============================================
// INITIALIZATION FUNCTIONS
// ============================================

function initializeMultiModelUI() {
    // Organize models by category
    const modelSelects = document.querySelectorAll('.model-select');
    modelSelects.forEach(select => {
        const category = select.getAttribute('data-category');
        const model = select.getAttribute('data-model');
        if (category && model) {
            multiModelState.modelsByCategory[category].push({
                element: select,
                model: model,
                label: select.parentElement.querySelector('.checkbox-label').textContent.trim()
            });
        }
    });
}

// ============================================
// TARGET COLUMN SELECTION
// ============================================

async function loadDatasetColumns(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            preview: 1, // Only read first row to get column names
            complete: function(results) {
                if (results.meta && results.meta.fields) {
                    multiModelState.availableColumns = results.meta.fields;
                    resolve(results.meta.fields);
                } else {
                    reject(new Error('Could not parse CSV columns'));
                }
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

function renderTargetColumnSelection() {
    const container = document.getElementById('target-column-selection');
    const section = document.getElementById('target-column-section');
    
    if (!container || !section) {
        console.warn('Target column section not found');
        return;
    }
    
    // Hide section if clustering or no file uploaded
    if (multiModelState.selectedModelType === 'Clustering' || !multiModelState.uploadedFile) {
        section.style.display = 'none';
        return;
    }
    
    // Show section for Regression and Classification
    if (multiModelState.selectedModelType === 'Regression' || 
        multiModelState.selectedModelType === 'Classification') {

        section.style.display = 'block';
        
        let html = '<label style="display: block; margin-bottom: 10px; font-weight: 600; color: #1a1a1a;">';
        html += '🎯 Select Target Column';
        html += '</label>';
        html += '<p style="margin: 0 0 15px 0; color: #666; font-size: 13px;">';
        html += 'Choose the column you want to predict';
        html += '</p>';
        
        if (multiModelState.availableColumns.length > 0) {
            html += '<select id="target-column-select" class="form-control" onchange="selectTargetColumn(this.value)" style="max-width: 400px;">';
            html += '<option value="">-- Select target column --</option>';
            
            multiModelState.availableColumns.forEach(column => {
                const selected = multiModelState.targetColumn === column ? 'selected' : '';
                html += `<option value="${column}" ${selected}>${column}</option>`;
            });
            
            html += '</select>';
            
            // Show selected column
            if (multiModelState.targetColumn) {
                html += `
                    <div style="
                        margin-top: 15px;
                        padding: 12px;
                        background: #f0feff;
                        border-left: 3px solid #1FB8CD;
                        border-radius: 4px;
                    ">
                        <p style="margin: 0; font-size: 13px; color: #1FB8CD; font-weight: 600;">
                            ✓ Target Column: <span id="selected-target-text">${multiModelState.targetColumn}</span>
                        </p>
                    </div>
                `;
            }
        } else {
            html += '<p style="color: #999; font-style: italic;">Loading columns...</p>';
        }
        
        container.innerHTML = html;
    }
}

function selectTargetColumn(column) {
    multiModelState.targetColumn = column || null;
    
    // Update the selected display
    const selectedText = document.getElementById('selected-target-text');
    if (selectedText && column) {
        selectedText.textContent = column;
    }
    
    renderTargetColumnSelection();
    updateUI();
}

function isTargetColumnSelected() {
    // Clustering doesn't need target column
    if (multiModelState.selectedModelType === 'Clustering') {
        return true;
    }
    // Regression and Classification need target column
    return multiModelState.targetColumn !== null;
}

function setupModelEventListeners() {
    // Model selection listeners
    const modelSelects = document.querySelectorAll('.model-select');
    modelSelects.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const model = this.getAttribute('data-model');
            const category = this.getAttribute('data-category');
            
            if (this.checked) {
                // If no model type is selected yet, set it
                if (multiModelState.selectedModelType === null) {
                    multiModelState.selectedModelType = category;
                    multiModelState.selectedModels.add(model);
                    multiModelState.selectedMetrics[model] = [];
                } 
                // If this model is from the same type, allow it
                else if (multiModelState.selectedModelType === category) {
                    multiModelState.selectedModels.add(model);
                    multiModelState.selectedMetrics[model] = [];
                } 
                // If this model is from a different type, show error and uncheck
                else {
                    this.checked = false;
                    alert(`⚠️ You can only select models from a single type.\n\nYou have already selected ${multiModelState.selectedModelType} models.\n\nPlease clear your selections first if you want to switch to ${category} models.`);
                    return;
                }
            } else {
                // Uncheck the model
                multiModelState.selectedModels.delete(model);
                delete multiModelState.selectedMetrics[model];
                delete multiModelState.selectedHyperparameters[model]; 
                
                // If no models are selected anymore, reset the model type
                if (multiModelState.selectedModels.size === 0) {
                    multiModelState.selectedModelType = null;
                }
            }
            updateUI();
            renderHyperparametersPanel();
            renderTargetColumnSelection();  // NEW: Add this
        });
    });

    // Model search
    const searchInput = document.getElementById('model-search');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            filterModels(this.value.toLowerCase());
        });
    }
}

function setupMetricsPanel() {
    // This will be populated dynamically
    updateMetricsPanel();
    renderHyperparametersPanel();
}

function setupFileUpload() {
    const fileUploadArea = document.getElementById('file-upload-area');
    const fileInput = document.getElementById('model-file-input');

    if (fileUploadArea && fileInput) {
        // Click to upload
        fileUploadArea.addEventListener('click', function(e) {
            if (e.target === fileUploadArea || e.target.closest('.upload-text') || e.target.closest('.upload-icon')) {
                fileInput.click();
            }
        });

        // File selected
        fileInput.addEventListener('change', function(e) {
            if (this.files.length > 0) {
                handleFileSelected(this.files[0]);
            }
        });

        // Drag and drop
        fileUploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#1FB8CD';
            this.style.background = '#ecf8fb';
        });

        fileUploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#1FB8CD';
            this.style.background = '#f8feff';
        });

        fileUploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#1FB8CD';
            this.style.background = '#f8feff';
            if (e.dataTransfer.files.length > 0) {
                handleFileSelected(e.dataTransfer.files[0]);
            }
        });
    }
}

function renderHyperparametersPanel() {
    const container = document.getElementById('hyperparameters-panel');
    const section = document.getElementById('hyperparameters-section');

    if (!container || !section) {
        console.warn('Hyperparameters panel or section not found in DOM');
        return;
    }
    
    // Show section only if models are selected
    if (multiModelState.selectedModels.size === 0) {
        // clearHyperparameters();
        return;
    }

    let hyperparamsHTML = '';

    // Iterate through each selected model
    multiModelState.selectedModels.forEach(modelName => {
        let params = {}
        if(modelName === "Classification_OVR"){
            params = HYPERPARAMETERS["Classification_OneVsRest"];
        }
        else if(modelName === "Classification_MLPC"){
            params = HYPERPARAMETERS["Classification_MultilayerPerceptron"];
        }
        else{
            params = HYPERPARAMETERS[modelName];
        }
        
        if (!params) {
            console.warn(`No hyperparameters found for model: ${modelName}`);
            return;
        }

        // Initialize hyperparameters for this model if not already done
        if (!multiModelState.selectedHyperparameters[modelName]) {
            multiModelState.selectedHyperparameters[modelName] = {};
        }

        hyperparamsHTML += `<div class="model-hyperparams">`;
        hyperparamsHTML += `<div class="model-hyperparams-header">⚙️ ${modelName.replace(/_/g, ' ')}</div>`;

        // Iterate through each hyperparameter
        Object.keys(params).forEach(paramName => {
            const param = params[paramName];
            const paramId = `hyperparam-${modelName}-${paramName}`;
            const checkboxId = `hyperparam-checkbox-${modelName}-${paramName}`;
            const isSelected = multiModelState.selectedHyperparameters[modelName][paramName] !== undefined;
            
            // Main hyperparameter container
            hyperparamsHTML += `<div class="hyperparam-item">`;
            
            // Checkbox to enable/disable this hyperparameter
            hyperparamsHTML += `
                <div class="hyperparam-checkbox-container">
                    <input 
                        type="checkbox" 
                        id="${checkboxId}" 
                        class="hyperparam-checkbox"
                        ${isSelected ? 'checked' : ''}
                        onchange="toggleHyperparameter('${modelName}', '${paramName}', this.checked)"
                    >
                    <label for="${checkboxId}" class="hyperparam-checkbox-label">${paramName}</label>
                </div>
            `;
            
            // Control wrapper - shown only when checkbox is checked
            hyperparamsHTML += `<div class="hyperparam-control-wrapper" id="control-${paramId}" style="display: ${isSelected ? 'block' : 'none'}; margin-top: 8px; padding-left: 24px;">`;
            
            if (param.description) {
                hyperparamsHTML += `<span class="hyperparam-description">${param.description}</span>`;
            }

            // Categorical Type
            if (param.type === 'categorical') {
                hyperparamsHTML += `<select class="hyperparam-select" id="${paramId}" onchange="updateHyperparameter('${modelName}', '${paramName}', this.value)">`;
                hyperparamsHTML += `<option value="">-- Select an option --</option>`;
                
                param.options.forEach(option => {
                    const displayValue = option === null ? 'None' : option;
                    const selectedAttr = multiModelState.selectedHyperparameters[modelName][paramName] === option ? 'selected' : '';
                    hyperparamsHTML += `<option value="${option}" ${selectedAttr}>${displayValue}</option>`;
                });
                
                hyperparamsHTML += `</select>`;
            }
            // Integer Range Type
            else if (param.type === 'int_range') {
                const [min, max] = param.default_range;
                const mid = multiModelState.selectedHyperparameters[modelName][paramName] !== undefined 
                    ? multiModelState.selectedHyperparameters[modelName][paramName]
                    : Math.floor((min + max) / 2);
                
                hyperparamsHTML += `<div class="hyperparam-range-container">`;
                hyperparamsHTML += `<span class="hyperparam-range-label">${min}</span>`;
                hyperparamsHTML += `<input type="range" class="hyperparam-input" id="${paramId}" 
                    min="${min}" max="${max}" value="${mid}" 
                    onchange="updateHyperparameter('${modelName}', '${paramName}', this.value)">`;
                hyperparamsHTML += `<span class="hyperparam-value-display" id="${paramId}-value">${mid}</span>`;
                hyperparamsHTML += `<span class="hyperparam-range-label">${max}</span>`;
                hyperparamsHTML += `</div>`;
                
                // Store initial value only if checkbox is checked
                if (isSelected) {
                    multiModelState.selectedHyperparameters[modelName][paramName] = parseInt(mid);
                }
            }
            // Float Range Type
            else if (param.type === 'float_range') {
                const [min, max] = param.default_range;
                const mid = multiModelState.selectedHyperparameters[modelName][paramName] !== undefined 
                    ? multiModelState.selectedHyperparameters[modelName][paramName]
                    : ((min + max) / 2).toFixed(4);
                const step = ((max - min) / 100).toFixed(4);
                
                hyperparamsHTML += `<div class="hyperparam-range-container">`;
                hyperparamsHTML += `<span class="hyperparam-range-label">${min}</span>`;
                hyperparamsHTML += `<input type="range" class="hyperparam-input" id="${paramId}" 
                    min="${min}" max="${max}" step="${step}" value="${mid}" 
                    onchange="updateHyperparameter('${modelName}', '${paramName}', this.value)">`;
                hyperparamsHTML += `<span class="hyperparam-value-display" id="${paramId}-value">${mid}</span>`;
                hyperparamsHTML += `<span class="hyperparam-range-label">${max}</span>`;
                hyperparamsHTML += `</div>`;
                
                // Store initial value only if checkbox is checked
                if (isSelected) {
                    multiModelState.selectedHyperparameters[modelName][paramName] = parseFloat(mid);
                }
            }

            // Close control wrapper
            hyperparamsHTML += `</div>`;
            
            // Close hyperparam item
            hyperparamsHTML += `</div>`;
        });

        hyperparamsHTML += `</div>`;
    });

    container.innerHTML = hyperparamsHTML;
}

function toggleHyperparameter(modelName, paramName, isChecked) {
    const paramId = `hyperparam-${modelName}-${paramName}`;
    const controlWrapper = document.getElementById(`control-${paramId}`);
    
    if (!controlWrapper) {
        console.warn(`Control wrapper not found for ${modelName}.${paramName}`);
        return;
    }
    
    if (isChecked) {
        // Show the control wrapper
        controlWrapper.style.display = 'block';
        
        // Initialize the hyperparameter with default value
        const params = HYPERPARAMETERS[modelName];
        if (params && params[paramName]) {
            const param = params[paramName];
            
            if (param.type === 'categorical') {
                // Set to first option or empty
                multiModelState.selectedHyperparameters[modelName][paramName] = '';
            } else if (param.type === 'int_range') {
                const [min, max] = param.default_range;
                const mid = Math.floor((min + max) / 2);
                multiModelState.selectedHyperparameters[modelName][paramName] = mid;
            } else if (param.type === 'float_range') {
                const [min, max] = param.default_range;
                const mid = parseFloat(((min + max) / 2).toFixed(4));
                multiModelState.selectedHyperparameters[modelName][paramName] = mid;
            }
        }
    } else {
        // Hide the control wrapper
        controlWrapper.style.display = 'none';
        
        // Remove the hyperparameter from state
        delete multiModelState.selectedHyperparameters[modelName][paramName];
    }

    updateUI();
}

function updateHyperparameter(modelName, paramName, value) {
    // Update the value display for range inputs
    const displayElement = document.getElementById(`hyperparam-${modelName}-${paramName}-value`);
    if (displayElement) {
        displayElement.textContent = value;
    }

    // Convert to appropriate type
    let convertedValue = value;
    if (value !== '' && value !== null) {
        const numValue = Number(value);
        if (!isNaN(numValue)) {
            convertedValue = numValue;
        } else if (value === 'null' || value === 'None') {
            convertedValue = null;
        }
    }

    // Store the hyperparameter value
    multiModelState.selectedHyperparameters[modelName][paramName] = convertedValue;
    updateUI();
}

// ============================================
// PREPROCESSING OPTION MANAGEMENT
// ============================================

/**
 * Handle preprocessing option selection
 * @param {string} option - 'remove' or 'fill'
 * @param {HTMLElement} cardElement - The card that was clicked
 */

function selectSplitRatio(inputElement) {
    const value = parseFloat(inputElement.value);

    if (!isNaN(value) && value > 0 && value < 1) {
        multiModelState.splitRatio = value;
    } else {
        console.warn("⚠️ Invalid split ratio:", value);
    }
    
    updateUI();
}

/**
 * Handle split type selection
 * @param {string} type - 'random' or 'sequential'
 * @param {HTMLElement} cardElement - The card that was clicked
 */
function selectSplitType(type, cardElement) {
    // Update radio buttons
    const randomRadio = document.getElementById('split-random');
    const sequentialRadio = document.getElementById('split-sequential');
    
    if (randomRadio) {
        randomRadio.checked = (type === 'random');
    }
    if (sequentialRadio) {
        sequentialRadio.checked = (type === 'sequential');
    }
    
    // Update card styling
    const randomCard = document.getElementById('split-random-card');
    const sequentialCard = document.getElementById('split-sequential-card');
    
    if (randomCard) {
        randomCard.style.borderColor = (type === 'random') ? '#1FB8CD' : '#e0e0e0';
        randomCard.style.backgroundColor = (type === 'random') ? '#f0feff' : 'white';
    }
    
    if (sequentialCard) {
        sequentialCard.style.borderColor = (type === 'sequential') ? '#1FB8CD' : '#e0e0e0';
        sequentialCard.style.backgroundColor = (type === 'sequential') ? '#f0feff' : 'white';
    }
    
    // Update selected display
    const selectedDisplay = document.getElementById('split-type-selected');
    const selectedText = document.getElementById('selected-split-type-text');
    
    if (selectedDisplay && selectedText) {
        if (type === 'random') {
            selectedText.textContent = 'Random Split (shuffled data)';
        } else if (type === 'sequential') {
            selectedText.textContent = 'Split in Order (preserve sequence)';
        }
        selectedDisplay.style.display = 'block';
    }
    
    // Store in state
    multiModelState.splitType = type;
    updateUI();
}

/**
 * Get selected split type from radio button
 * @returns {string|null} - 'random', 'sequential', or null if none selected
 */
function getSelectedSplitType() {
    const selected = document.querySelector('input[name="splitType"]:checked');
    return selected ? selected.value : null;
}

/**
 * Validate split type selection
 * @returns {boolean} - true if split type is selected
 */
function isSplitTypeSelected() {
    return getSelectedSplitType() !== null;
}

function selectPreprocessingOption(option, cardElement) {
    const removeRadio = document.getElementById('preprocess-remove');
    const fillRadio = document.getElementById('preprocess-fill');
    
    if (removeRadio) {
        removeRadio.checked = (option === 'remove');
    }
    if (fillRadio) {
        fillRadio.checked = (option === 'fill');
    }
    
    // Update card styling
    const removeCard = document.getElementById('preprocess-remove-card');
    const fillCard = document.getElementById('preprocess-fill-card');
    
    if (removeCard) {
        removeCard.style.borderColor = (option === 'remove') ? '#1FB8CD' : '#e0e0e0';
        removeCard.style.backgroundColor = (option === 'remove') ? '#f0feff' : 'white';
    }
    
    if (fillCard) {
        fillCard.style.borderColor = (option === 'fill') ? '#1FB8CD' : '#e0e0e0';
        fillCard.style.backgroundColor = (option === 'fill') ? '#f0feff' : 'white';
    }
    
    // Update selected display
    const selectedDisplay = document.getElementById('preprocessing-selected');
    const selectedText = document.getElementById('selected-preprocessing-text');
    
    if (selectedDisplay && selectedText) {
        if (option === 'remove') {
            selectedText.textContent = 'Remove rows with null values';
        } else if (option === 'fill') {
            selectedText.textContent = 'Fill null values with mean/mode';
        }
        selectedDisplay.style.display = 'block';
    }
    
    // Store in state
    multiModelState.preprocessingOption = option;
    updateUI();
}

/**
 * Get selected preprocessing option from radio button
 * @returns {string|null} - 'remove', 'fill', or null if none selected
 */
function getSelectedPreprocessingOption() {
    const selected = document.querySelector('input[name="preprocessingOption"]:checked');
    return selected ? selected.value : null;
}

/**
 * Validate preprocessing selection
 * @returns {boolean} - true if preprocessing is selected
 */
function isPreprocessingSelected() {
    return getSelectedPreprocessingOption() !== null;
}

// ============================================
// MODEL FILTERING & DISPLAY
// ============================================

function filterModels(searchTerm) {
    const modelCheckboxes = document.querySelectorAll('.model-checkbox');
    
    modelCheckboxes.forEach(checkbox => {
        const label = checkbox.querySelector('.checkbox-label').textContent.toLowerCase();
        const isVisible = label.includes(searchTerm);
        checkbox.style.display = isVisible ? '' : 'none';
    });
}

function toggleCategory(element) {
    const categoryToggle = element;
    const categoryName = categoryToggle.querySelector('.category-name').textContent.trim();
    
    // Find the matching category models container
    const categoryModels = document.querySelector(
        `.category-models[data-category="${categoryName.split(' ')[1] || categoryName.split(' ')[0]}"]`
    );
    
    if (categoryModels) {
        const isCollapsed = categoryModels.classList.contains('collapsed');
        categoryModels.classList.toggle('collapsed');
        categoryToggle.classList.toggle('collapsed');
    }
}

// ============================================
// METRICS PANEL MANAGEMENT
// ============================================

function updateMetricsPanel() {
    const metricsPanel = document.getElementById('metrics-panel');
    
    if (multiModelState.selectedModels.size === 0) {
        metricsPanel.innerHTML = `
            <div class="empty-state">
                <p class="empty-icon">📋</p>
                <p class="empty-text">Select models to see available metrics</p>
            </div>
        `;
        return;
    }

    // Group selected models by their metrics type
    const metricsByGroup = {};
    const displayedMetrics = new Set();

    multiModelState.selectedModels.forEach(model => {
        const metrics = MODEL_METRICS[model] || [];
        const modelType = model.split('_')[0]; // Regression, Classification, Clustering
        
        if (!metricsByGroup[modelType]) {
            metricsByGroup[modelType] = {
                type: modelType,
                metrics: new Set()
            };
        }

        metrics.forEach(metric => {
            metricsByGroup[modelType].metrics.add(metric);
            displayedMetrics.add(metric);
        });
    });

    // Generate HTML for metrics panel
    let html = '';
    const typeOrder = ['Classification', 'Regression', 'Clustering'];
    
    typeOrder.forEach(type => {
        if (metricsByGroup[type]) {
            const group = metricsByGroup[type];
            const typeEmoji = {
                'Classification': '🎯',
                'Regression': '📊',
                'Clustering': '👥'
            }[type];

            html += `
                <div class="metric-group">
                    <div class="metric-group-header">${typeEmoji} ${type} Metrics</div>
            `;

            Array.from(group.metrics).sort().forEach(metric => {
                const metricId = `metric-${metric.replace(/\s+/g, '-').toLowerCase()}`;
                const isChecked = isMetricSelected(metric);

                html += `
                    <div class="metric-checkbox">
                        <input 
                            type="checkbox" 
                            id="${metricId}" 
                            data-metric="${metric}"
                            ${isChecked ? 'checked' : ''}
                            onchange="handleMetricChange(this)"
                        >
                        <label for="${metricId}">${metric}</label>
                    </div>
                `;
            });

            html += `</div>`;
        }
    });

    metricsPanel.innerHTML = html;
}

function isMetricSelected(metric) {
    // Check if metric is selected in any of the selected models
    for (let model of multiModelState.selectedModels) {
        if (multiModelState.selectedMetrics[model] && 
            multiModelState.selectedMetrics[model].includes(metric)) {
            return true;
        }
    }
    return false;
}

function handleMetricChange(checkbox) {
    const metric = checkbox.getAttribute('data-metric');
    const isChecked = checkbox.checked;

    // Update metrics for ALL selected models that support this metric
    multiModelState.selectedModels.forEach(model => {
        const supportedMetrics = MODEL_METRICS[model] || [];
        
        if (supportedMetrics.includes(metric)) {
            if (isChecked) {
                if (!multiModelState.selectedMetrics[model].includes(metric)) {
                    multiModelState.selectedMetrics[model].push(metric);
                }
            } else {
                multiModelState.selectedMetrics[model] = 
                    multiModelState.selectedMetrics[model].filter(m => m !== metric);
            }
        }
    });

    updateUI();
    renderHyperparametersPanel();
}

// ============================================
// FILE UPLOAD MANAGEMENT
// ============================================

async function handleFileSelected(file) {
    if (!file.name.endsWith('.csv')) {
        alert('❌ Please upload a CSV file');
        return;
    }
    multiModelState.uploadedFile = file;
    
    // Load column names
    try {
        await loadDatasetColumns(file);
    } catch (error) {
        console.error('Error loading columns:', error);
        alert('❌ Error reading CSV file columns');
        return;
    }
    
    updateFileUploadUI();
    renderTargetColumnSelection();
    updateUI();
}

function updateFileUploadUI() {
    const uploadArea = document.getElementById('file-upload-area');
    const statusArea = document.getElementById('file-upload-status');
    const fileInput = document.getElementById('model-file-input');

    if (multiModelState.uploadedFile) {
        uploadArea.style.display = 'none';
        statusArea.classList.remove('hidden');
        document.getElementById('uploaded-filename').textContent = 
            `📄 ${multiModelState.uploadedFile.name}`;
    } else {
        uploadArea.style.display = 'block';
        statusArea.classList.add('hidden');
    }
}

function resetFileUpload() {
    multiModelState.uploadedFile = null;
    const fileInput = document.getElementById('model-file-input');
    if (fileInput) fileInput.value = '';
    updateFileUploadUI();
    updateUI();
}

// ============================================
// UI UPDATE FUNCTIONS
// ============================================

function updateUI() {
    updateModelCount();
    updateMetricCount();
    updateMetricsPanel();
    updateRunButtonState();
    updateSummarySection();
}

function updateModelCount() {
    const badge = document.getElementById('model-count');
    const count = multiModelState.selectedModels.size;
    badge.textContent = `${count} selected`;
}

function updateMetricCount() {
    const badge = document.getElementById('metric-count');
    const metrics = new Set();
    
    Object.values(multiModelState.selectedMetrics).forEach(modelMetrics => {
        modelMetrics.forEach(metric => metrics.add(metric));
    });
    
    badge.textContent = `${metrics.size} selected`;
}

function updateRunButtonState() {
    const runBtn = document.getElementById('run-models-btn');
    
    if (!runBtn) {
        console.error('❌ Run models button not found!');
        return;
    }

    const isValid = 
        multiModelState.selectedModels.size > 0 &&           // At least one model
        multiModelState.uploadedFile !== null &&             // Dataset uploaded
        getTotalSelectedMetrics() > 0 &&                     // At least one metric
        isPreprocessingSelected() &&                         // Preprocessing option selected
        isSplitTypeSelected() &&                             // Split type selected
        isTargetColumnSelected() &&                          // Target column selected (if needed)
        areHyperparametersConfigured();                      // At least one hyperparameter per model
    
    // ✅ FORCE the button state - remove disabled attribute entirely or add it back
    if (isValid) {
        runBtn.disabled = false;
        runBtn.removeAttribute('disabled');
        runBtn.style.opacity = '1';
        runBtn.style.cursor = 'pointer';
        runBtn.style.pointerEvents = 'auto';
        runBtn.classList.remove('disabled');
    } else {
        runBtn.disabled = true;
        runBtn.setAttribute('disabled', 'disabled');
        runBtn.style.opacity = '0.5';
        runBtn.style.cursor = 'not-allowed';
        runBtn.style.pointerEvents = 'none';
        runBtn.classList.add('disabled');
    }
}

/**
 * Check if at least one hyperparameter is configured for each selected model
 * @returns {boolean} - true if all models have at least one hyperparameter configured
 */
function areHyperparametersConfigured() {
    // If no models are selected, return false
    if (multiModelState.selectedModels.size === 0) {
        return false;
    }
    
    // Check each selected model
    for (let modelName of multiModelState.selectedModels) {
        const hyperparams = multiModelState.selectedHyperparameters[modelName];
        
        // Check if this model has any hyperparameters configured
        if (!hyperparams || Object.keys(hyperparams).length === 0) {
            return false;
        }
    }
    
    return true;
}

function updateSummarySection() {
    const summarySection = document.getElementById('summary-section');
    const shouldShow = 
        multiModelState.selectedModels.size > 0 &&
        multiModelState.uploadedFile !== null &&
        getTotalSelectedMetrics() > 0;

    if (shouldShow) {
        summarySection.classList.toggle('hidden', !shouldShow);
        
        document.getElementById('summary-models').textContent = 
            multiModelState.selectedModels.size;
        
        document.getElementById('summary-metrics').textContent = 
            getTotalSelectedMetrics();
        
        document.getElementById('summary-file').textContent = 
            multiModelState.uploadedFile?.name || 'Not selected';
    } else {
        summarySection.classList.add('hidden');
    }
}

function getTotalSelectedMetrics() {
    const metrics = new Set();
    Object.values(multiModelState.selectedMetrics).forEach(modelMetrics => {
        modelMetrics.forEach(metric => metrics.add(metric));
    });
    return metrics.size;
}

// ============================================
// ACTION FUNCTIONS
// ============================================

function clearAllSelections() {
    // Clear model selections
    document.querySelectorAll('.model-select').forEach(checkbox => {
        checkbox.checked = false;
    });

    // Clear metrics
    document.querySelectorAll('.metric-checkbox input').forEach(checkbox => {
        checkbox.checked = false;
    });

    // Clear preprocessing
    document.querySelectorAll('input[name="preprocessingOption"]').forEach(radio => {
        radio.checked = false;
    });
    const preprocessingDisplay = document.getElementById('preprocessing-selected');
    if (preprocessingDisplay) {
        preprocessingDisplay.style.display = 'none';
    }

    // Clear split type
    document.querySelectorAll('input[name="splitType"]').forEach(radio => {
        radio.checked = false;
    });
    const splitTypeDisplay = document.getElementById('split-type-selected');
    if (splitTypeDisplay) {
        splitTypeDisplay.style.display = 'none';
    }

    // Reset split type card styling
    const randomCard = document.getElementById('split-random-card');
    const sequentialCard = document.getElementById('split-sequential-card');
    if (randomCard) {
        randomCard.style.borderColor = '#e0e0e0';
        randomCard.style.backgroundColor = 'white';
    }
    if (sequentialCard) {
        sequentialCard.style.borderColor = '#e0e0e0';
        sequentialCard.style.backgroundColor = 'white';
    }

    // Clear state
    multiModelState.selectedModels.clear();
    multiModelState.selectedMetrics = {};
    multiModelState.uploadedFile = null;
    multiModelState.selectedModelType = null;
    multiModelState.preprocessingOption = null;
    multiModelState.splitType = null;
    multiModelState.splitRatio = 0.2;
    multiModelState.selectedHyperparameters = {};
    multiModelState.targetColumn = null;  // NEW: Add this
    multiModelState.availableColumns = [];  // NEW: Add this

    // Reset file input
    const fileInput = document.getElementById('model-file-input');
    if (fileInput) fileInput.value = '';

    updateUI();
    updateModelCount();
    updateMetricsPanel();
    renderHyperparametersPanel();
    updateRunButtonState();
    renderTargetColumnSelection();  // NEW: Add this
}

async function runMultipleModels() {
    // Validate selections
    if (multiModelState.selectedModels.size === 0) {
        alert('❌ Please select at least one model');
        return;
    }

    if (!multiModelState.uploadedFile) {
        alert('❌ Please upload a CSV file');
        return;
    }

    if (getTotalSelectedMetrics() === 0) {
        alert('❌ Please select at least one metric');
        return;
    }

    if (!isTargetColumnSelected()) {
        alert('❌ Please select a target column');
        return;
    }

    // ✅ GET PREPROCESSING OPTION DIRECTLY FROM RADIO BUTTON
    const preprocessingOption = getSelectedPreprocessingOption();
    if (!preprocessingOption) {
        alert('❌ Please select a preprocessing method');
        return;
    }

    // ✅ GET SPLIT TYPE DIRECTLY FROM RADIO BUTTON
    const splitType = getSelectedSplitType();
    if (!splitType) {
        alert('❌ Please select a split method (Random or Sequential)');
        return;
    }

    // Show loading modal
    showLoadingModal('Training multiple models... This may take a few minutes.');

    const formData = new FormData();

    // file
    formData.append('file', multiModelState.uploadedFile);

    // metadata
    formData.append('models', JSON.stringify(buildModelsConfig()));
    formData.append('hyperparameters', JSON.stringify(multiModelState.selectedHyperparameters)); 
    formData.append('selectedModelType', multiModelState.selectedModelType);
    formData.append('preprocessingOption', preprocessingOption);
    formData.append('splitType', splitType);
    formData.append('splitRatio', multiModelState.splitRatio);
    formData.append('total_models', multiModelState.selectedModels.size);
    formData.append('total_metrics', getTotalSelectedMetrics());
    formData.append('splitType', multiModelState.splitType);
    formData.append('targetColumn', multiModelState.targetColumn || '');

    try {
        const response = await fetch('http://localhost:5001/run-batch-models', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        hideLoadingModal();
        
        if (result.success) {
            showSuccessMessage(`✅ Models training completed!`);
            
            // Reset the form
            if (typeof resetModelRunForm === 'function') {
                resetModelRunForm();
            }
            clearAllSelections();
            
            // Optionally refresh experiments list
            setTimeout(() => {
                if (typeof renderExperimentsTable === 'function') {
                    renderExperimentsTable();
                }
            }, 1000);
        } else {
            throw new Error(result.error || 'Unknown error occurred');
        }
    } catch (error) {
        hideLoadingModal();
        alert('❌ Error: ' + error.message);
        console.error('Error details:', error);
    }
}

function buildModelsConfig() {
    const modelsConfig = {};
    multiModelState.selectedModels.forEach(model => {
        modelsConfig[model] = {
            name: model,
            metrics: multiModelState.selectedMetrics[model] || []
        };
    });
    return modelsConfig;
}

// ============================================
// HELPER FUNCTIONS
// ============================================

function showLoadingModal(message) {
    const modal = document.createElement('div');
    modal.id = 'batch-loading-modal';
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;
    modal.innerHTML = `
        <div style="
            background: white;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            min-width: 300px;
        ">
            <div style="font-size: 50px; margin-bottom: 20px; animation: spin 1s linear infinite;">⚙️</div>
            <p style="
                margin: 0 0 20px 0;
                font-size: 18px;
                color: #1a1a1a;
                font-weight: 600;
            ">${message}</p>
            <div style="display: flex; justify-content: center; gap: 6px;">
                <div style="
                    width: 8px;
                    height: 8px;
                    background: #1FB8CD;
                    border-radius: 50%;
                    animation: bounce 1.4s infinite;
                "></div>
                <div style="
                    width: 8px;
                    height: 8px;
                    background: #1FB8CD;
                    border-radius: 50%;
                    animation: bounce 1.4s infinite 0.2s;
                "></div>
                <div style="
                    width: 8px;
                    height: 8px;
                    background: #1FB8CD;
                    border-radius: 50%;
                    animation: bounce 1.4s infinite 0.4s;
                "></div>
            </div>
        </div>
    `;

    // Add styles for animations
    if (!document.getElementById('batch-loading-styles')) {
        const style = document.createElement('style');
        style.id = 'batch-loading-styles';
        style.textContent = `
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            @keyframes bounce {
                0%, 100% { opacity: 0.3; }
                50% { opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(modal);
}

function hideLoadingModal() {
    const modal = document.getElementById('batch-loading-modal');
    if (modal) {
        modal.remove();
    }
}

function showSuccessMessage(message) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background-color: #d4edda;
        color: #155724;
        padding: 16px 24px;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10001;
        font-weight: 500;
        max-width: 400px;
        animation: toastSlideIn 0.4s ease-out;
    `;
    toast.textContent = message;

    if (!document.getElementById('toast-styles')) {
        const style = document.createElement('style');
        style.id = 'toast-styles';
        style.textContent = `
            @keyframes toastSlideIn {
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
    }

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastSlideIn 0.4s ease-out reverse';
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// ============================================
// EXPORT FUNCTIONS TO GLOBAL SCOPE
// ============================================
// These functions MUST be accessible from HTML onclick/onchange attributes

window.toggleCategory = toggleCategory;
window.filterModels = filterModels;
window.handleMetricChange = handleMetricChange;
window.resetFileUpload = resetFileUpload;
window.clearAllSelections = clearAllSelections;
window.runMultipleModels = runMultipleModels;
window.toggleHyperparameter = toggleHyperparameter;
window.updateHyperparameter = updateHyperparameter;
window.selectPreprocessingOption = selectPreprocessingOption;
window.selectSplitRatio = selectSplitRatio;
window.selectSplitType = selectSplitType;
window.getSelectedPreprocessingOption = getSelectedPreprocessingOption;
window.isPreprocessingSelected = isPreprocessingSelected;
window.getSelectedSplitType = getSelectedSplitType;  // ✅ NEW: Export getter
window.isSplitTypeSelected = isSplitTypeSelected;  // ✅ NEW: Export validator
window.selectTargetColumn = selectTargetColumn;
window.isTargetColumnSelected = isTargetColumnSelected;
window.areHyperparametersConfigured = areHyperparametersConfigured;
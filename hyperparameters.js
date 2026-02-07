const HYPERPARAMETERS = {
    // ==================== REGRESSION MODELS ====================
    
    'Regression_LinearRegression': {
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum number of iterations',
            'default_range': [50, 300]
        },
        'regParam': {
            'type': 'float_range',
            'description': 'Regularization parameter',
            'default_range': [0.0, 1.0]
        },
        'elasticNetParam': {
            'type': 'float_range',
            'description': 'ElasticNet mixing parameter (0 = L2, 1 = L1)',
            'default_range': [0.0, 1.0]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Convergence tolerance',
            'default_range': [1e-8, 1e-4]
        },
        'fitIntercept': {
            'type': 'categorical',
            'description': 'Whether to fit an intercept term',
            'options': [true, false]
        },
        'standardization': {
            'type': 'categorical',
            'description': 'Whether to standardize features',
            'options': [true, false]
        },
        'solver': {
            'type': 'categorical',
            'description': 'Solver algorithm',
            'options': ['auto', 'normal', 'l-bfgs']
        }
    },

    'Regression_DecisionTree': {
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum depth of tree',
            'default_range': [2, 30]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances each child must have',
            'default_range': [1, 20]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Regression_RandomForest': {
        'numTrees': {
            'type': 'int_range',
            'description': 'Number of trees in the forest',
            'default_range': [10, 200]
        },
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum depth of each tree',
            'default_range': [5, 30]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances per node',
            'default_range': [1, 20]
        },
        'subsamplingRate': {
            'type': 'float_range',
            'description': 'Fraction of data to sample per iteration',
            'default_range': [0.5, 1.0]
        },
        'featureSubsetStrategy': {
            'type': 'categorical',
            'description': 'Number of features to consider per split',
            'options': ['auto', 'all', 'sqrt', 'log2', 'onethird']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Regression_GBTRegressor': {
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum number of iterations/trees',
            'default_range': [20, 200]
        },
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum depth of each tree',
            'default_range': [3, 15]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances per node',
            'default_range': [1, 20]
        },
        'stepSize': {
            'type': 'float_range',
            'description': 'Learning rate/step size',
            'default_range': [0.01, 0.5]
        },
        'subsamplingRate': {
            'type': 'float_range',
            'description': 'Fraction of data to sample per iteration',
            'default_range': [0.5, 1.0]
        },
        'lossType': {
            'type': 'categorical',
            'description': 'Loss function',
            'options': ['absolute_error', 'squared_error', 'quantile', 'huber']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    // ==================== CLASSIFICATION MODELS ====================
    
    'Classification_LogisticRegression': {
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [50, 300]
        },
        'regParam': {
            'type': 'float_range',
            'description': 'Regularization parameter',
            'default_range': [0.0, 1.0]
        },
        'elasticNetParam': {
            'type': 'float_range',
            'description': 'ElasticNet mixing parameter (0 = L2, 1 = L1)',
            'default_range': [0.0, 1.0]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Convergence tolerance',
            'default_range': [1e-8, 1e-4]
        },
        'fitIntercept': {
            'type': 'categorical',
            'description': 'Whether to fit an intercept term',
            'options': [true, false]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_DecisionTree': {
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum tree depth',
            'default_range': [2, 30]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances per node',
            'default_range': [1, 20]
        },
        'impurity': {
            'type': 'categorical',
            'description': 'Impurity measure',
            'options': ['gini', 'entropy']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_RandomForest': {
        'numTrees': {
            'type': 'int_range',
            'description': 'Number of trees',
            'default_range': [10, 200]
        },
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum depth per tree',
            'default_range': [5, 30]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances per node',
            'default_range': [1, 20]
        },
        'featureSubsetStrategy': {
            'type': 'categorical',
            'description': 'Feature subset strategy',
            'options': ['auto', 'all', 'sqrt', 'log2', 'onethird']
        },
        'impurity': {
            'type': 'categorical',
            'description': 'Impurity measure',
            'options': ['gini', 'entropy']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_GBTClassifier': {
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [20, 200]
        },
        'maxDepth': {
            'type': 'int_range',
            'description': 'Maximum tree depth',
            'default_range': [3, 15]
        },
        'minInstancesPerNode': {
            'type': 'int_range',
            'description': 'Minimum instances per node',
            'default_range': [1, 20]
        },
        'stepSize': {
            'type': 'float_range',
            'description': 'Learning rate',
            'default_range': [0.01, 0.5]
        },
        'subsamplingRate': {
            'type': 'float_range',
            'description': 'Data subsampling rate',
            'default_range': [0.5, 1.0]
        },
        'lossType': {
            'type': 'categorical',
            'description': 'Loss function',
            'options': ['log_loss', 'exponential']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_NaiveBayes': {
        'smoothing': {
            'type': 'float_range',
            'description': 'Smoothing parameter (variance smoothing)',
            'default_range': [1e-10, 1e-8]
        }
    },

    'Classification_MultilayerPerceptron': {
        'layers': {
            'type': 'categorical',
            'description': 'Layer sizes including input and output (input_features, hidden1, hidden2, ..., num_classes)',
            'options': [
                [10, 15, 10, 2],
                [10, 20, 10, 2],
                [10, 32, 16, 2],
                [10, 64, 32, 2],
                [10, 128, 64, 2],
                [20, 30, 20, 2],
                [20, 50, 30, 2],
                [20, 100, 50, 2],
                [50, 75, 50, 2],
                [100, 150, 100, 2]
            ]
        },
        'activation': {
            'type': 'categorical',
            'description': 'Activation function',
            'options': ['relu', 'tanh', 'logistic', 'identity']
        },
        'solver': {
            'type': 'categorical',
            'description': 'Solver algorithm',
            'options': ['adam', 'sgd']
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [50, 500]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Convergence tolerance',
            'default_range': [1e-8, 1e-4]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_LinearSVC': {
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [500, 5000]
        },
        'C': {
            'type': 'float_range',
            'description': 'Inverse regularization strength (higher C = less regularization)',
            'default_range': [0.01, 10.0]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Convergence tolerance',
            'default_range': [1e-5, 1e-3]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Classification_OneVsRest': {
        'baseClassifier': {
            'type': 'categorical',
            'description': 'Base classifier to use',
            'options': ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'NaiveBayes', 'LinearSVC']
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Max iterations for base classifier',
            'default_range': [50, 300]
        },
        'regParam': {
            'type': 'float_range',
            'description': 'Regularization parameter for base classifier',
            'default_range': [0.0, 1.0]
        }
    },

    // ==================== CLUSTERING MODELS ====================
    
    'Clustering_KMeans': {
        'k': {
            'type': 'int_range',
            'description': 'Number of clusters',
            'default_range': [2, 20]
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [10, 100]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Tolerance for convergence',
            'default_range': [1e-5, 1e-2]
        },
        'n_init': {
            'type': 'int_range',
            'description': 'Number of initializations',
            'default_range': [5, 20]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Clustering_BisectingKMeans': {
        'k': {
            'type': 'int_range',
            'description': 'Number of clusters',
            'default_range': [2, 20]
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [10, 100]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Tolerance for convergence',
            'default_range': [1e-5, 1e-2]
        },
        'n_init': {
            'type': 'int_range',
            'description': 'Number of initializations',
            'default_range': [5, 20]
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Clustering_GaussianMixture': {
        'k': {
            'type': 'int_range',
            'description': 'Number of Gaussian components',
            'default_range': [2, 20]
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum EM iterations',
            'default_range': [10, 200]
        },
        'tol': {
            'type': 'float_range',
            'description': 'Convergence tolerance',
            'default_range': [1e-5, 1e-1]
        },
        'covarianceType': {
            'type': 'categorical',
            'description': 'Type of covariance parameters',
            'options': ['full', 'tied', 'diag', 'spherical']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    },

    'Clustering_LDA': {
        'k': {
            'type': 'int_range',
            'description': 'Number of topics',
            'default_range': [2, 50]
        },
        'maxIter': {
            'type': 'int_range',
            'description': 'Maximum iterations',
            'default_range': [10, 100]
        },
        'optimizer': {
            'type': 'categorical',
            'description': 'Optimizer type',
            'options': ['online', 'batch']
        },
        'seed': {
            'type': 'int_range',
            'description': 'Random seed',
            'default_range': [0, 100]
        }
    }
};

window.HYPERPARAMETERS = HYPERPARAMETERS;
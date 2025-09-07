# AI Shield Test Data Summary

Generated on: 2025-08-29 19:37:22

## Overview
- **Total Files**: 191
- **Datasets**: 10
- **Models**: 51

## Datasets

### moons
- **Samples**: 800
- **Features**: 2
- **File**: `moons/moons_dataset.csv`

### circles
- **Samples**: 600
- **Features**: 2
- **File**: `circles/circles_dataset.csv`

### binary_complex
- **Samples**: 2000
- **Features**: 20
- **File**: `binary_complex/binary_complex_dataset.csv`

### high_dimensional
- **Samples**: 200
- **Features**: 50
- **File**: `edge_cases/high_dimensional_dataset.csv`

### small
- **Samples**: 50
- **Features**: 3
- **File**: `edge_cases/small_dataset.csv`

### medical_diagnosis
- **Samples**: 800
- **Features**: 6
- **File**: `medical_diagnosis/medical_diagnosis_dataset.csv`

### blobs
- **Samples**: 1200
- **Features**: 8
- **File**: `blobs/blobs_dataset.csv`

### binary_simple
- **Samples**: 1000
- **Features**: 5
- **File**: `binary_simple/binary_simple_dataset.csv`

### credit_risk
- **Samples**: 1000
- **Features**: 5
- **File**: `credit_risk/credit_risk_dataset.csv`

### multiclass
- **Samples**: 1500
- **Features**: 10
- **File**: `multiclass/multiclass_dataset.csv`

## Models by Dataset

### binary_complex
- **logistic_regression**: `binary_complex/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `binary_complex/decision_tree/decision_tree_model.joblib`
- **random_forest**: `binary_complex/random_forest/random_forest_model.joblib`
- **neural_network**: `binary_complex/neural_network/neural_network_model.joblib`
- **svm**: `binary_complex/svm/svm_model.joblib`
- **gradient_boosting**: `binary_complex/gradient_boosting/gradient_boosting_model.joblib`
### binary_simple
- **logistic_regression**: `binary_simple/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `binary_simple/decision_tree/decision_tree_model.joblib`
- **random_forest**: `binary_simple/random_forest/random_forest_model.joblib`
- **neural_network**: `binary_simple/neural_network/neural_network_model.joblib`
- **svm**: `binary_simple/svm/svm_model.joblib`
- **gradient_boosting**: `binary_simple/gradient_boosting/gradient_boosting_model.joblib`
### blobs
- **logistic_regression**: `blobs/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `blobs/decision_tree/decision_tree_model.joblib`
- **random_forest**: `blobs/random_forest/random_forest_model.joblib`
- **neural_network**: `blobs/neural_network/neural_network_model.joblib`
- **svm**: `blobs/svm/svm_model.joblib`
- **gradient_boosting**: `blobs/gradient_boosting/gradient_boosting_model.joblib`
### circles
- **logistic_regression**: `circles/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `circles/decision_tree/decision_tree_model.joblib`
- **random_forest**: `circles/random_forest/random_forest_model.joblib`
- **neural_network**: `circles/neural_network/neural_network_model.joblib`
- **svm**: `circles/svm/svm_model.joblib`
- **gradient_boosting**: `circles/gradient_boosting/gradient_boosting_model.joblib`
### credit_risk
- **logistic_regression**: `credit_risk/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `credit_risk/decision_tree/decision_tree_model.joblib`
- **random_forest**: `credit_risk/random_forest/random_forest_model.joblib`
- **neural_network**: `credit_risk/neural_network/neural_network_model.joblib`
- **svm**: `credit_risk/svm/svm_model.joblib`
- **gradient_boosting**: `credit_risk/gradient_boosting/gradient_boosting_model.joblib`
### medical_diagnosis
- **logistic_regression**: `medical_diagnosis/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `medical_diagnosis/decision_tree/decision_tree_model.joblib`
- **random_forest**: `medical_diagnosis/random_forest/random_forest_model.joblib`
- **neural_network**: `medical_diagnosis/neural_network/neural_network_model.joblib`
- **svm**: `medical_diagnosis/svm/svm_model.joblib`
- **gradient_boosting**: `medical_diagnosis/gradient_boosting/gradient_boosting_model.joblib`
### moons
- **logistic_regression**: `moons/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `moons/decision_tree/decision_tree_model.joblib`
- **random_forest**: `moons/random_forest/random_forest_model.joblib`
- **neural_network**: `moons/neural_network/neural_network_model.joblib`
- **svm**: `moons/svm/svm_model.joblib`
- **gradient_boosting**: `moons/gradient_boosting/gradient_boosting_model.joblib`
### multiclass
- **logistic_regression**: `multiclass/logistic_regression/logistic_regression_model.joblib`
- **decision_tree**: `multiclass/decision_tree/decision_tree_model.joblib`
- **random_forest**: `multiclass/random_forest/random_forest_model.joblib`
- **neural_network**: `multiclass/neural_network/neural_network_model.joblib`
- **svm**: `multiclass/svm/svm_model.joblib`
- **gradient_boosting**: `multiclass/gradient_boosting/gradient_boosting_model.joblib`
### test_data
- **small_dataset**: `edge_cases/small_dataset_model.joblib`
- **high_dimensional**: `edge_cases/high_dimensional_model.joblib`
- **feature_mismatch**: `edge_cases/feature_mismatch_model.joblib`

## Usage Instructions

1. Use the web interface: `./start_web.sh`
2. Navigate to http://localhost:5001/upload
3. Upload any model (.joblib or .pkl) and corresponding dataset (.csv)
4. Run adversarial analysis

## Test Scenarios

- **Normal cases**: All dataset folders contain compatible model-data pairs
- **Edge cases**: `edge_cases/` folder contains intentional mismatches for robustness testing
- **Feature alignment**: Test with different feature counts between model and data
- **Various algorithms**: RF, LR, SVM, GB, DT, NN models available
- **Different data types**: Binary/multi-class, linear/non-linear, synthetic/realistic

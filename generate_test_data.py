#!/usr/bin/env python3
"""
AI Shield Test Data Generator
Creates sample models and datasets for testing adversarial ML security analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification, make_blobs, make_moons, make_circles
from sklearn.metrics import accuracy_score, classification_report

def create_output_dir():
    """Create test_data directory if it doesn't exist"""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    return test_dir

def generate_synthetic_datasets():
    """Generate various synthetic datasets for testing"""
    datasets = {}
    
    print("📊 Generating synthetic datasets...")
    
    # 1. Binary Classification - Simple
    X1, y1 = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    datasets['binary_simple'] = {
        'X': X1, 'y': y1,
        'name': 'Binary Classification (Simple)',
        'features': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
        'target': 'class'
    }
    
    # 2. Binary Classification - Complex
    X2, y2 = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        random_state=123
    )
    datasets['binary_complex'] = {
        'X': X2, 'y': y2,
        'name': 'Binary Classification (Complex)',
        'features': [f'feature_{i+1}' for i in range(20)],
        'target': 'class'
    }
    
    # 3. Multi-class Classification
    X3, y3 = make_classification(
        n_samples=1500,
        n_features=10,
        n_informative=8,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=456
    )
    datasets['multiclass'] = {
        'X': X3, 'y': y3,
        'name': 'Multi-class Classification',
        'features': [f'feature_{i+1}' for i in range(10)],
        'target': 'class'
    }
    
    # 4. Non-linear separable data (Moons)
    X4, y4 = make_moons(n_samples=800, noise=0.1, random_state=789)
    datasets['moons'] = {
        'X': X4, 'y': y4,
        'name': 'Non-linear (Moons)',
        'features': ['x_coord', 'y_coord'],
        'target': 'class'
    }
    
    # 5. Circular data
    X5, y5 = make_circles(n_samples=600, noise=0.05, factor=0.6, random_state=101)
    datasets['circles'] = {
        'X': X5, 'y': y5,
        'name': 'Non-linear (Circles)',
        'features': ['x_coord', 'y_coord'],
        'target': 'class'
    }
    
    # 6. Blob clusters
    X6, y6 = make_blobs(n_samples=1200, centers=4, n_features=8, random_state=202)
    datasets['blobs'] = {
        'X': X6, 'y': y6,
        'name': 'Blob Clusters',
        'features': [f'feature_{i+1}' for i in range(8)],
        'target': 'cluster'
    }
    
    print(f"✅ Generated {len(datasets)} synthetic datasets")
    return datasets

def generate_real_world_like_datasets():
    """Generate datasets that mimic real-world scenarios"""
    datasets = {}
    
    print("🌍 Generating real-world-like datasets...")
    
    # 1. Credit Risk Dataset (Financial)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate correlated financial features
    age = np.random.normal(40, 12, n_samples)
    income = np.random.normal(50000, 20000, n_samples) + age * 1000
    credit_score = np.random.normal(650, 100, n_samples) + income * 0.001
    debt_ratio = np.random.beta(2, 5, n_samples) + np.random.normal(0, 0.1, n_samples)
    employment_length = np.random.exponential(5, n_samples)
    
    # Create target (default risk) based on features
    risk_score = (
        -0.01 * credit_score +
        0.000005 * income +
        2 * debt_ratio +
        -0.1 * employment_length +
        np.random.normal(0, 0.5, n_samples)
    )
    default_risk = (risk_score > np.percentile(risk_score, 70)).astype(int)
    
    credit_data = np.column_stack([age, income, credit_score, debt_ratio, employment_length])
    
    datasets['credit_risk'] = {
        'X': credit_data,
        'y': default_risk,
        'name': 'Credit Risk Prediction',
        'features': ['age', 'annual_income', 'credit_score', 'debt_to_income_ratio', 'employment_length_years'],
        'target': 'default_risk'
    }
    
    # 2. Medical Diagnosis Dataset
    np.random.seed(123)
    n_samples = 800
    
    # Generate medical features
    temperature = np.random.normal(98.6, 2, n_samples)
    blood_pressure_sys = np.random.normal(120, 20, n_samples)
    blood_pressure_dia = np.random.normal(80, 15, n_samples)
    heart_rate = np.random.normal(70, 15, n_samples)
    white_cell_count = np.random.lognormal(2, 0.5, n_samples)
    age = np.random.uniform(18, 80, n_samples)
    
    # Create diagnosis based on symptoms
    severity_score = (
        0.1 * (temperature - 98.6) +
        0.02 * (blood_pressure_sys - 120) +
        0.01 * heart_rate +
        0.1 * white_cell_count +
        0.05 * age +
        np.random.normal(0, 1, n_samples)
    )
    diagnosis = (severity_score > np.percentile(severity_score, 75)).astype(int)
    
    medical_data = np.column_stack([temperature, blood_pressure_sys, blood_pressure_dia, 
                                   heart_rate, white_cell_count, age])
    
    datasets['medical_diagnosis'] = {
        'X': medical_data,
        'y': diagnosis,
        'name': 'Medical Diagnosis',
        'features': ['temperature_f', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'wbc_count', 'age'],
        'target': 'requires_treatment'
    }
    
    print(f"✅ Generated {len(datasets)} real-world-like datasets")
    return datasets

def create_models_for_dataset(X, y, dataset_name):
    """Create and train multiple models for a given dataset"""
    models = {}
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  🔧 Training models for {dataset_name}...")
    
    # 1. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf.predict(X_test))
    models['random_forest'] = {
        'model': rf,
        'scaler': None,
        'accuracy': rf_accuracy,
        'type': 'tree_based'
    }
    
    # 2. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_accuracy = accuracy_score(y_test, lr.predict(X_test_scaled))
    models['logistic_regression'] = {
        'model': lr,
        'scaler': scaler,
        'accuracy': lr_accuracy,
        'type': 'linear'
    }
    
    # 3. Support Vector Machine
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_accuracy = accuracy_score(y_test, svm.predict(X_test_scaled))
    models['svm'] = {
        'model': svm,
        'scaler': scaler,
        'accuracy': svm_accuracy,
        'type': 'kernel'
    }
    
    # 4. Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    gb_accuracy = accuracy_score(y_test, gb.predict(X_test))
    models['gradient_boosting'] = {
        'model': gb,
        'scaler': None,
        'accuracy': gb_accuracy,
        'type': 'boosting'
    }
    
    # 5. Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt.predict(X_test))
    models['decision_tree'] = {
        'model': dt,
        'scaler': None,
        'accuracy': dt_accuracy,
        'type': 'tree'
    }
    
    # 6. Neural Network (if dataset is large enough)
    if X.shape[0] > 500:
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        mlp.fit(X_train_scaled, y_train)
        mlp_accuracy = accuracy_score(y_test, mlp.predict(X_test_scaled))
        models['neural_network'] = {
            'model': mlp,
            'scaler': scaler,
            'accuracy': mlp_accuracy,
            'type': 'neural'
        }
    
    return models, X_test, y_test

def save_dataset_and_models(dataset_name, dataset_info, models, X_test, y_test, test_dir):
    """Save dataset and trained models to files"""
    
    # Create dataset directory
    dataset_dir = test_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    print(f"  💾 Saving {dataset_name} dataset and models...")
    
    # Save dataset as CSV
    X, y = dataset_info['X'], dataset_info['y']
    
    # Create DataFrame with feature names
    df = pd.DataFrame(X, columns=dataset_info['features'])
    df[dataset_info['target']] = y
    
    csv_path = dataset_dir / f"{dataset_name}_dataset.csv"
    df.to_csv(csv_path, index=False)
    
    # Save test data separately
    test_df = pd.DataFrame(X_test, columns=dataset_info['features'])
    test_df[dataset_info['target']] = y_test
    test_csv_path = dataset_dir / f"{dataset_name}_test_data.csv"
    test_df.to_csv(test_csv_path, index=False)
    
    # Save models
    for model_name, model_info in models.items():
        model_dir = dataset_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save with joblib (recommended for sklearn)
        joblib_path = model_dir / f"{model_name}_model.joblib"
        joblib.dump(model_info['model'], joblib_path)
        
        # Save with pickle (for compatibility testing)
        pickle_path = model_dir / f"{model_name}_model.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(model_info['model'], f)
        
        # Save scaler if exists
        if model_info['scaler'] is not None:
            scaler_path = model_dir / f"{model_name}_scaler.joblib"
            joblib.dump(model_info['scaler'], scaler_path)
        
        # Save model info
        info_data = {
            'model_type': model_info['type'],
            'accuracy': model_info['accuracy'],
            'features': dataset_info['features'],
            'target': dataset_info['target'],
            'dataset_name': dataset_info['name'],
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'has_scaler': model_info['scaler'] is not None
        }
        
        info_df = pd.DataFrame([info_data])
        info_path = model_dir / f"{model_name}_info.csv"
        info_df.to_csv(info_path, index=False)

def create_test_scenarios():
    """Create specific test scenarios for edge cases"""
    test_dir = Path("test_data") / "edge_cases"
    test_dir.mkdir(exist_ok=True)
    
    print("⚠️  Creating edge case test scenarios...")
    
    # 1. Feature mismatch scenario
    print("  📐 Creating feature mismatch scenario...")
    X_train = np.random.rand(500, 10)
    y_train = np.random.randint(0, 2, 500)
    X_test_mismatch = np.random.rand(100, 15)  # Different number of features
    
    # Train model on 10 features
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, test_dir / "feature_mismatch_model.joblib")
    
    # Save test data with different features
    df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(10)])
    df_train['target'] = y_train
    df_train.to_csv(test_dir / "feature_mismatch_train.csv", index=False)
    
    df_test = pd.DataFrame(X_test_mismatch, columns=[f'feature_{i}' for i in range(15)])
    df_test['target'] = np.random.randint(0, 2, 100)
    df_test.to_csv(test_dir / "feature_mismatch_test.csv", index=False)
    
    # 2. Very small dataset
    print("  🔬 Creating small dataset scenario...")
    X_small = np.random.rand(50, 3)
    y_small = np.random.randint(0, 2, 50)
    model_small = LogisticRegression(random_state=42)
    model_small.fit(X_small, y_small)
    
    joblib.dump(model_small, test_dir / "small_dataset_model.joblib")
    df_small = pd.DataFrame(X_small, columns=['feat_a', 'feat_b', 'feat_c'])
    df_small['label'] = y_small
    df_small.to_csv(test_dir / "small_dataset.csv", index=False)
    
    # 3. High dimensional dataset
    print("  📊 Creating high dimensional scenario...")
    X_high_dim = np.random.rand(200, 50)
    y_high_dim = np.random.randint(0, 3, 200)
    model_high_dim = RandomForestClassifier(n_estimators=50, random_state=42)
    model_high_dim.fit(X_high_dim, y_high_dim)
    
    joblib.dump(model_high_dim, test_dir / "high_dimensional_model.joblib")
    df_high_dim = pd.DataFrame(X_high_dim, columns=[f'dim_{i}' for i in range(50)])
    df_high_dim['multiclass_target'] = y_high_dim
    df_high_dim.to_csv(test_dir / "high_dimensional_dataset.csv", index=False)

def generate_summary_report(test_dir):
    """Generate a summary report of all generated test data"""
    
    print("\n📋 Generating summary report...")
    
    summary = {
        'datasets': [],
        'models': [],
        'total_files': 0
    }
    
    # Count all files and gather info
    for item in test_dir.rglob('*'):
        if item.is_file():
            summary['total_files'] += 1
            
            if item.name.endswith('_dataset.csv'):
                df = pd.read_csv(item)
                summary['datasets'].append({
                    'name': item.stem.replace('_dataset', ''),
                    'samples': len(df),
                    'features': len(df.columns) - 1,
                    'path': str(item.relative_to(test_dir))
                })
            
            elif item.name.endswith('_model.joblib'):
                model_name = item.stem.replace('_model', '')
                dataset_name = item.parent.parent.name
                summary['models'].append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'path': str(item.relative_to(test_dir))
                })
    
    # Create summary report
    report_path = test_dir / "TEST_DATA_SUMMARY.md"
    
    with open(report_path, 'w') as f:
        f.write("# AI Shield Test Data Summary\n\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Overview\n")
        f.write(f"- **Total Files**: {summary['total_files']}\n")
        f.write(f"- **Datasets**: {len(summary['datasets'])}\n")
        f.write(f"- **Models**: {len(summary['models'])}\n\n")
        
        f.write("## Datasets\n\n")
        for ds in summary['datasets']:
            f.write(f"### {ds['name']}\n")
            f.write(f"- **Samples**: {ds['samples']}\n")
            f.write(f"- **Features**: {ds['features']}\n")
            f.write(f"- **File**: `{ds['path']}`\n\n")
        
        f.write("## Models by Dataset\n\n")
        current_dataset = None
        for model in sorted(summary['models'], key=lambda x: x['dataset']):
            if model['dataset'] != current_dataset:
                current_dataset = model['dataset']
                f.write(f"### {current_dataset}\n")
            f.write(f"- **{model['model']}**: `{model['path']}`\n")
        
        f.write("\n## Usage Instructions\n\n")
        f.write("1. Use the web interface: `./start_web.sh`\n")
        f.write("2. Navigate to http://localhost:5001/upload\n")
        f.write("3. Upload any model (.joblib or .pkl) and corresponding dataset (.csv)\n")
        f.write("4. Run adversarial analysis\n\n")
        
        f.write("## Test Scenarios\n\n")
        f.write("- **Normal cases**: All dataset folders contain compatible model-data pairs\n")
        f.write("- **Edge cases**: `edge_cases/` folder contains intentional mismatches for robustness testing\n")
        f.write("- **Feature alignment**: Test with different feature counts between model and data\n")
        f.write("- **Various algorithms**: RF, LR, SVM, GB, DT, NN models available\n")
        f.write("- **Different data types**: Binary/multi-class, linear/non-linear, synthetic/realistic\n")
    
    print(f"✅ Summary report saved to: {report_path}")
    return report_path

def main():
    """Main function to generate all test data"""
    
    print("🚀 AI Shield Test Data Generator")
    print("=" * 50)
    
    # Create output directory
    test_dir = create_output_dir()
    
    # Generate datasets
    synthetic_datasets = generate_synthetic_datasets()
    real_world_datasets = generate_real_world_like_datasets()
    
    all_datasets = {**synthetic_datasets, **real_world_datasets}
    
    print(f"\n🤖 Training models for {len(all_datasets)} datasets...")
    
    # Create and save models for each dataset
    for dataset_name, dataset_info in all_datasets.items():
        print(f"\n📊 Processing {dataset_info['name']}...")
        
        models, X_test, y_test = create_models_for_dataset(
            dataset_info['X'], 
            dataset_info['y'], 
            dataset_info['name']
        )
        
        save_dataset_and_models(
            dataset_name, 
            dataset_info, 
            models, 
            X_test, 
            y_test, 
            test_dir
        )
        
        print(f"  ✅ Saved {len(models)} models for {dataset_name}")
    
    # Create edge case scenarios
    create_test_scenarios()
    
    # Generate summary report
    summary_path = generate_summary_report(test_dir)
    
    print("\n" + "=" * 50)
    print("🎉 Test data generation complete!")
    print(f"📁 All files saved in: {test_dir.absolute()}")
    print(f"📋 Summary report: {summary_path}")
    print("\n🌐 To test with web interface:")
    print("   ./start_web.sh")
    print("   Navigate to http://localhost:5001/upload")
    print("\n💡 Try uploading different model-dataset combinations!")

if __name__ == "__main__":
    main()
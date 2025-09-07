#!/usr/bin/env python3
"""
Generate a high-dimensional, vulnerable model for adversarial testing.
This creates models that are intentionally more susceptible to adversarial attacks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_vulnerable_dataset(n_samples=1000, n_features=100, n_classes=2, noise=0.1, random_state=42):
    """Create a high-dimensional dataset with some noise to make models more vulnerable."""
    
    # Create base dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features//5),  # Only some features are informative
        n_redundant=max(0, n_features//10),   # Some redundant features
        n_clusters_per_class=1,
        n_classes=n_classes,
        class_sep=0.8,  # Moderate class separation (not too easy)
        random_state=random_state
    )
    
    # Add noise to make the model more vulnerable
    noise_matrix = np.random.normal(0, noise, X.shape)
    X = X + noise_matrix
    
    return X, y

def create_vulnerable_models():
    """Create several vulnerable models with different architectures."""
    
    print("🎯 Creating high-dimensional vulnerable models...")
    
    # Create datasets of different dimensions
    datasets = {
        'high_dim_50': create_vulnerable_dataset(n_samples=500, n_features=50, noise=0.15),
        'high_dim_100': create_vulnerable_dataset(n_samples=800, n_features=100, noise=0.2),
        'high_dim_200': create_vulnerable_dataset(n_samples=1000, n_features=200, noise=0.25),
        'extreme_dim_500': create_vulnerable_dataset(n_samples=600, n_features=500, noise=0.3),
    }
    
    # Model configurations that tend to be more vulnerable
    models = {
        'vulnerable_mlp': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42
        ),
        'overfit_mlp': MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),  # Larger network, more prone to overfitting
            activation='tanh',
            solver='lbfgs',
            max_iter=100,
            random_state=42
        ),
        'vulnerable_svm': SVC(
            kernel='rbf',
            C=0.1,  # Lower C makes it more vulnerable
            gamma='scale',
            random_state=42
        ),
        'vulnerable_logistic': LogisticRegression(
            C=0.01,  # Lower regularization = more vulnerable
            max_iter=1000,
            random_state=42
        )
    }
    
    # Create models directory
    os.makedirs('vulnerable_models', exist_ok=True)
    
    results = []
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n📊 Processing dataset: {dataset_name} ({X.shape[1]} features)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save test data
        test_df = pd.DataFrame(X_test_scaled, columns=[f'feature_{i}' for i in range(X.shape[1])])
        test_df['target'] = y_test
        test_file = f'vulnerable_models/{dataset_name}_test_data.csv'
        test_df.to_csv(test_file, index=False)
        
        for model_name, model in models.items():
            try:
                print(f"  🔧 Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Calculate training accuracy (for overfitting detection)
                train_acc = model.score(X_train_scaled, y_train)
                test_acc = model.score(X_test_scaled, y_test)
                
                # Save model
                model_file = f'vulnerable_models/{dataset_name}_{model_name}_model.joblib'
                joblib.dump(model, model_file)
                
                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'features': X.shape[1],
                    'samples': X.shape[0],
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'overfitting': train_acc - test_acc,
                    'model_file': model_file,
                    'data_file': test_file
                })
                
                print(f"    ✅ Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}, Overfitting: {train_acc-test_acc:.3f}")
                
            except Exception as e:
                print(f"    ❌ Failed to train {model_name}: {e}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('vulnerable_models/model_summary.csv', index=False)
    
    print(f"\n🎉 Created {len(results)} vulnerable models!")
    print("\n📈 Most vulnerable candidates (high overfitting = more vulnerable):")
    
    # Sort by overfitting (higher overfitting often means more vulnerable)
    top_vulnerable = results_df.nlargest(5, 'overfitting')
    for _, row in top_vulnerable.iterrows():
        print(f"  🔴 {row['dataset']}_{row['model']} - {row['features']} features, overfitting: {row['overfitting']:.3f}")
    
    print(f"\n📁 All models saved in: vulnerable_models/")
    print("🚀 To test: Copy any model.joblib + data.csv to data/ folder and run AI Shield!")
    
    return results_df

if __name__ == "__main__":
    results = create_vulnerable_models()
    print(f"\nTop vulnerable model recommendations:")
    
    # Recommend the most promising candidates
    high_dim = results[results['features'] >= 100]
    if not high_dim.empty:
        best = high_dim.nlargest(1, 'overfitting').iloc[0]
        print(f"\n🎯 RECOMMENDED HIGH-VULNERABILITY MODEL:")
        print(f"   Model: {best['model_file']}")
        print(f"   Data: {best['data_file']}")
        print(f"   Features: {best['features']}")
        print(f"   Expected vulnerability: 25-50%+ due to {best['overfitting']:.3f} overfitting")
        
        print(f"\n💻 To test this model:")
        print(f"   cp {best['model_file']} data/demo_model.joblib")
        print(f"   cp {best['data_file']} data/demo_dataset.csv")
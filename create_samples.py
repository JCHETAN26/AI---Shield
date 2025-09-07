#!/usr/bin/env python3
"""
Sample data and model generator for AI Shield.

This script creates sample datasets and models for testing the AI Shield system.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def create_sample_dataset():
    """Create a sample dataset for adversarial testing."""
    print("Creating sample dataset...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_redundant=0,
        n_informative=8,
        random_state=42,
        n_clusters_per_class=1
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(10)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save dataset
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    df.to_csv(data_dir / 'sample_dataset.csv', index=False)
    
    print(f"Dataset saved to {data_dir / 'sample_dataset.csv'}")
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, feature_names

def create_sample_model(X, y):
    """Create and train a sample model."""
    print("\\nCreating sample model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"Model training accuracy: {train_acc:.3f}")
    print(f"Model test accuracy: {test_acc:.3f}")
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save as pickle
    with open(models_dir / 'sample_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save as joblib (alternative format)
    joblib.dump(model, models_dir / 'sample_model.joblib')
    
    print(f"Model saved to {models_dir / 'sample_model.pkl'} and {models_dir / 'sample_model.joblib'}")
    
    return model

def create_multiple_datasets():
    """Create multiple sample datasets for testing."""
    print("\\nCreating additional sample datasets...")
    
    datasets = {
        'small': {'n_samples': 500, 'n_features': 5},
        'medium': {'n_samples': 1000, 'n_features': 10},
        'large': {'n_samples': 2000, 'n_features': 20}
    }
    
    data_dir = Path('data')
    
    for name, params in datasets.items():
        X, y = make_classification(
            n_samples=params['n_samples'],
            n_features=params['n_features'],
            n_classes=2,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(params['n_features'])]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        filename = data_dir / f'sample_dataset_{name}.csv'
        df.to_csv(filename, index=False)
        print(f"Created {filename} with shape {df.shape}")

def main():
    """Main function to create sample data and models."""
    print("AI Shield - Sample Data and Model Generator")
    print("=" * 50)
    
    # Create main sample dataset
    X, y, feature_names = create_sample_dataset()
    
    # Create main sample model
    model = create_sample_model(X, y)
    
    # Create additional datasets
    create_multiple_datasets()
    
    print("\\n" + "=" * 50)
    print("Sample data and models created successfully!")
    print("\\nNext steps:")
    print("1. Upload files to S3:")
    print("   aws s3 cp data/ s3://your-bucket/data/ --recursive")
    print("   aws s3 cp models/ s3://your-bucket/models/ --recursive")
    print("\\n2. Test AI Shield:")
    print("   python main.py --bucket your-bucket --model models/sample_model.pkl --data data/sample_dataset.csv")

if __name__ == "__main__":
    main()
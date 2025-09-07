"""
Data Processor for AI Shield.

This module handles loading and processing datasets for adversarial analysis.
Supports various data formats and preprocessing operations.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer


class DataProcessor:
    """
    Data processor for loading and preprocessing datasets.
    
    Handles various file formats and preprocessing operations for ML workflows.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.csv', '.json', '.parquet', '.xlsx', '.npy']
        self.scaler = None
        self.label_encoder = None
        self.logger.info("Data Processor initialized")
    
    def load_and_process_data(self, data_path: str, 
                            target_column: Optional[str] = None,
                            test_size: float = 0.2,
                            random_state: int = 42,
                            scale_features: bool = True) -> Dict[str, Any]:
        """
        Load and process data from file.
        
        Args:
            data_path: Path to the data file
            target_column: Name of target column (if None, assumes last column)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scale_features: Whether to scale features
            
        Returns:
            Dictionary containing processed data splits and metadata
        """
        try:
            self.logger.info(f"Loading data from {data_path}")
            
            data_path = Path(data_path)
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load data based on file extension
            extension = data_path.suffix.lower()
            
            if extension == '.csv':
                df = pd.read_csv(data_path)
            elif extension == '.json':
                df = pd.read_json(data_path)
            elif extension == '.parquet':
                df = pd.read_parquet(data_path)
            elif extension == '.xlsx':
                df = pd.read_excel(data_path)
            elif extension == '.npy':
                # Handle numpy arrays
                data = np.load(data_path)
                if data.ndim == 1:
                    df = pd.DataFrame({'feature_0': data})
                else:
                    df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
            
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Process the dataframe
            return self._process_dataframe(df, target_column, test_size, random_state, scale_features)
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame,
                          target_column: Optional[str] = None,
                          test_size: float = 0.2,
                          random_state: int = 42,
                          scale_features: bool = True) -> Dict[str, Any]:
        """
        Process a pandas DataFrame for ML workflow.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
            random_state: Random seed
            scale_features: Whether to scale features
            
        Returns:
            Dictionary containing processed data
        """
        try:
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Separate features and target
            if target_column is None:
                # Assume last column is target
                target_column = df.columns[-1]
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical features
            X = self._encode_categorical_features(X)
            
            # Encode target if necessary
            y_encoded = self._encode_target(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            # Scale features if requested
            if scale_features:
                X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
            else:
                # Convert to numpy arrays if they're DataFrames
                X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
                X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test
            
            # Create feature names
            feature_names = list(X.columns)
            
            # Get data statistics
            data_stats = self._get_data_statistics(df, X, y)
            
            result = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train.values if hasattr(y_train, 'values') else y_train,
                'y_test': y_test.values if hasattr(y_test, 'values') else y_test,
                'feature_names': feature_names,
                'target_column': target_column,
                'num_classes': len(np.unique(y_encoded)),
                'class_names': list(np.unique(y)) if y.dtype == 'object' else list(np.unique(y_encoded)),
                'data_shape': df.shape,
                'statistics': data_stats,
                'preprocessing': {
                    'scaled': scale_features,
                    'scaler_type': type(self.scaler).__name__ if self.scaler else None,
                    'label_encoded': self.label_encoder is not None
                }
            }
            
            self.logger.info(f"Data processed successfully. Features: {len(feature_names)}, Classes: {result['num_classes']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing dataframe: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Check for missing values
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() > 0:
            self.logger.warning(f"Found {missing_counts.sum()} missing values")
            
            # For numerical columns, fill with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
        
        return df
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            self.logger.info(f"Encoding {len(categorical_cols)} categorical features")
            X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
        
        return X
    
    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable if it's categorical."""
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            self.logger.info(f"Target encoded. Classes: {self.label_encoder.classes_}")
            return y_encoded
        else:
            return y.values
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info("Features scaled using StandardScaler")
        return X_train_scaled, X_test_scaled
    
    def _get_data_statistics(self, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            'total_samples': len(df),
            'num_features': len(X.columns),
            'num_categorical_features': len(X.select_dtypes(include=['object']).columns),
            'num_numerical_features': len(X.select_dtypes(include=[np.number]).columns),
            'target_distribution': y.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'feature_statistics': X.describe().to_dict()
        }
        
        return stats
    
    def load_sample_dataset(self, dataset_name: str = 'iris') -> Dict[str, Any]:
        """
        Load a sample dataset for testing purposes.
        
        Args:
            dataset_name: Name of the dataset ('iris', 'wine', 'breast_cancer')
            
        Returns:
            Dictionary containing processed sample data
        """
        try:
            self.logger.info(f"Loading sample dataset: {dataset_name}")
            
            if dataset_name == 'iris':
                data = load_iris()
            elif dataset_name == 'wine':
                data = load_wine()
            elif dataset_name == 'breast_cancer':
                data = load_breast_cancer()
            else:
                raise ValueError(f"Unknown sample dataset: {dataset_name}")
            
            # Create DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            # Process the data
            result = self._process_dataframe(df, target_column='target')
            result['dataset_name'] = dataset_name
            result['dataset_description'] = data.DESCR
            
            self.logger.info(f"Sample dataset '{dataset_name}' loaded successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading sample dataset: {str(e)}")
            raise
    
    def create_synthetic_data(self, n_samples: int = 1000, n_features: int = 10, 
                            n_classes: int = 2, random_state: int = 42) -> Dict[str, Any]:
        """
        Create synthetic data for testing purposes.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Number of features
            n_classes: Number of classes
            random_state: Random seed
            
        Returns:
            Dictionary containing synthetic data
        """
        try:
            from sklearn.datasets import make_classification
            
            self.logger.info(f"Creating synthetic data: {n_samples} samples, {n_features} features, {n_classes} classes")
            
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=n_features,
                random_state=random_state,
                n_clusters_per_class=1
            )
            
            # Create DataFrame
            feature_names = [f"feature_{i}" for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Process the data
            result = self._process_dataframe(df, target_column='target')
            result['dataset_name'] = 'synthetic'
            result['generation_params'] = {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'random_state': random_state
            }
            
            self.logger.info("Synthetic data created successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating synthetic data: {str(e)}")
            raise
    
    def save_processed_data(self, data: Dict[str, Any], output_path: str):
        """
        Save processed data to file.
        
        Args:
            data: Processed data dictionary
            output_path: Path to save the data
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays
            np.savez(
                output_path,
                X_train=data['X_train'],
                X_test=data['X_test'],
                y_train=data['y_train'],
                y_test=data['y_test'],
                feature_names=data['feature_names'],
                class_names=data['class_names']
            )
            
            self.logger.info(f"Processed data saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise
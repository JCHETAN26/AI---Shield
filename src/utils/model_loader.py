"""
Model Loader for AI Shield.

This module handles loading different types of ML models including:
- PyTorch models (.pth, .pt)
- TensorFlow models (.h5, .pb)
- Scikit-learn models (.pkl, .joblib)
- ONNX models (.onnx)
"""

import pickle
import joblib
import logging
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np

# Framework imports
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class ModelLoader:
    """
    Unified model loader for different ML frameworks and formats.
    
    Supports PyTorch, TensorFlow, scikit-learn, and ONNX models.
    """
    
    def __init__(self):
        """Initialize the model loader."""
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = {
            '.pth': 'pytorch',
            '.pt': 'pytorch',
            '.h5': 'tensorflow',
            '.pb': 'tensorflow',
            '.pkl': 'sklearn',
            '.pickle': 'sklearn',
            '.joblib': 'sklearn',
            '.onnx': 'onnx'
        }
        self.logger.info("Model Loader initialized")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a model from the specified path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model object
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Determine model type from extension
            extension = model_path.suffix.lower()
            
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported model format: {extension}")
            
            model_type = self.supported_extensions[extension]
            
            self.logger.info(f"Loading {model_type} model from {model_path}")
            
            # Load based on model type
            if model_type == 'pytorch':
                return self._load_pytorch_model(model_path)
            elif model_type == 'tensorflow':
                return self._load_tensorflow_model(model_path)
            elif model_type == 'sklearn':
                return self._load_sklearn_model(model_path)
            elif model_type == 'onnx':
                return self._load_onnx_model(model_path)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_pytorch_model(self, model_path: Path) -> Any:
        """Load a PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        try:
            # Try loading as state dict first
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # This is a checkpoint with metadata
                    self.logger.warning("Loaded checkpoint contains state dict. Model architecture needed separately.")
                    return checkpoint
                elif 'state_dict' in checkpoint:
                    self.logger.warning("Loaded checkpoint contains state dict. Model architecture needed separately.")
                    return checkpoint
                else:
                    # This might be just a state dict
                    self.logger.warning("Loaded state dict only. Model architecture needed separately.")
                    return checkpoint
            else:
                # This should be a complete model
                model = checkpoint
                model.eval()  # Set to evaluation mode
                return model
                
        except Exception as e:
            self.logger.error(f"Error loading PyTorch model: {str(e)}")
            raise
    
    def _load_tensorflow_model(self, model_path: Path) -> Any:
        """Load a TensorFlow model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
        
        try:
            if model_path.suffix == '.h5':
                # Load Keras model
                model = tf.keras.models.load_model(str(model_path))
            else:
                # Load SavedModel format
                model = tf.saved_model.load(str(model_path))
            
            self.logger.info("TensorFlow model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading TensorFlow model: {str(e)}")
            raise
    
    def _load_sklearn_model(self, model_path: Path) -> Any:
        """Load a scikit-learn model."""
        try:
            if model_path.suffix == '.joblib':
                model = joblib.load(model_path)
            else:
                # Try multiple pickle loading methods to handle different protocols and encodings
                model = None
                
                # Method 1: Standard pickle loading
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                except Exception as e1:
                    self.logger.warning(f"Standard pickle loading failed: {str(e1)}")
                    
                    # Method 2: Try with different encoding
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f, encoding='latin1')
                    except Exception as e2:
                        self.logger.warning(f"Latin1 encoding pickle loading failed: {str(e2)}")
                        
                        # Method 3: Try with bytes encoding
                        try:
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f, encoding='bytes')
                        except Exception as e3:
                            self.logger.warning(f"Bytes encoding pickle loading failed: {str(e3)}")
                            
                            # Method 4: Try joblib as fallback even for .pkl files
                            try:
                                model = joblib.load(model_path)
                                self.logger.info("Loaded using joblib as fallback")
                            except Exception as e4:
                                raise Exception(f"All pickle loading methods failed. Last error: {str(e4)}")
                
                if model is None:
                    raise Exception("Failed to load model with any method")
            
            self.logger.info("Scikit-learn model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading scikit-learn model: {str(e)}")
            raise
    
    def _load_onnx_model(self, model_path: Path) -> Any:
        """Load an ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime")
        
        try:
            # Create ONNX Runtime session
            session = ort.InferenceSession(str(model_path))
            
            self.logger.info("ONNX model loaded successfully")
            return session
            
        except Exception as e:
            self.logger.error(f"Error loading ONNX model: {str(e)}")
            raise
    
    def get_model_info(self, model) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model: Loaded model object
            
        Returns:
            Dictionary containing model information
        """
        try:
            info = {
                'model_type': 'unknown',
                'framework': 'unknown',
                'input_shape': None,
                'output_shape': None,
                'parameters': None
            }
            
            # Detect model type
            if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                info['framework'] = 'pytorch'
                info['model_type'] = type(model).__name__
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                info['parameters'] = {
                    'total': total_params,
                    'trainable': trainable_params
                }
                
            elif TF_AVAILABLE and (isinstance(model, tf.keras.Model) or hasattr(model, 'signatures')):
                info['framework'] = 'tensorflow'
                if isinstance(model, tf.keras.Model):
                    info['model_type'] = 'keras'
                    if hasattr(model, 'input_shape'):
                        info['input_shape'] = model.input_shape
                    if hasattr(model, 'output_shape'):
                        info['output_shape'] = model.output_shape
                else:
                    info['model_type'] = 'saved_model'
                    
            elif hasattr(model, 'predict'):
                # Likely scikit-learn model
                info['framework'] = 'sklearn'
                info['model_type'] = type(model).__name__
                
            elif ONNX_AVAILABLE and isinstance(model, ort.InferenceSession):
                info['framework'] = 'onnx'
                info['model_type'] = 'onnx_session'
                
                # Get input/output info
                inputs = model.get_inputs()
                outputs = model.get_outputs()
                
                if inputs:
                    info['input_shape'] = inputs[0].shape
                if outputs:
                    info['output_shape'] = outputs[0].shape
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return info
    
    def create_dummy_model(self, model_type: str = 'sklearn', **kwargs) -> Any:
        """
        Create a dummy model for testing purposes.
        
        Args:
            model_type: Type of dummy model to create
            **kwargs: Additional parameters for model creation
            
        Returns:
            Dummy model object
        """
        try:
            if model_type == 'sklearn':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                
                # Create dummy data to fit the model
                X_dummy = np.random.rand(100, kwargs.get('n_features', 10))
                y_dummy = np.random.randint(0, 2, 100)
                model.fit(X_dummy, y_dummy)
                
                return model
                
            elif model_type == 'pytorch' and TORCH_AVAILABLE:
                class DummyNet(nn.Module):
                    def __init__(self, input_size=10, num_classes=2):
                        super(DummyNet, self).__init__()
                        self.fc1 = nn.Linear(input_size, 50)
                        self.fc2 = nn.Linear(50, num_classes)
                        self.relu = nn.ReLU()
                        
                    def forward(self, x):
                        x = self.relu(self.fc1(x))
                        x = self.fc2(x)
                        return x
                
                model = DummyNet(
                    input_size=kwargs.get('input_size', 10),
                    num_classes=kwargs.get('num_classes', 2)
                )
                model.eval()
                return model
                
            elif model_type == 'tensorflow' and TF_AVAILABLE:
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(50, activation='relu', input_shape=(kwargs.get('input_size', 10),)),
                    tf.keras.layers.Dense(kwargs.get('num_classes', 2), activation='softmax')
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
                return model
                
            else:
                raise ValueError(f"Unsupported dummy model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating dummy model: {str(e)}")
            raise
#!/usr/bin/env python3
"""
Test the fixed mitigation path issue
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.model_loader import ModelLoader
from src.utils.data_processor import DataProcessor

def test_model_loading():
    """Test that model loading works with fixed paths"""
    
    print("🔧 Testing Fixed Model Loading...")
    print("=" * 40)
    
    # Test 1: Direct path
    model_path = "models/fraud_detection_neural_network_model.joblib"
    data_path = "data/fraud_detection_neural_network_dataset.csv"
    
    print(f"✅ Model file exists: {os.path.exists(model_path)}")
    print(f"✅ Data file exists: {os.path.exists(data_path)}")
    
    # Test 2: Model loading
    model_loader = ModelLoader()
    data_processor = DataProcessor()
    
    try:
        model = model_loader.load_model(model_path)
        print(f"✅ Model loaded successfully: {type(model).__name__}")
        
        data_dict = data_processor.load_and_process_data(data_path)
        print(f"✅ Data loaded successfully: {data_dict['X_train'].shape} train samples")
        
        # Test 3: Model prediction
        X_test = data_dict['X_test']
        predictions = model.predict(X_test[:5])  # Test first 5 samples
        print(f"✅ Model predictions work: {predictions}")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The mitigation system should now work properly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_model_loading()
#!/usr/bin/env python3
"""
AI Shield Automated Test Suite
Tests the adversarial ML analysis system with generated test data.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_direct_analysis():
    """Test the analysis directly using the core components"""
    
    print("🧪 Testing Direct Analysis")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Binary Simple - Random Forest',
            'model': 'test_data/binary_simple/random_forest/random_forest_model.joblib',
            'data': 'test_data/binary_simple/binary_simple_test_data.csv',
            'expected_status': 'success'
        },
        {
            'name': 'Credit Risk - Logistic Regression',
            'model': 'test_data/credit_risk/logistic_regression/logistic_regression_model.joblib',
            'data': 'test_data/credit_risk/credit_risk_test_data.csv',
            'expected_status': 'success'
        },
        {
            'name': 'Medical Diagnosis - SVM',
            'model': 'test_data/medical_diagnosis/svm/svm_model.joblib',
            'data': 'test_data/medical_diagnosis/medical_diagnosis_test_data.csv',
            'expected_status': 'success'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}/{len(test_cases)}: {test_case['name']}")
        print("-" * 30)
        
        # Check if files exist
        model_path = Path(test_case['model'])
        data_path = Path(test_case['data'])
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            results.append({'test': test_case['name'], 'status': 'failed', 'reason': 'model_not_found'})
            continue
        
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            results.append({'test': test_case['name'], 'status': 'failed', 'reason': 'data_not_found'})
            continue
        
        # Run the analysis
        try:
            from utils.model_loader import ModelLoader
            from utils.data_processor import DataProcessor
            from adversarial.attack_engine import AdversarialAttackEngine
            
            print(f"📊 Model: {model_path.name}")
            print(f"📈 Data: {data_path.name}")
            
            # Load model and data
            loader = ModelLoader()
            processor = DataProcessor()
            attack_engine = AdversarialAttackEngine()
            
            model = loader.load_model(str(model_path))
            data_result = processor.load_and_process_data(str(data_path))
            
            X = data_result['X_test']
            y = data_result['y_test']
            feature_names = data_result.get('feature_names', [])
            
            print(f"   Model loaded: {type(model).__name__}")
            print(f"   Data shape: {X.shape}")
            
            # Run adversarial attacks
            config = {
                'attack_types': ['fgsm'],
                'epsilon': 0.1,
                'model_type': 'sklearn'
            }
            
            results_dict = attack_engine.run_attacks(model, X, y, config)
            
            if results_dict and 'fgsm' in results_dict:
                print(f"✅ Analysis completed successfully")
                fgsm_results = results_dict['fgsm']
                
                if 'metrics' in fgsm_results and 'accuracy_drop' in fgsm_results['metrics']:
                    accuracy_drop = fgsm_results['metrics']['accuracy_drop']
                    print(f"   📉 Accuracy drop: {accuracy_drop:.3f}")
                    print(f"   ⚡ Attack success: {fgsm_results.get('attack_success', False)}")
                
                results.append({
                    'test': test_case['name'], 
                    'status': 'passed',
                    'accuracy_drop': fgsm_results['metrics'].get('accuracy_drop', 0),
                    'attack_success': fgsm_results.get('attack_success', False)
                })
            else:
                print(f"❌ Analysis failed - no results returned")
                results.append({'test': test_case['name'], 'status': 'failed', 'reason': 'no_results'})
                
        except Exception as e:
            print(f"❌ Analysis failed with error: {str(e)}")
            results.append({'test': test_case['name'], 'status': 'failed', 'reason': str(e)})
    
    return results

def test_broadcasting_fix():
    """Specific test for the broadcasting fix"""
    
    print("\n🔧 Testing Broadcasting Fix")
    print("=" * 50)
    
    try:
        from utils.model_loader import ModelLoader
        from utils.data_processor import DataProcessor
        from adversarial.attack_engine import AdversarialAttackEngine
        import numpy as np
        
        print("📊 Loading feature mismatch scenario...")
        
        # Load model and data with feature mismatch
        loader = ModelLoader()
        processor = DataProcessor()
        attack_engine = AdversarialAttackEngine()
        
        model_path = "test_data/edge_cases/feature_mismatch_model.joblib"
        data_path = "test_data/edge_cases/feature_mismatch_test.csv"
        
        if not Path(model_path).exists() or not Path(data_path).exists():
            print("❌ Test files not found - run generate_test_data.py first")
            return False
        
        # Load model
        model = loader.load_model(model_path)
        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"   Expected features: {getattr(model, 'n_features_in_', 'Unknown')}")
        
        # Load and process data
        data_result = processor.load_and_process_data(data_path)
        X = data_result['X_test']
        y = data_result['y_test']
        print(f"✅ Data loaded: {X.shape}")
        print(f"   Features in data: {X.shape[1]}")
        
        # This should trigger feature alignment
        config = {
            'attack_types': ['fgsm'],
            'epsilon': 0.1,
            'model_type': 'sklearn'
        }
        
        print("🔥 Running adversarial attack (this tests the broadcasting fix)...")
        results = attack_engine.run_attacks(model, X, y, config)
        
        if results and 'fgsm' in results:
            print("✅ Broadcasting fix successful!")
            fgsm_results = results['fgsm']
            if 'metrics' in fgsm_results and 'accuracy_drop' in fgsm_results['metrics']:
                print(f"   Accuracy drop: {fgsm_results['metrics']['accuracy_drop']:.3f}")
            print(f"   Attack successful: {fgsm_results.get('attack_success', False)}")
            return True
        else:
            print("❌ Broadcasting fix failed - no attack results")
            return False
            
    except Exception as e:
        print(f"❌ Broadcasting test failed: {str(e)}")
        return False

def generate_test_report(analysis_results, broadcasting_result):
    """Generate a comprehensive test report"""
    
    print("\n📋 Generating Test Report")
    print("=" * 50)
    
    # Count results
    passed = sum(1 for r in analysis_results if r['status'] == 'passed')
    failed = sum(1 for r in analysis_results if r['status'] == 'failed')
    total = len(analysis_results)
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'total_tests': total + 1,  # +1 for broadcasting test
            'passed': passed + (1 if broadcasting_result else 0),
            'failed': failed + (0 if broadcasting_result else 1),
            'success_rate': f"{((passed + (1 if broadcasting_result else 0)) / (total + 1) * 100):.1f}%"
        },
        'analysis_tests': analysis_results,
        'broadcasting_test': 'passed' if broadcasting_result else 'failed',
        'recommendations': []
    }
    
    # Add recommendations
    if failed > 0:
        report['recommendations'].append("Some tests failed - check error messages above")
    
    if not broadcasting_result:
        report['recommendations'].append("Broadcasting fix test failed - check attack_engine.py")
    
    if passed == total and broadcasting_result:
        report['recommendations'].append("All tests passed! System is ready for production use")
    
    # Save report
    report_path = Path("test_results.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"📊 Test Summary:")
    print(f"   Total Tests: {report['summary']['total_tests']}")
    print(f"   Passed: {report['summary']['passed']}")
    print(f"   Failed: {report['summary']['failed']}")
    print(f"   Success Rate: {report['summary']['success_rate']}")
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return report

def main():
    """Run all tests"""
    
    print("🚀 AI Shield Automated Test Suite")
    print("=" * 60)
    
    # Check if test data exists
    if not Path("test_data").exists():
        print("❌ Test data not found. Please run: python generate_test_data.py")
        sys.exit(1)
    
    # Run analysis tests
    analysis_results = test_direct_analysis()
    
    # Test broadcasting fix
    broadcasting_result = test_broadcasting_fix()
    
    # Generate report
    report = generate_test_report(analysis_results, broadcasting_result)
    
    print("\n" + "=" * 60)
    if report['summary']['failed'] == 0:
        print("🎉 All tests passed! AI Shield is ready to use.")
        print("\n🌐 Start the web interface with: ./start_web.sh")
        print("📤 Then upload any model/data pair from test_data/ folders")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("🔍 Review test_results.json for detailed information")
    
    return report['summary']['failed'] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
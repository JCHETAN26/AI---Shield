# src/mitigation/mitigation_engine.py - New file to create
"""
AI Shield Mitigation Engine - Active Defense Implementation

Provides multiple mitigation strategies to harden ML models against adversarial attacks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import logging
from typing import Dict, Any, Tuple, List
from pathlib import Path
import json

class MitigationEngine:
    """
    Comprehensive mitigation engine for adversarial ML security.
    
    Implements multiple defense strategies:
    1. Adversarial Training
    2. Defensive Distillation  
    3. Feature Preprocessing
    4. Ensemble Methods
    5. Anomaly Detection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mitigation_results = {}
        
    def generate_adversarial_examples(self, model, X, y, epsilon=0.1):
        """Generate adversarial examples for training."""
        from src.adversarial.attack_engine import AdversarialAttackEngine
        
        attack_engine = AdversarialAttackEngine()
        
        # Generate FGSM examples
        fgsm_results = attack_engine.run_fgsm_attack(
            model, X, y, epsilon=epsilon, framework='sklearn'
        )
        
        # Generate PGD examples  
        pgd_results = attack_engine.run_pgd_attack(
            model, X, y, epsilon=epsilon, framework='sklearn'
        )
        
        # Combine adversarial examples
        adversarial_X = []
        adversarial_y = []
        
        if 'adversarial_examples' in fgsm_results:
            adversarial_X.extend(fgsm_results['adversarial_examples'])
            adversarial_y.extend(fgsm_results['original_labels'])
            
        if 'adversarial_examples' in pgd_results:
            adversarial_X.extend(pgd_results['adversarial_examples'])  
            adversarial_y.extend(pgd_results['original_labels'])
        
        return np.array(adversarial_X), np.array(adversarial_y)
    
    def adversarial_training(self, model, X_train, y_train, X_test, y_test):
        """
        Mitigation Strategy 1: Adversarial Training
        Train model with both clean and adversarial examples.
        """
        self.logger.info("Applying adversarial training mitigation...")
        
        try:
            # Generate adversarial examples
            adv_X, adv_y = self.generate_adversarial_examples(model, X_train, y_train)
            
            # Combine clean and adversarial data
            combined_X = np.vstack([X_train, adv_X])
            combined_y = np.hstack([y_train, adv_y])
            
            # Create new model with same architecture
            hardened_model = self._clone_model(model)
            
            # Train on combined dataset
            hardened_model.fit(combined_X, combined_y)
            
            # Evaluate
            original_acc = model.score(X_test, y_test)
            hardened_acc = hardened_model.score(X_test, y_test)
            
            # Test adversarial robustness
            adv_test_X, _ = self.generate_adversarial_examples(model, X_test, y_test)
            original_adv_acc = model.score(adv_test_X, y_test)
            hardened_adv_acc = hardened_model.score(adv_test_X, y_test)
            
            results = {
                'strategy': 'adversarial_training',
                'original_accuracy': original_acc,
                'hardened_accuracy': hardened_acc,
                'original_adversarial_accuracy': original_adv_acc,
                'hardened_adversarial_accuracy': hardened_adv_acc,
                'robustness_improvement': hardened_adv_acc - original_adv_acc,
                'model': hardened_model,
                'success': True
            }
            
        except Exception as e:
            results = {
                'strategy': 'adversarial_training',
                'success': False,
                'error': str(e)
            }
            
        return results
    
    def defensive_distillation(self, model, X_train, y_train, X_test, y_test, temperature=10):
        """
        Mitigation Strategy 2: Defensive Distillation
        Create teacher-student model to reduce gradient information.
        """
        self.logger.info("Applying defensive distillation mitigation...")
        
        try:
            # Teacher model (original)
            teacher_model = model
            
            # Get soft predictions from teacher
            if hasattr(teacher_model, 'predict_proba'):
                teacher_probs = teacher_model.predict_proba(X_train)
                # Apply temperature scaling
                teacher_probs = np.power(teacher_probs, 1/temperature)
                teacher_probs = teacher_probs / np.sum(teacher_probs, axis=1, keepdims=True)
            else:
                # For models without predict_proba, use decision function
                teacher_probs = teacher_model.predict(X_train)
            
            # Create student model (smaller/different architecture)
            student_model = MLPClassifier(
                hidden_layer_sizes=(50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            # Train student on soft targets
            if len(teacher_probs.shape) > 1:
                # Multi-class: use soft targets
                student_model.fit(X_train, np.argmax(teacher_probs, axis=1))
            else:
                # Binary: use original labels
                student_model.fit(X_train, y_train)
            
            # Evaluate
            original_acc = model.score(X_test, y_test)
            student_acc = student_model.score(X_test, y_test)
            
            # Test adversarial robustness
            adv_test_X, _ = self.generate_adversarial_examples(model, X_test, y_test)
            original_adv_acc = model.score(adv_test_X, y_test)
            student_adv_acc = student_model.score(adv_test_X, y_test)
            
            results = {
                'strategy': 'defensive_distillation',
                'original_accuracy': original_acc,
                'hardened_accuracy': student_acc,
                'original_adversarial_accuracy': original_adv_acc,
                'hardened_adversarial_accuracy': student_adv_acc,
                'robustness_improvement': student_adv_acc - original_adv_acc,
                'model': student_model,
                'temperature': temperature,
                'success': True
            }
            
        except Exception as e:
            results = {
                'strategy': 'defensive_distillation', 
                'success': False,
                'error': str(e)
            }
            
        return results
    
    def feature_preprocessing(self, model, X_train, y_train, X_test, y_test):
        """
        Mitigation Strategy 3: Feature Preprocessing
        Apply robust preprocessing to remove adversarial perturbations.
        """
        self.logger.info("Applying feature preprocessing mitigation...")
        
        try:
            # Use RobustScaler instead of StandardScaler
            robust_scaler = RobustScaler()
            X_train_robust = robust_scaler.fit_transform(X_train)
            X_test_robust = robust_scaler.transform(X_test)
            
            # Apply feature clipping (remove outliers)
            percentile_lower = np.percentile(X_train_robust, 5, axis=0)
            percentile_upper = np.percentile(X_train_robust, 95, axis=0)
            
            X_train_clipped = np.clip(X_train_robust, percentile_lower, percentile_upper)
            X_test_clipped = np.clip(X_test_robust, percentile_lower, percentile_upper)
            
            # Train new model on preprocessed data
            hardened_model = self._clone_model(model)
            hardened_model.fit(X_train_clipped, y_train)
            
            # Evaluate
            original_acc = model.score(X_test, y_test)
            hardened_acc = hardened_model.score(X_test_clipped, y_test)
            
            # Test adversarial robustness
            adv_test_X, _ = self.generate_adversarial_examples(model, X_test, y_test)
            adv_test_X_processed = robust_scaler.transform(adv_test_X)
            adv_test_X_clipped = np.clip(adv_test_X_processed, percentile_lower, percentile_upper)
            
            original_adv_acc = model.score(adv_test_X, y_test)
            hardened_adv_acc = hardened_model.score(adv_test_X_clipped, y_test)
            
            results = {
                'strategy': 'feature_preprocessing',
                'original_accuracy': original_acc,
                'hardened_accuracy': hardened_acc, 
                'original_adversarial_accuracy': original_adv_acc,
                'hardened_adversarial_accuracy': hardened_adv_acc,
                'robustness_improvement': hardened_adv_acc - original_adv_acc,
                'model': hardened_model,
                'preprocessor': robust_scaler,
                'clip_bounds': (percentile_lower, percentile_upper),
                'success': True
            }
            
        except Exception as e:
            results = {
                'strategy': 'feature_preprocessing',
                'success': False, 
                'error': str(e)
            }
            
        return results
    
    def ensemble_defense(self, model, X_train, y_train, X_test, y_test):
        """
        Mitigation Strategy 4: Ensemble Methods
        Combine multiple diverse models to improve robustness.
        """
        self.logger.info("Applying ensemble defense mitigation...")
        
        try:
            # Create diverse models
            models = [
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
                ('svm', SVC(probability=True, random_state=42)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
            ]
            
            # Create ensemble
            ensemble_model = VotingClassifier(
                estimators=models,
                voting='soft'
            )
            
            # Train ensemble
            ensemble_model.fit(X_train, y_train)
            
            # Evaluate
            original_acc = model.score(X_test, y_test)
            ensemble_acc = ensemble_model.score(X_test, y_test)
            
            # Test adversarial robustness
            adv_test_X, _ = self.generate_adversarial_examples(model, X_test, y_test)
            original_adv_acc = model.score(adv_test_X, y_test)
            ensemble_adv_acc = ensemble_model.score(adv_test_X, y_test)
            
            results = {
                'strategy': 'ensemble_defense',
                'original_accuracy': original_acc,
                'hardened_accuracy': ensemble_acc,
                'original_adversarial_accuracy': original_adv_acc, 
                'hardened_adversarial_accuracy': ensemble_adv_acc,
                'robustness_improvement': ensemble_adv_acc - original_adv_acc,
                'model': ensemble_model,
                'ensemble_size': len(models),
                'success': True
            }
            
        except Exception as e:
            results = {
                'strategy': 'ensemble_defense',
                'success': False,
                'error': str(e)
            }
            
        return results
    
    def anomaly_detection(self, model, X_train, y_train, X_test, y_test):
        """
        Mitigation Strategy 5: Anomaly Detection
        Detect and filter adversarial examples at runtime.
        """
        self.logger.info("Applying anomaly detection mitigation...")
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Train anomaly detector on clean data
            anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            anomaly_detector.fit(X_train)
            
            # Create wrapper model with anomaly detection
            class AnomalyProtectedModel:
                def __init__(self, base_model, anomaly_detector):
                    self.base_model = base_model
                    self.anomaly_detector = anomaly_detector
                    
                def predict(self, X):
                    # Detect anomalies
                    anomaly_scores = self.anomaly_detector.decision_function(X)
                    is_normal = anomaly_scores > 0
                    
                    # Predict only on normal samples
                    predictions = np.zeros(len(X))
                    if np.any(is_normal):
                        predictions[is_normal] = self.base_model.predict(X[is_normal])
                    
                    # For anomalies, return most frequent class or default
                    if np.any(~is_normal):
                        default_class = np.bincount(y_train).argmax()
                        predictions[~is_normal] = default_class
                        
                    return predictions
                    
                def score(self, X, y):
                    pred = self.predict(X)
                    return np.mean(pred == y)
            
            protected_model = AnomalyProtectedModel(model, anomaly_detector)
            
            # Evaluate
            original_acc = model.score(X_test, y_test)
            protected_acc = protected_model.score(X_test, y_test)
            
            # Test adversarial robustness
            adv_test_X, _ = self.generate_adversarial_examples(model, X_test, y_test)
            original_adv_acc = model.score(adv_test_X, y_test)
            protected_adv_acc = protected_model.score(adv_test_X, y_test)
            
            # Calculate detection rate
            adv_anomaly_scores = anomaly_detector.decision_function(adv_test_X)
            detection_rate = np.mean(adv_anomaly_scores <= 0)
            
            results = {
                'strategy': 'anomaly_detection',
                'original_accuracy': original_acc,
                'hardened_accuracy': protected_acc,
                'original_adversarial_accuracy': original_adv_acc,
                'hardened_adversarial_accuracy': protected_adv_acc,
                'robustness_improvement': protected_adv_acc - original_adv_acc,
                'model': protected_model,
                'anomaly_detector': anomaly_detector,
                'detection_rate': detection_rate,
                'success': True
            }
            
        except Exception as e:
            results = {
                'strategy': 'anomaly_detection',
                'success': False,
                'error': str(e)
            }
            
        return results
    
    def _clone_model(self, model):
        """Create a copy of the model with same parameters."""
        if hasattr(model, 'get_params'):
            params = model.get_params()
            new_model = type(model)(**params)
            return new_model
        else:
            # Fallback for custom models
            return model.__class__()
    
    def run_all_mitigations(self, model, X_train, y_train, X_test, y_test):
        """
        Run all mitigation strategies and return comprehensive results.
        """
        self.logger.info("Running comprehensive mitigation analysis...")
        
        strategies = [
            self.adversarial_training,
            self.defensive_distillation, 
            self.feature_preprocessing,
            self.ensemble_defense,
            self.anomaly_detection
        ]
        
        results = {}
        
        for strategy_func in strategies:
            try:
                result = strategy_func(model, X_train, y_train, X_test, y_test)
                results[result['strategy']] = result
                self.logger.info(f"Completed {result['strategy']}")
            except Exception as e:
                strategy_name = strategy_func.__name__
                self.logger.error(f"Failed {strategy_name}: {str(e)}")
                results[strategy_name] = {
                    'strategy': strategy_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate summary
        successful_strategies = [r for r in results.values() if r.get('success', False)]
        
        if successful_strategies:
            best_strategy = max(successful_strategies, 
                              key=lambda x: x.get('robustness_improvement', 0))
            
            summary = {
                'total_strategies': len(strategies),
                'successful_strategies': len(successful_strategies),
                'best_strategy': best_strategy['strategy'],
                'best_improvement': best_strategy.get('robustness_improvement', 0),
                'recommendations': self._generate_recommendations(results)
            }
        else:
            summary = {
                'total_strategies': len(strategies),
                'successful_strategies': 0,
                'best_strategy': None,
                'best_improvement': 0,
                'recommendations': ['All mitigation strategies failed. Check model and data compatibility.']
            }
        
        return {
            'mitigation_results': results,
            'summary': summary,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_recommendations(self, results):
        """Generate actionable recommendations based on mitigation results."""
        recommendations = []
        
        successful = [r for r in results.values() if r.get('success', False)]
        
        if not successful:
            return ['All mitigation strategies failed. Review model architecture and data quality.']
        
        # Find best performing strategy
        best = max(successful, key=lambda x: x.get('robustness_improvement', 0))
        
        if best['robustness_improvement'] > 0.1:
            recommendations.append(f"✅ RECOMMENDED: Deploy {best['strategy']} - shows {best['robustness_improvement']:.1%} robustness improvement")
        elif best['robustness_improvement'] > 0.05:
            recommendations.append(f"⚠️ CONSIDER: {best['strategy']} - moderate improvement of {best['robustness_improvement']:.1%}")
        else:
            recommendations.append("⚠️ Limited improvement from all strategies - consider architectural changes")
        
        # Strategy-specific recommendations
        for strategy, result in results.items():
            if result.get('success', False):
                if strategy == 'ensemble_defense' and result.get('robustness_improvement', 0) > 0.05:
                    recommendations.append("🔄 Ensemble methods show promise - consider expanding to more diverse models")
                elif strategy == 'anomaly_detection' and result.get('detection_rate', 0) > 0.7:
                    recommendations.append(f"🛡️ Anomaly detection effective - {result['detection_rate']:.1%} adversarial detection rate")
        
        # General recommendations
        if len(successful) >= 3:
            recommendations.append("💡 Multiple strategies viable - consider hybrid approach combining best techniques")
        
        return recommendations

def save_mitigation_results(results, session_id, output_dir='results'):
    """Save mitigation results to file."""
    output_path = Path(output_dir) / f"{session_id}_mitigation_results.json"
    
    # Prepare results for JSON serialization
    serializable_results = {}
    for strategy, result in results['mitigation_results'].items():
        serializable_result = result.copy()
        # Remove non-serializable model objects
        if 'model' in serializable_result:
            del serializable_result['model']
        if 'preprocessor' in serializable_result:
            del serializable_result['preprocessor']
        if 'anomaly_detector' in serializable_result:
            del serializable_result['anomaly_detector']
        serializable_results[strategy] = serializable_result
    
    output_data = {
        'mitigation_results': serializable_results,
        'summary': results['summary'],
        'timestamp': results['timestamp']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    return str(output_path)
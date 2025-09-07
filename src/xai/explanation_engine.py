"""
XAI Explanation Engine for AI Shield.

This module provides explainable AI capabilities using SHAP and LIME
to understand model vulnerabilities and feature importance.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
import shap
try:
    # LIME imports
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    from lime.lime_image import LimeImageExplainer
except ImportError:
    logging.warning("LIME not available. Install with: pip install lime")


class XAIExplanationEngine:
    """
    Engine for generating explainable AI insights using SHAP and LIME.
    
    Provides explanations for model predictions and adversarial vulnerabilities.
    """
    
    def __init__(self):
        """Initialize the XAI explanation engine."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("XAI Explanation Engine initialized")
    
    def generate_shap_explanations(self, model, X_test: np.ndarray, 
                                 attack_results: Dict[str, Any],
                                 feature_names: Optional[List[str]] = None,
                                 max_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model predictions and adversarial examples.
        
        Args:
            model: The ML model to explain
            X_test: Test input data
            attack_results: Results from adversarial attacks
            feature_names: Names of input features
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP explanations
        """
        try:
            self.logger.info("Generating SHAP explanations")
            
            # Limit samples for performance
            X_explain = X_test[:max_samples]
            
            # Initialize SHAP explainer
            explainer = None
            
            # Try different SHAP explainers based on model type
            try:
                # Try TreeExplainer first (for tree-based models)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_explain)
                explainer_type = "TreeExplainer"
            except Exception as tree_error:
                self.logger.info(f"TreeExplainer failed: {tree_error}. Trying KernelExplainer...")
                try:
                    # Try KernelExplainer (model-agnostic)
                    background = shap.sample(X_test, min(50, len(X_test)))
                    explainer = shap.KernelExplainer(model.predict, background)
                    shap_values = explainer.shap_values(X_explain[:10])  # Limit for performance
                    explainer_type = "KernelExplainer"
                except Exception as kernel_error:
                    self.logger.info(f"KernelExplainer failed: {kernel_error}. Using permutation-based approach...")
                    # Fallback to permutation-based importance
                    return self._calculate_permutation_importance(model, X_explain, feature_names)
            
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Multi-class case - take the first class for simplicity
                shap_values_array = np.array(shap_values[0])
            else:
                shap_values_array = shap_values
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(shap_values_array), axis=0)
            
            # Get top important features
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(feature_importance)],
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Analyze adversarial examples if available
            adversarial_analysis = {}
            for attack_type, attack_data in attack_results.items():
                if 'adversarial_examples' in attack_data:
                    adv_examples = np.array(attack_data['adversarial_examples'][:max_samples])
                    if len(adv_examples) > 0:
                        try:
                            adv_shap_values = explainer.shap_values(adv_examples)
                            if isinstance(adv_shap_values, list):
                                adv_shap_values = adv_shap_values[0]
                            
                            # Compare feature importance between original and adversarial
                            adv_importance = np.mean(np.abs(adv_shap_values), axis=0)
                            importance_diff = adv_importance - feature_importance[:len(adv_importance)]
                            
                            adversarial_analysis[attack_type] = {
                                'importance_shift': importance_diff.tolist(),
                                'most_affected_features': [
                                    feature_names[i] for i in np.argsort(np.abs(importance_diff))[-5:][::-1]
                                ]
                            }
                        except Exception as e:
                            self.logger.warning(f"Could not analyze adversarial examples for {attack_type}: {str(e)}")
            
            results = {
                'explainer_type': explainer_type,
                'feature_importance': importance_df.to_dict('records'),
                'important_features': importance_df['feature'].head(10).tolist(),
                'shap_values_summary': {
                    'mean_abs_shap': float(np.mean(np.abs(shap_values_array))),
                    'max_abs_shap': float(np.max(np.abs(shap_values_array))),
                    'feature_count': len(feature_importance)
                },
                'adversarial_analysis': adversarial_analysis,
                'top_positive_features': importance_df[importance_df['importance'] > 0]['feature'].head(5).tolist(),
                'top_negative_features': importance_df[importance_df['importance'] < 0]['feature'].head(5).tolist()
            }
            
            self.logger.info("SHAP explanations generated successfully")
            return results
        
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {str(e)}")
            # Return basic results even if SHAP fails
            return {
                'explainer_type': 'failed',
                'error': str(e),
                'feature_importance': [],
                'important_features': [],
                'shap_values_summary': {},
                'adversarial_analysis': {}
            }
    
    def _calculate_permutation_importance(self, model, X_test: np.ndarray, 
                                        feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate feature importance using permutation-based approach.
        
        Args:
            model: The ML model
            X_test: Test data
            feature_names: Feature names
            
        Returns:
            Dictionary with permutation importance results
        """
        try:
            self.logger.info("Calculating permutation-based feature importance")
            
            # Get baseline accuracy
            baseline_predictions = model.predict(X_test)
            baseline_score = np.mean(model.predict(X_test) == baseline_predictions)  # Dummy score for consistency
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            importance_scores = []
            
            # Calculate importance for each feature
            for i in range(X_test.shape[1]):
                X_permuted = X_test.copy()
                # Permute the feature
                np.random.shuffle(X_permuted[:, i])
                
                # Calculate new score
                permuted_predictions = model.predict(X_permuted)
                permuted_score = np.mean(permuted_predictions == baseline_predictions)
                
                # Importance is the drop in accuracy
                importance = abs(baseline_score - permuted_score)
                importance_scores.append(importance)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance_scores)],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            return {
                'explainer_type': 'PermutationImportance',
                'feature_importance': importance_df.to_dict('records'),
                'important_features': importance_df['feature'].head(10).tolist(),
                'shap_values_summary': {
                    'mean_abs_shap': float(np.mean(importance_scores)),
                    'max_abs_shap': float(np.max(importance_scores)),
                    'feature_count': len(importance_scores)
                },
                'adversarial_analysis': {},
                'note': 'Using permutation-based importance as fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Error in permutation importance: {str(e)}")
            return {
                'explainer_type': 'failed',
                'error': str(e),
                'feature_importance': [],
                'important_features': [],
                'shap_values_summary': {},
                'adversarial_analysis': {}
            }
    
    def generate_lime_explanations(self, model, X_test: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 attack_results: Optional[Dict[str, Any]] = None,
                                 max_samples: int = 50) -> Dict[str, Any]:
        """
        Generate LIME explanations for model predictions.
        
        Args:
            model: The ML model to explain
            X_test: Test input data
            feature_names: Names of input features
            attack_results: Results from adversarial attacks
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing LIME explanations
        """
        try:
            self.logger.info("Generating LIME explanations")
            
            # Check if LIME is available
            if 'lime_tabular' not in globals():
                return {
                    'error': 'LIME not available. Install with: pip install lime',
                    'explanations': [],
                    'feature_importance': []
                }
            
            # Prepare data
            X_explain = X_test[:max_samples]
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_explain.shape[1])]
            
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_test,
                feature_names=feature_names,
                class_names=['class_0', 'class_1'] if hasattr(model, 'predict_proba') else None,
                mode='classification',
                discretize_continuous=True
            )
            
            explanations = []
            feature_importance_sum = np.zeros(len(feature_names))
            
            # Generate explanations for sample instances
            for i in range(min(10, len(X_explain))):  # Explain first 10 instances
                try:
                    # Get prediction function
                    if hasattr(model, 'predict_proba'):
                        predict_fn = model.predict_proba
                    else:
                        predict_fn = lambda x: np.column_stack([1 - model.predict(x), model.predict(x)])
                    
                    # Generate explanation
                    exp = explainer.explain_instance(
                        X_explain[i],
                        predict_fn,
                        num_features=min(10, len(feature_names))
                    )
                    
                    # Extract feature importance from explanation
                    exp_list = exp.as_list()
                    instance_importance = {}
                    
                    for feature_desc, importance in exp_list:
                        # Parse feature name from description
                        feature_idx = None
                        for j, fname in enumerate(feature_names):
                            if fname in feature_desc:
                                feature_idx = j
                                break
                        
                        if feature_idx is not None:
                            instance_importance[feature_names[feature_idx]] = importance
                            feature_importance_sum[feature_idx] += abs(importance)
                    
                    explanations.append({
                        'instance_id': i,
                        'prediction': float(model.predict([X_explain[i]])[0]) if hasattr(model, 'predict') else None,
                        'feature_importance': instance_importance,
                        'explanation_text': str(exp_list)
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Could not explain instance {i}: {str(e)}")
            
            # Calculate overall feature importance
            feature_importance_avg = feature_importance_sum / max(len(explanations), 1)
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance_avg
            }).sort_values('importance', ascending=False)
            
            # Analyze adversarial examples if available
            adversarial_analysis = {}
            if attack_results:
                for attack_type, attack_data in attack_results.items():
                    if 'adversarial_examples' in attack_data:
                        adv_examples = np.array(attack_data['adversarial_examples'][:5])  # Analyze first 5
                        if len(adv_examples) > 0:
                            try:
                                adv_explanations = []
                                for j, adv_example in enumerate(adv_examples):
                                    exp = explainer.explain_instance(
                                        adv_example,
                                        predict_fn,
                                        num_features=5
                                    )
                                    adv_explanations.append({
                                        'instance_id': j,
                                        'explanation': exp.as_list()
                                    })
                                
                                adversarial_analysis[attack_type] = {
                                    'num_explained': len(adv_explanations),
                                    'sample_explanations': adv_explanations
                                }
                            except Exception as e:
                                self.logger.warning(f"Could not explain adversarial examples for {attack_type}: {str(e)}")
            
            results = {
                'num_explanations': len(explanations),
                'explanations': explanations,
                'feature_importance': importance_df.to_dict('records'),
                'important_features': importance_df['feature'].head(10).tolist(),
                'adversarial_analysis': adversarial_analysis,
                'summary': {
                    'total_features': len(feature_names),
                    'explained_instances': len(explanations),
                    'avg_importance': float(np.mean(feature_importance_avg))
                }
            }
            
            self.logger.info("LIME explanations generated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanations: {str(e)}")
            return {
                'error': str(e),
                'explanations': [],
                'feature_importance': [],
                'important_features': []
            }
    
    def compare_explanations(self, shap_results: Dict[str, Any], 
                           lime_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare SHAP and LIME explanations to find consensus and conflicts.
        
        Args:
            shap_results: Results from SHAP analysis
            lime_results: Results from LIME analysis
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            comparison = {
                'consensus_features': [],
                'conflicting_features': [],
                'shap_unique': [],
                'lime_unique': [],
                'correlation_score': 0.0
            }
            
            # Get top features from each method
            shap_features = set(shap_results.get('important_features', [])[:10])
            lime_features = set(lime_results.get('important_features', [])[:10])
            
            # Find consensus and conflicts
            comparison['consensus_features'] = list(shap_features.intersection(lime_features))
            comparison['shap_unique'] = list(shap_features - lime_features)
            comparison['lime_unique'] = list(lime_features - shap_features)
            
            # Calculate correlation if both have feature importance
            shap_importance = {item['feature']: item['importance'] 
                             for item in shap_results.get('feature_importance', [])}
            lime_importance = {item['feature']: item['importance'] 
                             for item in lime_results.get('feature_importance', [])}
            
            common_features = set(shap_importance.keys()).intersection(set(lime_importance.keys()))
            
            if len(common_features) > 1:
                shap_values = [shap_importance[f] for f in common_features]
                lime_values = [lime_importance[f] for f in common_features]
                
                correlation = np.corrcoef(shap_values, lime_values)[0, 1]
                comparison['correlation_score'] = float(correlation) if not np.isnan(correlation) else 0.0
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing explanations: {str(e)}")
            return {
                'error': str(e),
                'consensus_features': [],
                'conflicting_features': [],
                'correlation_score': 0.0
            }
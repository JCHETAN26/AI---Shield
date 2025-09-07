"""
AI Shield - Main execution script for adversarial machine learning security analysis.

This script orchestrates the entire workflow:
1. Downloads models and data from S3
2. Runs adversarial attacks (FGSM, PGD)
3. Generates XAI explanations (SHAP, LIME)
4. Produces structured analysis results
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.aws.s3_manager import S3Manager
from src.adversarial.attack_engine import AdversarialAttackEngine
from src.xai.explanation_engine import XAIExplanationEngine
from src.utils.model_loader import ModelLoader
from src.utils.data_processor import DataProcessor
from src.utils.logger import setup_logger


class AIShieldEngine:
    """
    Main engine for AI Shield adversarial security analysis.
    
    This class coordinates the entire analysis workflow including:
    - Model and data retrieval from S3
    - Adversarial attack execution
    - XAI explanation generation
    - Results aggregation and reporting
    """
    
    def __init__(self, bucket_name: str, aws_region: str = "us-east-1"):
        """
        Initialize the AI Shield engine.
        
        Args:
            bucket_name: Name of the S3 bucket containing models and data
            aws_region: AWS region for S3 operations
        """
        self.bucket_name = bucket_name
        self.aws_region = aws_region
        self.logger = setup_logger("ai_shield")
        
        # Initialize components
        self.s3_manager = S3Manager(bucket_name, aws_region)
        self.model_loader = ModelLoader()
        self.data_processor = DataProcessor()
        self.attack_engine = AdversarialAttackEngine()
        self.xai_engine = XAIExplanationEngine()
        
        self.logger.info("AI Shield Engine initialized successfully")
    
    def download_assets(self, model_key: str, data_key: str) -> tuple:
        """
        Download model and dataset from S3 or use local files.
        
        Args:
            model_key: S3 key for the model file or local path
            data_key: S3 key for the dataset file or local path
            
        Returns:
            Tuple of (model_path, data_path) - local file paths
        """
        self.logger.info(f"Downloading assets from S3 bucket: {self.bucket_name}")
        
        try:
            # Download model
            model_path = self.s3_manager.download_file(
                model_key, 
                f"models/{Path(model_key).name}"
            )
            
            # Download data
            data_path = self.s3_manager.download_file(
                data_key,
                f"data/{Path(data_key).name}"
            )
            
            self.logger.info("Assets downloaded successfully")
            return model_path, data_path
            
        except Exception as e:
            self.logger.warning(f"S3 download failed: {e}. Trying local files...")
            
            # Fallback to local files
            model_path = Path(model_key)
            data_path = Path(data_key)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
                
            self.logger.info("Using local files")
            return str(model_path), str(data_path)
    
    def run_adversarial_attacks(self, model, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute adversarial attacks on the model.
        
        Args:
            model: Loaded ML model
            data: Processed dataset dictionary
            
        Returns:
            Dictionary containing attack results
        """
        self.logger.info("Starting adversarial attacks")
        
        attack_results = {}
        
        # Run FGSM attack
        self.logger.info("Executing FGSM attack")
        fgsm_results = self.attack_engine.run_fgsm_attack(
            model, 
            data['X_test'], 
            data['y_test'],
            framework='sklearn'
        )
        attack_results['fgsm'] = fgsm_results
        
        # Run PGD attack
        self.logger.info("Executing PGD attack")
        pgd_results = self.attack_engine.run_pgd_attack(
            model,
            data['X_test'],
            data['y_test'],
            framework='sklearn'
        )
        attack_results['pgd'] = pgd_results
        
        self.logger.info("Adversarial attacks completed")
        return attack_results
    
    def generate_xai_insights(self, model, data: Dict[str, Any], attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate XAI explanations for the model vulnerabilities.
        
        Args:
            model: Loaded ML model
            data: Processed dataset dictionary
            attack_results: Results from adversarial attacks
            
        Returns:
            Dictionary containing XAI insights
        """
        self.logger.info("Generating XAI explanations")
        
        xai_results = {}
        
        # Generate SHAP explanations
        self.logger.info("Computing SHAP explanations")
        shap_insights = self.xai_engine.generate_shap_explanations(
            model,
            data['X_test'],
            attack_results
        )
        xai_results['shap'] = shap_insights
        
        # Generate LIME explanations
        self.logger.info("Computing LIME explanations")
        lime_insights = self.xai_engine.generate_lime_explanations(
            model,
            data['X_test'],
            data['feature_names'] if 'feature_names' in data else None,
            attack_results
        )
        xai_results['lime'] = lime_insights
        
        self.logger.info("XAI explanations completed")
        return xai_results
    
    def run_security_analysis(self, model_key: str, data_key: str) -> Dict[str, Any]:
        """
        Run the complete security analysis workflow.
        
        Args:
            model_key: S3 key for the model file
            data_key: S3 key for the dataset file
            
        Returns:
            Complete analysis results dictionary
        """
        try:
            self.logger.info("Starting AI Shield security analysis")
            start_time = datetime.now()
            
            # Step 1: Download assets from S3
            model_path, data_path = self.download_assets(model_key, data_key)
            
            # Step 2: Load model and process data
            self.logger.info("Loading model and processing data")
            model = self.model_loader.load_model(model_path)
            data = self.data_processor.load_and_process_data(data_path)
            
            # Step 3: Run adversarial attacks
            attack_results = self.run_adversarial_attacks(model, data)
            
            # Step 4: Generate XAI insights
            xai_results = self.generate_xai_insights(model, data, attack_results)
            
            # Step 5: Compile final results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            final_results = {
                'metadata': {
                    'timestamp': end_time.isoformat(),
                    'execution_time_seconds': execution_time,
                    'model_key': model_key,
                    'data_key': data_key,
                    'bucket_name': self.bucket_name
                },
                'dataset_info': {
                    'num_samples': len(data['X_test']),
                    'num_features': data['X_test'].shape[1] if hasattr(data['X_test'], 'shape') else None,
                    'feature_names': data.get('feature_names', [])
                },
                'adversarial_attacks': attack_results,
                'xai_explanations': xai_results,
                'vulnerability_summary': self._generate_vulnerability_summary(attack_results, xai_results)
            }
            
            self.logger.info(f"Security analysis completed in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error during security analysis: {str(e)}")
            raise
    
    def _generate_vulnerability_summary(self, attack_results: Dict[str, Any], xai_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a high-level vulnerability summary.
        
        Args:
            attack_results: Results from adversarial attacks
            xai_results: Results from XAI analysis
            
        Returns:
            Vulnerability summary dictionary
        """
        summary = {
            'overall_vulnerability_score': 0.0,
            'attack_success_rates': {},
            'critical_features': [],
            'recommendations': []
        }
        
        # Calculate attack success rates
        total_success_rate = 0
        attack_count = 0
        
        for attack_type, results in attack_results.items():
            if 'success_rate' in results:
                success_rate = results['success_rate']
                summary['attack_success_rates'][attack_type] = success_rate
                total_success_rate += success_rate
                attack_count += 1
        
        if attack_count > 0:
            summary['overall_vulnerability_score'] = total_success_rate / attack_count
        
        # Extract critical features from XAI results
        if 'shap' in xai_results and 'important_features' in xai_results['shap']:
            summary['critical_features'].extend(xai_results['shap']['important_features'][:5])
        
        # Generate recommendations
        if summary['overall_vulnerability_score'] > 0.7:
            summary['recommendations'].append("High vulnerability detected - consider adversarial training")
        elif summary['overall_vulnerability_score'] > 0.4:
            summary['recommendations'].append("Moderate vulnerability - implement input validation")
        else:
            summary['recommendations'].append("Low vulnerability - monitor for new attack vectors")
        
        return summary


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="AI Shield - Adversarial ML Security Analysis")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--model", required=True, help="S3 key for model file")
    parser.add_argument("--data", required=True, help="S3 key for data file")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--output", default="results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize AI Shield engine
        engine = AIShieldEngine(args.bucket, args.region)
        
        # Run security analysis
        results = engine.run_security_analysis(args.model, args.data)
        
        # Save results to file
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("=" * 60)
        print("AI SHIELD SECURITY ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Overall Vulnerability Score: {results['vulnerability_summary']['overall_vulnerability_score']:.2f}")
        print(f"Execution Time: {results['metadata']['execution_time_seconds']:.2f} seconds")
        print(f"Results saved to: {args.output}")
        print("\nAttack Success Rates:")
        for attack, rate in results['vulnerability_summary']['attack_success_rates'].items():
            print(f"  {attack.upper()}: {rate:.2f}")
        
        print("\nRecommendations:")
        for rec in results['vulnerability_summary']['recommendations']:
            print(f"  - {rec}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
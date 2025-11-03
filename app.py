"""
AI Shield Web Frontend - Flask Application

A web interface for the AI Shield adversarial machine learning security analysis system.
Provides an intuitive UI for uploading models, running attacks, and viewing results.
"""

import os
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# AI Shield imports
import sys
sys.path.append('.')
from main import AIShieldEngine
from src.utils.model_loader import ModelLoader
from src.utils.data_processor import DataProcessor
from src.mitigation.fast_mitigation_engine import FastMitigationEngine

def save_mitigation_results(results, session_id):
    """Save mitigation results to file."""
    try:
        results_file = f"results/mitigation_{session_id}.json"
        os.makedirs('results', exist_ok=True)
        
        # Convert any non-serializable objects
        serializable_results = {}
        for strategy, result in results.items():
            serializable_result = {
                'strategy': result.get('strategy', strategy),
                'success': result.get('success', False),
                'robustness_improvement': result.get('robustness_improvement', 0),
                'original_accuracy': result.get('original_accuracy', 0),
                'hardened_accuracy': result.get('hardened_accuracy', 0),
                'processing_time': result.get('processing_time', 'Unknown'),
                'error': result.get('error', None)
            }
            serializable_results[strategy] = serializable_result
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results_file
    except Exception as e:
        logging.error(f"Failed to save mitigation results: {e}")
        return None

app = Flask(__name__)
app.secret_key = 'ai-shield-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configuration
UPLOAD_FOLDER = Path('uploads')
RESULTS_FOLDER = Path('results')
ALLOWED_MODEL_EXTENSIONS = {'.pkl', '.joblib', '.pth', '.pt', '.h5', '.onnx'}
ALLOWED_DATA_EXTENSIONS = {'.csv', '.json', '.parquet', '.xlsx'}

# Create directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Store analysis status
analysis_status = {}

# Store mitigation status
mitigation_status = {}

def allowed_file(filename, allowed_extensions):
    """Check if file has allowed extension."""
    return Path(filename).suffix.lower() in allowed_extensions

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    """Handle file uploads."""
    if request.method == 'GET':
        return render_template('upload.html')
    
    # Check if files are present
    if 'model_file' not in request.files or 'data_file' not in request.files:
        flash('Both model and data files are required!', 'error')
        return redirect(request.url)
    
    model_file = request.files['model_file']
    data_file = request.files['data_file']
    
    # Check if files are selected
    if model_file.filename == '' or data_file.filename == '':
        flash('Please select both model and data files!', 'error')
        return redirect(request.url)
    
    # Validate file types
    if not allowed_file(model_file.filename, ALLOWED_MODEL_EXTENSIONS):
        flash(f'Model file must be one of: {", ".join(ALLOWED_MODEL_EXTENSIONS)}', 'error')
        return redirect(request.url)
    
    if not allowed_file(data_file.filename, ALLOWED_DATA_EXTENSIONS):
        flash(f'Data file must be one of: {", ".join(ALLOWED_DATA_EXTENSIONS)}', 'error')
        return redirect(request.url)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_folder = UPLOAD_FOLDER / session_id
        session_folder.mkdir(exist_ok=True)
        
        # Save files
        model_filename = secure_filename(model_file.filename)
        data_filename = secure_filename(data_file.filename)
        
        model_path = session_folder / model_filename
        data_path = session_folder / data_filename
        
        model_file.save(str(model_path))
        data_file.save(str(data_path))
        
        # Initialize analysis status
        analysis_status[session_id] = {
            'status': 'uploaded',
            'model_file': model_filename,
            'data_file': data_filename,
            'model_path': str(model_path),
            'data_path': str(data_path),
            'created_at': datetime.now().isoformat(),
            'progress': 0
        }
        
        flash('Files uploaded successfully!', 'success')
        return redirect(url_for('configure_analysis', session_id=session_id))
    
    except Exception as e:
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/configure/<session_id>')
def configure_analysis(session_id):
    """Configure analysis parameters."""
    
    # Handle demo session
    if session_id == 'demo':
        return setup_demo_session()
    
    if session_id not in analysis_status:
        flash('Session not found!', 'error')
        return redirect(url_for('upload_files'))
    
    session_data = analysis_status[session_id]
    return render_template('configure.html', session_id=session_id, session_data=session_data)

def setup_demo_session():
    """Set up a demo session with financial sector sample data."""
    demo_session_id = 'demo_' + str(uuid.uuid4())[:8]
    
    # Check if we have financial demo data available
    demo_model_path = None
    demo_data_path = None
    demo_description = ""
    
    # Priority order for financial demo models (most interesting first)
    financial_demos = [
        {
            'model': 'models/fraud_detection_neural_network_model.joblib',
            'data': 'data/fraud_detection_neural_network_dataset.csv',
            'description': 'Credit Card Fraud Detection (Neural Network)',
            'icon': '🔍'
        },
        {
            'model': 'models/credit_risk_random_forest_model.joblib',
            'data': 'data/credit_risk_random_forest_dataset.csv',
            'description': 'Credit Risk Assessment (Random Forest)',
            'icon': '📊'
        },
        {
            'model': 'models/algorithmic_trading_svm_model.joblib',
            'data': 'data/algorithmic_trading_svm_dataset.csv',
            'description': 'Algorithmic Trading Signals (SVM)',
            'icon': '📈'
        },
        {
            'model': 'models/aml_detection_logistic_model.joblib',
            'data': 'data/aml_detection_logistic_dataset.csv',
            'description': 'Anti-Money Laundering Detection (Logistic Regression)',
            'icon': '🕵️'
        },
        {
            'model': 'models/insurance_fraud_decision_tree_model.joblib',
            'data': 'data/insurance_fraud_decision_tree_dataset.csv',
            'description': 'Insurance Fraud Detection (Decision Tree)',
            'icon': '🛡️'
        }
    ]
    
    # Try each financial demo in priority order
    for demo in financial_demos:
        model_path = Path(demo['model'])
        data_path = Path(demo['data'])
        
        if model_path.exists() and data_path.exists():
            demo_model_path = str(model_path)
            demo_data_path = str(data_path)
            demo_description = f"{demo['icon']} {demo['description']}"
            break
    
    # Fallback: any available financial model/data pair
    if not demo_model_path:
        models_dir = Path('models')
        data_dir = Path('data')
        
        if models_dir.exists() and data_dir.exists():
            # Look for any financial model pairs
            financial_prefixes = ['fraud_detection', 'credit_risk', 'algorithmic_trading', 'aml_detection', 'insurance_fraud']
            
            for prefix in financial_prefixes:
                model_files = list(models_dir.glob(f'{prefix}_*.joblib'))
                if model_files:
                    model_file = model_files[0]
                    # Find corresponding dataset
                    model_name = model_file.stem  # e.g., 'fraud_detection_neural_network_model'
                    data_name = model_name.replace('_model', '_dataset.csv')
                    data_file = data_dir / data_name
                    
                    if data_file.exists():
                        demo_model_path = str(model_file)
                        demo_data_path = str(data_file)
                        demo_description = f"💼 Financial Model: {prefix.replace('_', ' ').title()}"
                        break
    
    # Final fallback: legacy demo files
    if not demo_model_path:
        demo_model = Path('data/demo_model.joblib')
        demo_dataset = Path('data/demo_dataset.csv')
        
        if demo_model.exists() and demo_dataset.exists():
            demo_model_path = str(demo_model)
            demo_data_path = str(demo_dataset)
            demo_description = "📋 Legacy Demo Model"
    
    if not demo_model_path or not demo_data_path:
        flash('Financial demo data not available. Please run: python create_demo_models.py to create financial models.', 'warning')
        return redirect(url_for('upload_files'))
    
    # Create demo session
    analysis_status[demo_session_id] = {
        'model_path': demo_model_path,
        'data_path': demo_data_path,
        'model_file': Path(demo_model_path).name,
        'data_file': Path(demo_data_path).name,
        'status': 'configured',
        'upload_time': datetime.now().isoformat(),
        'is_demo': True,
        'demo_description': demo_description
    }
    
    flash(f'Financial Demo Created! {demo_description}', 'success')
    return render_template('configure.html', session_id=demo_session_id, session_data=analysis_status[demo_session_id])

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start the adversarial analysis."""
    session_id = request.form.get('session_id')
    
    if session_id not in analysis_status:
        return jsonify({'error': 'Session not found'}), 404
    
    # Get configuration parameters
    config = {
        'fgsm_epsilon': float(request.form.get('fgsm_epsilon', 0.1)),
        'pgd_epsilon': float(request.form.get('pgd_epsilon', 0.1)),
        'pgd_alpha': float(request.form.get('pgd_alpha', 0.01)),
        'pgd_iterations': int(request.form.get('pgd_iterations', 40)),
        'max_samples': int(request.form.get('max_samples', 100)),
        'include_shap': request.form.get('include_shap') == 'on',
        'include_lime': request.form.get('include_lime') == 'on'
    }
    
    # Update status
    analysis_status[session_id].update({
        'status': 'running',
        'config': config,
        'progress': 10
    })
    
    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis_background, args=(session_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'session_id': session_id})

def run_analysis_background(session_id):
    """Run analysis in background thread."""
    try:
        session_data = analysis_status[session_id]
        model_path = session_data['model_path']
        data_path = session_data['data_path']
        config = session_data['config']
        
        # Update progress
        analysis_status[session_id]['progress'] = 20
        analysis_status[session_id]['current_step'] = 'Loading model and data'
        
        # Load model and data
        model_loader = ModelLoader()
        data_processor = DataProcessor()
        
        model = model_loader.load_model(model_path)
        data = data_processor.load_and_process_data(data_path)
        
        analysis_status[session_id]['progress'] = 40
        analysis_status[session_id]['current_step'] = 'Running adversarial attacks'
        
        # Run attacks
        from src.adversarial.attack_engine import AdversarialAttackEngine
        attack_engine = AdversarialAttackEngine()
        
        # FGSM Attack
        fgsm_results = attack_engine.run_fgsm_attack(
            model, data['X_test'], data['y_test'], 
            epsilon=config['fgsm_epsilon'], framework='sklearn'
        )
        
        analysis_status[session_id]['progress'] = 60
        
        # PGD Attack
        pgd_results = attack_engine.run_pgd_attack(
            model, data['X_test'], data['y_test'],
            epsilon=config['pgd_epsilon'], alpha=config['pgd_alpha'], 
            max_iter=config['pgd_iterations'], framework='sklearn'
        )
        
        analysis_status[session_id]['progress'] = 75
        analysis_status[session_id]['current_step'] = 'Generating explanations'
        
        # XAI Analysis
        xai_results = {}
        if config['include_shap'] or config['include_lime']:
            from src.xai.explanation_engine import XAIExplanationEngine
            xai_engine = XAIExplanationEngine()
            
            attack_results = {'fgsm': fgsm_results, 'pgd': pgd_results}
            
            if config['include_shap']:
                shap_results = xai_engine.generate_shap_explanations(
                    model, data['X_test'], attack_results, 
                    data['feature_names'], max_samples=config['max_samples']
                )
                xai_results['shap'] = shap_results
            
            if config['include_lime']:
                lime_results = xai_engine.generate_lime_explanations(
                    model, data['X_test'], data['feature_names'], 
                    attack_results, max_samples=config['max_samples']
                )
                xai_results['lime'] = lime_results
        
        analysis_status[session_id]['progress'] = 90
        analysis_status[session_id]['current_step'] = 'Generating report'
        
        # Compile results
        results = {
            'metadata': {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'model_file': session_data.get('model_file', Path(session_data.get('model_path', '')).name),
                'data_file': session_data.get('data_file', Path(session_data.get('data_path', '')).name),
                'configuration': config
            },
            'dataset_info': {
                'num_samples': len(data['X_test']),
                'num_features': len(data['feature_names']),
                'feature_names': data['feature_names']
            },
            'adversarial_attacks': {
                'fgsm': fgsm_results,
                'pgd': pgd_results
            },
            'xai_explanations': xai_results,
            'vulnerability_summary': {
                'overall_vulnerability_score': (fgsm_results['success_rate'] + pgd_results['success_rate']) / 2,
                'attack_success_rates': {
                    'fgsm': fgsm_results['success_rate'],
                    'pgd': pgd_results['success_rate']
                },
                'critical_features': xai_results.get('shap', {}).get('important_features', [])[:5],
                'recommendations': []
            }
        }
        
        # Add recommendations
        vulnerability_score = results['vulnerability_summary']['overall_vulnerability_score']
        if vulnerability_score > 0.7:
            results['vulnerability_summary']['recommendations'].append("HIGH RISK: Implement adversarial training immediately")
            results['vulnerability_summary']['recommendations'].append("Add input validation and preprocessing")
            results['vulnerability_summary']['recommendations'].append("Consider ensemble methods for robustness")
        elif vulnerability_score > 0.4:
            results['vulnerability_summary']['recommendations'].append("MEDIUM RISK: Implement defensive distillation")
            results['vulnerability_summary']['recommendations'].append("Add feature preprocessing and normalization")
            results['vulnerability_summary']['recommendations'].append("Monitor for adversarial patterns in production")
        else:
            results['vulnerability_summary']['recommendations'].append("LOW RISK: Continue monitoring for new attack vectors")
            results['vulnerability_summary']['recommendations'].append("Regular security assessments recommended")
        
        # Save results
        results_file = RESULTS_FOLDER / f"{session_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update final status
        analysis_status[session_id].update({
            'status': 'completed',
            'progress': 100,
            'current_step': 'Analysis complete',
            'results_file': str(results_file),
            'results': results,
            'completed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        analysis_status[session_id].update({
            'status': 'error',
            'error': str(e),
            'progress': 0
        })
        logging.error(f"Analysis failed for session {session_id}: {str(e)}")

@app.route('/status/<session_id>')
def get_status(session_id):
    """Get analysis status."""
    if session_id not in analysis_status:
        return jsonify({'error': 'Session not found'}), 404
    
    status_data = analysis_status[session_id].copy()
    # Remove large data from status response
    if 'results' in status_data:
        del status_data['results']
    
    return jsonify(status_data)

@app.route('/results/<session_id>')
def view_results(session_id):
    """View analysis results."""
    if session_id not in analysis_status:
        flash('Session not found!', 'error')
        return redirect(url_for('index'))
    
    session_data = analysis_status[session_id]
    
    if session_data['status'] != 'completed':
        return render_template('progress.html', session_id=session_id, session_data=session_data)
    
    results = session_data.get('results', {})
    return render_template('results.html', session_id=session_id, results=results)

@app.route('/download/<session_id>')
def download_results(session_id):
    """Download results as JSON."""
    if session_id not in analysis_status:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = analysis_status[session_id]
    
    if session_data['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    results_file = session_data.get('results_file')
    if not results_file or not os.path.exists(results_file):
        return jsonify({'error': 'Results file not found'}), 404
    
    return send_file(results_file, as_attachment=True, download_name=f'ai_shield_results_{session_id}.json')

@app.route('/api/sessions')
def list_sessions():
    """List all analysis sessions."""
    sessions = []
    for sid, data in analysis_status.items():
        session_info = {
            'session_id': sid,
            'status': data['status'],
            'model_file': data.get('model_file', ''),
            'data_file': data.get('data_file', ''),
            'created_at': data.get('created_at', ''),
            'progress': data.get('progress', 0)
        }
        if data['status'] == 'completed':
            session_info['vulnerability_score'] = data.get('results', {}).get('vulnerability_summary', {}).get('overall_vulnerability_score', 0)
        sessions.append(session_info)
    
    return jsonify(sessions)

@app.route('/dashboard')
def dashboard():
    """Analysis dashboard."""
    return render_template('dashboard.html')

@app.route('/mitigate/<session_id>', methods=['GET', 'POST'])
def mitigate_model(session_id):
    """Apply mitigation strategies to vulnerable model."""
    if session_id not in analysis_status:
        flash('Session not found!', 'error')
        return redirect(url_for('index'))
    
    session_data = analysis_status[session_id]
    
    if session_data['status'] != 'completed':
        flash('Analysis must be completed before mitigation!', 'error')
        return redirect(url_for('view_results', session_id=session_id))
    
    if request.method == 'GET':
        # Show mitigation options page
        return render_template('mitigation.html', 
                             session_id=session_id, 
                             session_data=session_data)
    
    elif request.method == 'POST':
        # Run mitigation strategies
        selected_strategies = request.form.getlist('strategies')
        
        if not selected_strategies:
            flash('Please select at least one mitigation strategy!', 'warning')
            return render_template('mitigation.html', 
                                 session_id=session_id, 
                                 session_data=session_data)
        
        # Start mitigation in background
        mitigation_session_id = f"{session_id}_mitigation"
        
        analysis_status[mitigation_session_id] = {
            'status': 'running',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'original_session': session_id,
            'selected_strategies': selected_strategies
        }
        
        # Run mitigation in background thread
        thread = threading.Thread(
            target=run_mitigation_analysis,
            args=(mitigation_session_id, session_id, selected_strategies)
        )
        thread.daemon = True
        thread.start()
        
        return render_template('mitigation_progress.html', 
                             session_id=mitigation_session_id,
                             original_session=session_id)

@app.route('/mitigation_results/<session_id>')
def view_mitigation_results(session_id):
    """View mitigation analysis results."""
    if session_id not in analysis_status:
        flash('Mitigation session not found!', 'error')
        return redirect(url_for('index'))
    
    session_data = analysis_status[session_id]
    
    if session_data['status'] != 'completed':
        return render_template('mitigation_progress.html', 
                             session_id=session_id,
                             session_data=session_data)
    
    results = session_data.get('results', {})
    original_session = session_data.get('original_session', '')
    
    return render_template('mitigation_results.html', 
                         session_id=session_id,
                         original_session=original_session,
                         results=results)

@app.route('/download_hardened_model/<session_id>')
def download_hardened_model(session_id):
    """Download the best hardened model."""
    if session_id not in analysis_status:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = analysis_status[session_id]
    
    if session_data['status'] != 'completed':
        return jsonify({'error': 'Mitigation not completed'}), 400
    
    hardened_model_file = session_data.get('hardened_model_file')
    if not hardened_model_file or not os.path.exists(hardened_model_file):
        return jsonify({'error': 'Hardened model file not found'}), 404
    
    return send_file(hardened_model_file, 
                    as_attachment=True, 
                    download_name=f'hardened_model_{session_id}.joblib')

def run_mitigation_analysis(mitigation_session_id, original_session_id, selected_strategies):
    """Run mitigation analysis in background thread."""
    try:
        # Update progress
        analysis_status[mitigation_session_id].update({
            'status': 'running',
            'progress': 10,
            'message': 'Loading original analysis results...'
        })
        
        # Get original session data
        original_session = analysis_status[original_session_id]
        model_path = original_session.get('model_path', original_session['model_file'])
        data_path = original_session.get('data_path', original_session['data_file'])
        
        # Ensure we have full paths - avoid double path prefixes
        if not os.path.isabs(model_path) and not model_path.startswith('models/'):
            model_path = os.path.join('models', model_path)
        if not os.path.isabs(data_path) and not data_path.startswith('data/'):
            data_path = os.path.join('data', data_path)
        
        # Load model and data
        analysis_status[mitigation_session_id].update({
            'progress': 20,
            'message': 'Loading model and data...'
        })
        
        model_loader = ModelLoader()
        data_processor = DataProcessor()
        
        model = model_loader.load_model(model_path)
        data_dict = data_processor.load_and_process_data(data_path)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test'] 
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # Initialize mitigation engine
        analysis_status[mitigation_session_id].update({
            'progress': 30,
            'message': 'Initializing mitigation engine...'
        })
        
        mitigation_engine = FastMitigationEngine()
        
        # Run selected mitigation strategies using the optimized method
        analysis_status[mitigation_session_id].update({
            'progress': 40,
            'message': 'Running mitigation strategies...'
        })
        
        results = mitigation_engine.apply_mitigation_strategies(
            model, X_train, y_train, X_test, y_test, selected_strategies
        )
        
        analysis_status[mitigation_session_id].update({
            'progress': 80,
            'message': 'Processing results...'
        })
        
        # Find best strategy
        analysis_status[mitigation_session_id].update({
            'progress': 90,
            'message': 'Generating recommendations...'
        })
        
        successful_strategies = [r for r in results.values() if r.get('success', False)]
        
        if successful_strategies:
            best_strategy = max(successful_strategies, 
                              key=lambda x: x.get('robustness_improvement', 0))
            
            # Save best hardened model if available
            hardened_model_path = None
            if 'model' in best_strategy:
                hardened_model_path = RESULTS_FOLDER / f'hardened_model_{mitigation_session_id}.joblib'
                import joblib
                joblib.dump(best_strategy['model'], hardened_model_path)
            
            summary = {
                'total_strategies': len(selected_strategies),
                'successful_strategies': len(successful_strategies),
                'best_strategy': best_strategy.get('strategy', 'Unknown'),
                'best_improvement': best_strategy.get('robustness_improvement', 0),
                'original_vulnerability': original_session.get('results', {}).get('vulnerability_summary', {}).get('overall_vulnerability_score', 0),
                'recommendations': mitigation_engine._generate_recommendations(results)
            }
        else:
            hardened_model_path = None
            summary = {
                'total_strategies': len(selected_strategies),
                'successful_strategies': 0,
                'best_strategy': None,
                'best_improvement': 0,
                'recommendations': ['All selected mitigation strategies failed. Try different approaches.']
            }
        
        # Save results
        final_results = {
            'mitigation_results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            'selected_strategies': selected_strategies
        }
        
        results_file = save_mitigation_results(final_results['mitigation_results'], mitigation_session_id)
        
        # Update session status
        analysis_status[mitigation_session_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Mitigation analysis completed!',
            'results': final_results,
            'results_file': results_file,
            'hardened_model_file': str(hardened_model_path) if hardened_model_path else None
        })
        
    except Exception as e:
        analysis_status[mitigation_session_id].update({
            'status': 'failed',
            'progress': 0,
            'error': str(e),
            'message': f'Mitigation failed: {str(e)}'
        })
        logging.error(f"Mitigation failed for session {mitigation_session_id}: {str(e)}")

@app.errorhandler(413)
def too_large(e):
    flash('File is too large! Maximum size is 500MB.', 'error')
    return redirect(url_for('upload_files'))

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    
    print("🚀 Starting AI Shield Web Interface...")
    print(f"📊 Dashboard: http://localhost:{port}")
    print(f"📤 Upload: http://localhost:{port}/upload")
    print(f"🔍 API: http://localhost:{port}/api/sessions")
    
    app.run(debug=False, host='0.0.0.0', port=port)
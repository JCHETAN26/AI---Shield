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
    """Set up a demo session with sample data."""
    demo_session_id = 'demo_' + str(uuid.uuid4())[:8]
    
    # Check if we have demo data available
    demo_model_path = None
    demo_data_path = None
    
    # First priority: dedicated demo files in data directory
    demo_model = Path('data/demo_model.joblib')
    demo_dataset = Path('data/demo_dataset.csv')
    
    if demo_model.exists() and demo_dataset.exists():
        demo_model_path = str(demo_model)
        demo_data_path = str(demo_dataset)
    else:
        # Second priority: test_data directory
        test_data_dir = Path('test_data')
        if test_data_dir.exists():
            # Try to find a good demo model/data pair
            binary_simple_dir = test_data_dir / 'binary_simple'
            if binary_simple_dir.exists():
                rf_model = binary_simple_dir / 'random_forest' / 'random_forest_model.joblib'
                test_data = binary_simple_dir / 'binary_simple_test_data.csv'
                
                if rf_model.exists() and test_data.exists():
                    demo_model_path = str(rf_model)
                    demo_data_path = str(test_data)
    
    # Fallback: any available demo data
    if not demo_model_path:
        data_dir = Path('data')
        if data_dir.exists():
            # Look for sample model and data
            model_files = list(data_dir.glob('*.pkl')) + list(data_dir.glob('*.joblib'))
            data_files = list(data_dir.glob('*.csv'))
            
            if model_files and data_files:
                demo_model_path = str(model_files[0])
                demo_data_path = str(data_files[0])
    
    if not demo_model_path or not demo_data_path:
        flash('Demo data not available. Please run: python generate_test_data.py to create sample data.', 'warning')
        return redirect(url_for('upload_files'))
    
    # Create demo session
    analysis_status[demo_session_id] = {
        'model_path': demo_model_path,
        'data_path': demo_data_path,
        'model_file': Path(demo_model_path).name,
        'data_file': Path(demo_data_path).name,
        'status': 'configured',
        'upload_time': datetime.now().isoformat(),
        'is_demo': True
    }
    
    flash(f'Demo session created! Using {Path(demo_model_path).name} and {Path(demo_data_path).name}', 'success')
    return render_template('configure.html', session_id=demo_session_id, session_data=analysis_status[demo_session_id])

@app.route('/demo')
def demo():
    """Direct route for demo access."""
    return setup_demo_session()

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

@app.errorhandler(413)
def too_large(e):
    flash('File is too large! Maximum size is 500MB.', 'error')
    return redirect(url_for('upload_files'))

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting AI Shield Web Interface...")
    print("📊 Dashboard: http://localhost:5000")
    print("📤 Upload: http://localhost:5000/upload")
    print("🔍 API: http://localhost:5000/api/sessions")
    
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def create_fraud_detection_dataset(n_samples=5000):
    """Create a dataset for credit card fraud detection"""
    print("Creating fraud detection dataset...")
    
    features = {}
    
    # Transaction features
    features['transaction_amount'] = np.random.lognormal(3, 2, n_samples)  # Log-normal for realistic amounts
    features['merchant_category'] = np.random.randint(1, 20, n_samples)
    features['transaction_hour'] = np.random.randint(0, 24, n_samples)
    features['transaction_day_of_week'] = np.random.randint(0, 7, n_samples)
    
    # Account features
    features['account_age_days'] = np.random.randint(30, 3650, n_samples)
    features['previous_transactions_24h'] = np.random.poisson(3, n_samples)
    features['avg_transaction_amount'] = np.random.normal(150, 100, n_samples)
    features['account_balance'] = np.random.normal(2000, 1500, n_samples)
    
    # Location features (normalized)
    features['location_risk_score'] = np.random.beta(2, 5, n_samples)
    features['distance_from_home'] = np.random.exponential(50, n_samples)
    features['distance_from_last_transaction'] = np.random.exponential(20, n_samples)
    
    # Behavioral features
    features['velocity_1h'] = np.random.poisson(1, n_samples)
    features['velocity_24h'] = np.random.poisson(5, n_samples)
    features['unusual_time'] = (features['transaction_hour'] < 6) | (features['transaction_hour'] > 23)
    features['weekend_transaction'] = features['transaction_day_of_week'] >= 5
    
    # Risk indicators
    features['merchant_risk_score'] = np.random.beta(3, 7, n_samples)
    features['ip_risk_score'] = np.random.beta(2, 8, n_samples)
    features['device_risk_score'] = np.random.beta(2, 8, n_samples)
    
    # Create fraud labels (imbalanced dataset - 2% fraud)
    fraud_probability = (
        (features['transaction_amount'] > np.percentile(features['transaction_amount'], 95)) * 0.3 +
        (features['location_risk_score'] > 0.7) * 0.4 +
        (features['distance_from_home'] > 100) * 0.2 +
        (features['velocity_24h'] > 10) * 0.3 +
        features['unusual_time'].astype(float) * 0.2 +
        (features['merchant_risk_score'] > 0.8) * 0.4 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to binary classification
    fraud_threshold = np.percentile(fraud_probability, 98)  # Top 2% are fraud
    targets = (fraud_probability > fraud_threshold).astype(int)
    
    df = pd.DataFrame(features)
    df['fraud_probability'] = fraud_probability
    df['target'] = targets
    
    return df

def create_credit_risk_dataset(n_samples=3000):
    """Create a dataset for credit risk assessment"""
    print("Creating credit risk dataset...")
    
    features = {}
    
    # Personal demographics
    features['age'] = np.random.normal(40, 15, n_samples).clip(18, 80)
    features['income'] = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000)
    features['employment_length'] = np.random.exponential(5, n_samples).clip(0, 40)
    features['debt_to_income'] = np.random.beta(2, 5, n_samples)
    
    # Credit history
    features['credit_score'] = np.random.normal(680, 120, n_samples).clip(300, 850)
    features['credit_history_length'] = np.random.exponential(8, n_samples).clip(0, 50)
    features['num_credit_lines'] = np.random.poisson(5, n_samples).clip(0, 20)
    features['credit_utilization'] = np.random.beta(2, 3, n_samples)
    
    # Loan details
    features['loan_amount'] = np.random.lognormal(9.5, 1.2, n_samples).clip(1000, 100000)
    features['loan_term'] = np.random.choice([12, 24, 36, 48, 60], n_samples)
    features['loan_to_income'] = features['loan_amount'] / features['income']
    
    # Financial behavior
    features['num_delinquencies'] = np.random.poisson(0.5, n_samples)
    features['num_public_records'] = np.random.poisson(0.1, n_samples)
    features['num_inquiries_6m'] = np.random.poisson(1, n_samples)
    features['savings_account'] = np.random.binomial(1, 0.7, n_samples)
    features['checking_account'] = np.random.binomial(1, 0.8, n_samples)
    
    # Housing
    features['home_ownership'] = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.6, 0.1])  # 0=Rent, 1=Own, 2=Mortgage
    features['monthly_housing_cost'] = features['income'] * np.random.beta(2, 6, n_samples) / 12
    
    # Calculate default probability
    default_score = (
        -0.3 * (features['credit_score'] - 600) / 100 +
        0.4 * features['debt_to_income'] +
        0.3 * features['loan_to_income'] +
        0.2 * features['num_delinquencies'] +
        0.1 * features['credit_utilization'] +
        -0.2 * features['employment_length'] / 10 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to risk categories: 0=Low, 1=Medium, 2=High
    targets = np.zeros(n_samples)
    targets[default_score > np.percentile(default_score, 70)] = 1
    targets[default_score > np.percentile(default_score, 90)] = 2
    
    df = pd.DataFrame(features)
    df['default_score'] = default_score
    df['target'] = targets.astype(int)
    
    return df

def create_algorithmic_trading_dataset(n_samples=2000):
    """Create a dataset for algorithmic trading signal prediction"""
    print("Creating algorithmic trading dataset...")
    
    features = {}
    
    # Technical indicators
    features['rsi_14'] = np.random.beta(2, 2, n_samples) * 100
    features['macd'] = np.random.normal(0, 2, n_samples)
    features['bollinger_position'] = np.random.beta(2, 2, n_samples)
    features['volume_ratio'] = np.random.lognormal(0, 0.5, n_samples)
    
    # Price features (normalized)
    features['price_change_1d'] = np.random.normal(0, 0.02, n_samples)
    features['price_change_5d'] = np.random.normal(0, 0.05, n_samples)
    features['price_change_20d'] = np.random.normal(0, 0.1, n_samples)
    features['volatility_20d'] = np.random.exponential(0.02, n_samples)
    
    # Market indicators
    features['market_sentiment'] = np.random.normal(0, 1, n_samples)
    features['sector_performance'] = np.random.normal(0, 0.03, n_samples)
    features['vix_level'] = np.random.gamma(2, 10, n_samples)
    features['yield_curve_slope'] = np.random.normal(1, 0.5, n_samples)
    
    # Order book features
    features['bid_ask_spread'] = np.random.exponential(0.001, n_samples)
    features['order_imbalance'] = np.random.normal(0, 0.3, n_samples)
    features['trade_intensity'] = np.random.exponential(100, n_samples)
    
    # News sentiment
    features['news_sentiment'] = np.random.normal(0, 1, n_samples)
    features['social_sentiment'] = np.random.normal(0, 1, n_samples)
    features['analyst_consensus'] = np.random.normal(0, 1, n_samples)
    
    # Generate trading signals: 0=Sell, 1=Hold, 2=Buy
    signal_score = (
        0.3 * (features['rsi_14'] - 50) / 50 +
        0.2 * features['macd'] +
        0.2 * features['price_change_5d'] / 0.05 +
        0.1 * features['market_sentiment'] +
        0.1 * features['news_sentiment'] +
        0.1 * features['volume_ratio'] +
        np.random.normal(0, 0.3, n_samples)
    )
    
    targets = np.ones(n_samples)  # Default to hold
    targets[signal_score < -0.5] = 0  # Sell
    targets[signal_score > 0.5] = 2   # Buy
    
    df = pd.DataFrame(features)
    df['signal_score'] = signal_score
    df['target'] = targets.astype(int)
    
    return df

def create_aml_detection_dataset(n_samples=2500):
    """Create a dataset for Anti-Money Laundering (AML) detection"""
    print("Creating AML detection dataset...")
    
    features = {}
    
    # Account characteristics
    features['account_age_months'] = np.random.exponential(24, n_samples)
    features['account_type'] = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1])  # Personal, Business, Corporate
    features['customer_risk_rating'] = np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05])  # Low, Medium, High
    
    # Transaction patterns
    features['monthly_transaction_count'] = np.random.poisson(15, n_samples)
    features['monthly_transaction_volume'] = np.random.lognormal(8, 2, n_samples)
    features['avg_transaction_size'] = features['monthly_transaction_volume'] / features['monthly_transaction_count']
    features['transaction_variance'] = np.random.exponential(1000, n_samples)
    
    # Geographic features
    features['num_countries'] = np.random.poisson(1.5, n_samples)
    features['high_risk_country_exposure'] = np.random.binomial(1, 0.1, n_samples)
    features['cross_border_ratio'] = np.random.beta(1, 4, n_samples)
    
    # Network features
    features['num_counterparties'] = np.random.poisson(8, n_samples)
    features['counterparty_risk_score'] = np.random.beta(2, 8, n_samples)
    features['network_centrality'] = np.random.exponential(0.1, n_samples)
    features['shell_company_exposure'] = np.random.binomial(1, 0.05, n_samples)
    
    # Behavioral indicators
    features['cash_intensive_ratio'] = np.random.beta(2, 8, n_samples)
    features['round_amount_ratio'] = np.random.beta(1, 9, n_samples)
    features['structuring_indicator'] = np.random.binomial(1, 0.02, n_samples)
    features['rapid_movement_score'] = np.random.exponential(0.1, n_samples)
    
    # Industry/occupation risk
    features['industry_risk_score'] = np.random.beta(3, 7, n_samples)
    features['pep_exposure'] = np.random.binomial(1, 0.01, n_samples)  # Politically Exposed Person
    features['sanctions_screening_hits'] = np.random.poisson(0.02, n_samples)
    
    # Calculate suspicious activity score
    suspicious_score = (
        features['high_risk_country_exposure'] * 2 +
        features['shell_company_exposure'] * 3 +
        features['structuring_indicator'] * 4 +
        features['pep_exposure'] * 2 +
        features['sanctions_screening_hits'] * 5 +
        features['cash_intensive_ratio'] * 1.5 +
        features['rapid_movement_score'] * 2 +
        (features['customer_risk_rating'] == 2) * 1.5 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to binary: 0=Clean, 1=Suspicious (top 1% are suspicious)
    suspicious_threshold = np.percentile(suspicious_score, 99)
    targets = (suspicious_score > suspicious_threshold).astype(int)
    
    df = pd.DataFrame(features)
    df['suspicious_score'] = suspicious_score
    df['target'] = targets
    
    return df

def create_insurance_fraud_dataset(n_samples=2000):
    """Create a dataset for insurance fraud detection"""
    print("Creating insurance fraud dataset...")
    
    features = {}
    
    # Policy holder information
    features['age'] = np.random.normal(45, 18, n_samples).clip(18, 90)
    features['policy_tenure_years'] = np.random.exponential(3, n_samples).clip(0, 30)
    features['num_previous_claims'] = np.random.poisson(0.8, n_samples)
    features['policy_premium'] = np.random.lognormal(7, 1, n_samples)
    
    # Claim details
    features['claim_amount'] = np.random.lognormal(8.5, 1.5, n_samples)
    features['claim_to_premium_ratio'] = features['claim_amount'] / features['policy_premium']
    features['days_policy_to_claim'] = np.random.exponential(200, n_samples)
    features['claim_complexity_score'] = np.random.beta(3, 7, n_samples)
    
    # Incident characteristics
    features['incident_severity'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    features['police_report_filed'] = np.random.binomial(1, 0.3, n_samples)
    features['witness_present'] = np.random.binomial(1, 0.4, n_samples)
    features['incident_hour'] = np.random.randint(0, 24, n_samples)
    
    # Geographic and temporal patterns
    features['high_fraud_area'] = np.random.binomial(1, 0.15, n_samples)
    features['claim_frequency_in_area'] = np.random.poisson(2, n_samples)
    features['seasonal_factor'] = np.random.normal(1, 0.3, n_samples)
    
    # Network indicators
    features['shared_address_claims'] = np.random.poisson(0.1, n_samples)
    features['shared_phone_claims'] = np.random.poisson(0.05, n_samples)
    features['attorney_represented'] = np.random.binomial(1, 0.2, n_samples)
    features['medical_provider_risk'] = np.random.beta(2, 8, n_samples)
    
    # Behavioral flags
    features['immediate_medical_attention'] = np.random.binomial(1, 0.6, n_samples)
    features['claim_reporting_delay'] = np.random.exponential(2, n_samples)
    features['documentation_completeness'] = np.random.beta(8, 2, n_samples)
    features['story_consistency_score'] = np.random.beta(7, 3, n_samples)
    
    # Calculate fraud probability
    fraud_score = (
        features['high_fraud_area'] * 2 +
        (features['claim_amount'] > np.percentile(features['claim_amount'], 90)) * 1.5 +
        (features['days_policy_to_claim'] < 30) * 2 +
        features['shared_address_claims'] * 3 +
        features['shared_phone_claims'] * 4 +
        (features['num_previous_claims'] > 3) * 1.5 +
        (1 - features['story_consistency_score']) * 2 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Top 3% are fraudulent
    fraud_threshold = np.percentile(fraud_score, 97)
    targets = (fraud_score > fraud_threshold).astype(int)
    
    df = pd.DataFrame(features)
    df['fraud_score'] = fraud_score
    df['target'] = targets
    
    return df

def create_vulnerable_model(X, y, model_type='random_forest'):
    """Create intentionally vulnerable models for adversarial testing"""
    
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=5,
            max_depth=3,
            min_samples_split=10,
            random_state=42
        ),
        'svm': SVC(
            C=0.1,
            kernel='linear', 
            probability=True,
            random_state=42
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(10,),
            alpha=0.001,
            max_iter=100,
            random_state=42
        ),
        'logistic': LogisticRegression(
            C=0.1,
            max_iter=100,
            random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
    }
    
    return models[model_type]

def save_model_and_data(model, X_test, y_test, dataset_name, model_type):
    """Save model and test data"""
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Save model
    model_filename = f'models/{dataset_name}_{model_type}_model.joblib'
    joblib.dump(model, model_filename)
    
    # Save test data
    test_data = X_test.copy()
    test_data['target'] = y_test
    data_filename = f'data/{dataset_name}_{model_type}_dataset.csv'
    test_data.to_csv(data_filename, index=False)
    
    return model_filename, data_filename

def create_all_financial_models():
    """Create all financial sector demo models and datasets"""
    
    print("🏦 Creating AI Shield Financial Sector Models and Datasets")
    print("=" * 70)
    
    # Financial dataset configurations
    datasets = [
        {
            'name': 'fraud_detection',
            'creator': create_fraud_detection_dataset,
            'description': 'Credit card fraud detection (2% fraud rate)'
        },
        {
            'name': 'credit_risk',
            'creator': create_credit_risk_dataset,
            'description': 'Credit risk assessment (Low/Medium/High risk)'
        },
        {
            'name': 'algorithmic_trading',
            'creator': create_algorithmic_trading_dataset,
            'description': 'Algorithmic trading signals (Buy/Hold/Sell)'
        },
        {
            'name': 'aml_detection',
            'creator': create_aml_detection_dataset,
            'description': 'Anti-Money Laundering detection (1% suspicious)'
        },
        {
            'name': 'insurance_fraud',
            'creator': create_insurance_fraud_dataset,
            'description': 'Insurance fraud detection (3% fraud rate)'
        }
    ]
    
    # Model types to create for each dataset
    model_types = ['random_forest', 'svm', 'neural_network', 'logistic', 'decision_tree']
    
    created_files = []
    
    for dataset_config in datasets:
        print(f"\n💰 Processing: {dataset_config['description']}")
        print("-" * 50)
        
        # Create dataset
        df = dataset_config['creator']()
        
        # Prepare features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Classes: {sorted(y.unique())}")
        print(f"Class distribution: {dict(y.value_counts().sort_index())}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create models
        for model_type in model_types:
            print(f"  🤖 Training {model_type} model...")
            
            try:
                # Create and train model
                model = create_vulnerable_model(X_train, y_train, model_type)
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Save model and data
                model_file, data_file = save_model_and_data(
                    model, X_test, y_test, dataset_config['name'], model_type
                )
                
                created_files.extend([model_file, data_file])
                
                print(f"    ✓ Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
                print(f"    📁 Saved: {model_file}")
                print(f"    📁 Saved: {data_file}")
                
            except Exception as e:
                print(f"    ❌ Error with {model_type}: {str(e)}")
    
    print("\n" + "=" * 70)
    print("🎉 Financial Sector Model Creation Complete!")
    print(f"💼 Created {len(created_files)//2} financial models with datasets")
    print("\nFiles created:")
    for i, file in enumerate(created_files):
        print(f"  {i+1:2d}. {file}")
    
    print("\n💡 Financial Sector Use Cases:")
    print("  🔍 Fraud Detection: Credit card transaction monitoring")
    print("  📊 Credit Risk: Loan default probability assessment") 
    print("  📈 Algorithmic Trading: Market signal prediction")
    print("  🕵️  AML Detection: Suspicious activity identification")
    print("  🛡️  Insurance Fraud: Claims fraud detection")
    
    print("\n🔧 Usage:")
    print("  - Upload any .joblib model file and corresponding .csv dataset")
    print("  - Use the AI Shield web interface to analyze vulnerabilities")
    print("  - Test different adversarial attacks (FGSM, PGD)")
    print("  - View XAI explanations with SHAP and LIME")

if __name__ == "__main__":
    create_all_financial_models()
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import os

def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)

def prepare_data(data, is_training=False):
    """Prepare data for model training or prediction."""
    # Convert age ranges to numeric
    age_map = {'1-5': 3, '6-10': 8, '11-15': 13, '16-20': 18, '21-25': 23, '26-30': 28}
    data['Age'] = data['Age'].map(age_map)
    
    # Get categorical columns
    categorical_cols = ['Gender', 'Education Level', 'Institution Type', 'IT Student', 
                       'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
                       'Network Type', 'Class Duration', 'Self Lms', 'Device']
    
    # One-hot encode categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    
    # Only process target variable during training
    if is_training and 'Adaptivity Level' in data.columns:
        adaptivity_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        data_encoded['Adaptivity Level'] = data['Adaptivity Level'].map(adaptivity_map)
    
    return data_encoded

# function to save the model
def save_model(model, scaler, model_dir='models'):
    """Save model and scaler."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f'{model_dir}/best_model.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    # Save feature names
    feature_names = pd.DataFrame(columns=scaler.feature_names_in_)
    feature_names.to_csv(f'{model_dir}/feature_names.csv', index=False)




def load_model(model_dir='models'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, model_dir, 'best_model.pkl')
    scaler_path = os.path.join(base_dir, model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# feature names
def get_feature_names(data):
    """Get feature names after encoding."""
    return data.columns.tolist()

# for scaling the model
def scale_features(X_train, X_test=None):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

# prediction converter 
def convert_prediction_to_label(prediction):
    """Convert numeric prediction to categorical label."""
    if prediction < 0.5:
        return "Low"
    elif prediction < 1.5:
        return "Moderate"
    return "High"
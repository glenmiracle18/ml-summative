import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, prepare_data, save_model, scale_features

def train_model(data_path, model_dir='models'):
    """
    Train and evaluate models for student adaptability prediction.
    
    Args:
        data_path (str): Path to the dataset
        model_dir (str): Directory to save trained model
    """
    # Load and prepare data
    data = load_data(data_path)
    data_prepared = prepare_data(data, is_training=True)
    
    # Split features and target
    X = data_prepared.drop('Adaptivity Level', axis=1)
    y = data_prepared['Adaptivity Level']
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100)
    }
    
    # Train and evaluate models
    results = []
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert predictions to classes for accuracy
        y_pred_classes = np.where(y_pred < 0.5, 0, np.where(y_pred < 1.5, 1, 2))
        accuracy = accuracy_score(y_test, y_pred_classes)
        
        results.append({
            'Model': name,
            'MSE': mse,
            'R2': r2,
            'Accuracy': accuracy
        })
        
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model
    best_model_name = results_df.loc[results_df['R2'].idxmax(), 'Model']
    best_model = models[best_model_name]
    
    # Save best model and scaler
    save_model(best_model, scaler, model_dir)
    
    # Print results
    print("\nModel Performance:")
    print(results_df.round(4))
    print(f"\nBest performing model: {best_model_name}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R2 scores
    sns.barplot(data=results_df, x='Model', y='R2', ax=axes[0])
    axes[0].set_title('R2 Scores by Model')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Accuracy scores
    sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[1])
    axes[1].set_title('Accuracy Scores by Model')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return best_model, scaler, results_df

if __name__ == '__main__':
    # Train model
    best_model, scaler, results = train_model('data/student-adaptability-dataset.csv')
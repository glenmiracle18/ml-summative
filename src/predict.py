import pandas as pd
from utils import load_model, convert_prediction_to_label, prepare_data

def predict_student_adaptability(student_info, model_dir='models'):
    """Predict student adaptability level."""
    # Load model and scaler
    model, scaler = load_model(model_dir)
    
    # Get feature names from the model
    feature_names = scaler.feature_names_in_
    
    # Convert input to DataFrame and prepare
    df = pd.DataFrame([student_info])
    df_prepared = prepare_data(df)
    
    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in df_prepared.columns:
            df_prepared[feature] = 0
            
    # Reorder columns to match training data
    df_prepared = df_prepared.reindex(columns=feature_names, fill_value=0)
    
    # Scale features
    scaled_features = scaler.transform(df_prepared)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    return convert_prediction_to_label(prediction)


def batch_predict(student_list, model_dir='models'):
    """
    Make predictions for multiple students.
    
    Args:
        student_list (list): List of dictionaries containing student information
        model_dir (str): Directory containing saved model files
    
    Returns:
        list: Predicted adaptability levels
    """
    predictions = []
    for student in student_list:
        pred = predict_student_adaptability(student, model_dir)
        predictions.append(pred)
    return predictions

if __name__ == '__main__':
    # Example usage
    sample_student = {
        'Gender': 'Boy',
        'Age': '16-20',
        'Education Level': 'College',
        'Institution Type': 'Government',
        'IT Student': 'No',
        'Location': 'Yes',
        'Load-shedding': 'Low',
        'Financial Condition': 'Mid',
        'Internet Type': 'Wifi',
        'Network Type': '4G',
        'Class Duration': '1-3',
        'Self Lms': 'No',
        'Device': 'Mobile'
    }
    
    result = predict_student_adaptability(sample_student)
    print(f"Predicted Adaptability Level: {result}")
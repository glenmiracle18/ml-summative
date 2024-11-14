# Student Adaptability Prediction Project

This project aims to predict the adaptability level of students based on various factors such as demographics, educational background, and technology usage. The project involves data preprocessing, model training, and deployment of a machine learning model to make predictions.

## Project Structure

The project consists of the following components:

1. **Data**: The dataset used for training and testing the model is stored in the `data` directory.
2. **Model Training**: The `train_model.py` script is responsible for training and evaluating different machine learning models on the dataset. The best-performing model is saved along with its scaler for future predictions.
3. **Prediction**: The `predict.py` script contains functions for making predictions on new, unseen data. It loads the trained model and scaler, preprocesses the input data, and makes predictions.
4. **Mobile Application**: The project also includes a Flutter mobile application that allows users to input student information and receive predictions on their adaptability level.

## Technologies Used

* Python
* Pandas for data manipulation and analysis
* NumPy for numerical computing
* Scikit-learn for machine learning
* Flask for web application development
* Flutter for mobile app interface
  

## How to Use

1. Install the required packages by running `pip install -r requirements.txt`.
2. Train the model by running `python train_model.py`.
3. Start the server by running `python app.py`.
4. Open a web browser and navigate to `http://localhost:5000` to interact with the api.

## Future Work

* Collect more data to improve the model's accuracy and generalizability.
* Experiment with different machine learning models and techniques to improve predictions.
* Enhance the web application's user interface and user experience.
* Consider deploying the application on a cloud platform for wider accessibility.

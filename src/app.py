from flask import Flask, request, jsonify, render_template
from predict import predict_student_adaptability
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            student_data = request.get_json()
        else:
            student_data = request.form.to_dict()
            
        print("Received data:", student_data)  # Debug print
        prediction = predict_student_adaptability(student_data)
        return jsonify({'prediction': prediction})
    except Exception as e:
        print("Error:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
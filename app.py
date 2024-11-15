from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load ML model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input data
            data = [float(x) for x in request.form.values()]
            features = np.array([data])
            
            # Model prediction
            prediction = model.predict(features)
            result = prediction[0]

            return render_template('result.html', prediction=result)
        except Exception as e:
            return jsonify({'error': str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

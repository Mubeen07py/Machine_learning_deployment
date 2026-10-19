import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the saved model and scaler
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# We need the list of features the model expects after one-hot encoding.
# To dynamically reconstruct the input shape, we will read the feature names 
# that were saved with our scaler, or we can align inputs dynamically.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert data into DataFrame
        input_df = pd.DataFrame([data])
        
        # Handle One-Hot Encoding just like the training step
        # We align columns with the model's training columns
        # During training, the columns used were stored in the features of the scaler or training data.
        # Let's align them dynamically based on the scaler's expected feature count or pass them.
        # For simplicity, we assume the user sends pre-aligned features, or we can align them:
        # Let's align input_df with the training columns saved in our session.
        
        # In a real API, we need to ensure all 53 columns are present
        # We will dynamically reconstruct any missing dummies as False
        encoded_cols = scaler.feature_names_in_
        
        # Fill missing encoded columns with False/0
        for col in encoded_cols:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Ensure correct column order
        input_df = input_df[encoded_cols]
        
        # Scale the features
        scaled_features = scaler.transform(input_df)
        
        # Predict class and probabilities
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        response = {
            'prediction': int(prediction),
            'probability': {
                'class_0': float(prediction_proba[0]),
                'class_1': float(prediction_proba[1])
            }
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

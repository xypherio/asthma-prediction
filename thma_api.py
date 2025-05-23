import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the model
model = load('decision_tree_model.joblib')

# Load dataset used during training to extract feature names
x = pd.read_csv('asthma_data.csv')

# Print columns so we can debug and see the target column
print("Columns in dataset:", x.columns.tolist())

# Drop the target column to get features – update 'Target' to your actual target column name
try:
    feature_columns = x.drop(columns=['Asthma_Diagnosis']).columns.tolist()
except KeyError:
    raise KeyError("Update 'Target' to the actual name of your label column in asthma_data.csv")

print("Feature columns used for prediction:", feature_columns)

# Flask app setup
api = Flask(__name__)
CORS(api)

@api.route('/api/hfp_prediction', methods=['POST'])
def predict_brain_stroke():
    try:
        # Get JSON data
        data = request.json['inputs']
        input_df = pd.DataFrame(data)

        # Reorder input columns to match training feature order
        input_df = input_df[feature_columns]

        # Make prediction
        prediction = model.predict_proba(input_df)
        class_labels = model.classes_

        # Build response
        response = []
        for prob in prediction:
            prob_dict = {str(k): round(float(v) * 100, 2) for k, v in zip(class_labels, prob)}
            response.append(prob_dict)

        return jsonify({"Prediction": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    api.run(host='0.0.0.0', debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('model/global_model.pkl')

# Initialize scaler with your actual training statistics
scaler = StandardScaler()
scaler.mean_ = np.array([5.8, 120.9, 69.1, 20.5, 79.8, 31.9, 0.47, 33.2])
scaler.scale_ = np.array([3.4, 31.8, 19.3, 15.7, 115.2, 7.8, 0.33, 11.8])

# Store model performance metrics (you can update these from your training)
MODEL_ACCURACY = 0.7677  # Update with your actual global model accuracy
TOTAL_ASSESSMENTS = 12847  # Track total predictions made
FEATURES_COUNT = 8

# Risk thresholds and interpretations
def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Moderate"
    else:
        return "High"

def get_risk_interpretation(prediction, probability):
    """Get detailed interpretation based on prediction and confidence"""
    risk_level = get_risk_level(probability)
    
    if prediction == 1:  # Diabetic
        if probability >= 0.8:
            return {
                'result': 'High Diabetes Risk',
                'description': f'Our model indicates a high probability ({probability:.1%}) of diabetes risk based on your health indicators. Immediate medical consultation is strongly recommended.',
                'recommendations': [
                    'Schedule an immediate appointment with your healthcare provider',
                    'Request comprehensive diabetes screening tests (HbA1c, fasting glucose)',
                    'Begin monitoring blood glucose levels daily',
                    'Adopt a strict low-glycemic diet immediately',
                    'Increase physical activity to at least 200 minutes per week',
                    'Consider working with a certified diabetes educator'
                ]
            }
        elif probability >= 0.5:
            return {
                'result': 'Moderate Diabetes Risk',
                'description': f'Your health data suggests a moderate risk ({probability:.1%}) for diabetes. Preventive measures and medical consultation are recommended.',
                'recommendations': [
                    'Consult with your healthcare provider within 2-4 weeks',
                    'Consider diabetes screening tests',
                    'Monitor blood glucose levels regularly',
                    'Implement a balanced, low-sugar diet',
                    'Increase physical activity to 150 minutes per week',
                    'Focus on weight management if needed'
                ]
            }
    else:  # Non-diabetic
        if probability <= 0.2:
            return {
                'result': 'Low Diabetes Risk',
                'description': f'Your current health indicators suggest a low risk ({(1-probability):.1%} confidence) for diabetes. Continue your healthy lifestyle habits.',
                'recommendations': [
                    'Maintain regular annual health checkups',
                    'Continue current healthy dietary habits',
                    'Keep up with regular physical activity',
                    'Monitor weight and BMI regularly',
                    'Stay informed about diabetes prevention',
                    'Consider periodic glucose screening as preventive care'
                ]
            }
        else:
            return {
                'result': 'Lower Diabetes Risk',
                'description': f'While your risk is currently lower ({(1-probability):.1%} confidence), some health indicators warrant attention and monitoring.',
                'recommendations': [
                    'Schedule regular health checkups every 6-12 months',
                    'Discuss risk factors with your healthcare provider',
                    'Implement or maintain a balanced diet',
                    'Engage in regular moderate physical activity',
                    'Monitor health indicators that may have elevated your risk',
                    'Consider lifestyle modifications for optimal health'
                ]
            }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        features = [float(request.form.get(f)) for f in [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]]
        
        # Scale the input
        input_scaled = scaler.transform([features])
        
        # Get prediction and probability
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Get the probability for the predicted class
        confidence = max(probability) * 100
        
        # Get detailed interpretation
        interpretation = get_risk_interpretation(prediction, max(probability))
        
        # Update assessment counter (in production, use database)
        global TOTAL_ASSESSMENTS
        TOTAL_ASSESSMENTS += 1
        
        # Prepare result data
        result_data = {
            'prediction': prediction,
            'prediction_text': "Diabetic" if prediction == 1 else "Non-Diabetic",
            'confidence': round(confidence, 1),
            'probability_diabetic': round(probability[1] * 100, 1),
            'probability_non_diabetic': round(probability[0] * 100, 1),
            'risk_level': get_risk_level(max(probability)),
            'interpretation': interpretation,
            'model_stats': {
                'accuracy': round(MODEL_ACCURACY * 100, 1),
                'total_assessments': TOTAL_ASSESSMENTS,
                'features_count': FEATURES_COUNT
            },
            'input_data': {
                'pregnancies': features[0],
                'glucose': features[1],
                'blood_pressure': features[2],
                'skin_thickness': features[3],
                'insulin': features[4],
                'bmi': features[5],
                'pedigree_function': features[6],
                'age': features[7]
            }
        }
        
        return render_template('result.html', **result_data)
        
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON responses"""
    try:
        data = request.get_json()
        
        features = [
            data['pregnancies'], data['glucose'], data['blood_pressure'],
            data['skin_thickness'], data['insulin'], data['bmi'],
            data['pedigree_function'], data['age']
        ]
        
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        confidence = max(probability) * 100
        
        interpretation = get_risk_interpretation(prediction, max(probability))
        
        return jsonify({
            'prediction': int(prediction),
            'prediction_text': "Diabetic" if prediction == 1 else "Non-Diabetic",
            'confidence': round(confidence, 1),
            'probability_diabetic': round(probability[1] * 100, 1),
            'probability_non_diabetic': round(probability[0] * 100, 1),
            'risk_level': get_risk_level(max(probability)),
            'interpretation': interpretation,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/stats')
def get_stats():
    """Endpoint to get model statistics"""
    return jsonify({
        'accuracy': round(MODEL_ACCURACY * 100, 1),
        'total_assessments': TOTAL_ASSESSMENTS,
        'features_count': FEATURES_COUNT
    })

if __name__ == '__main__':
    app.run(debug=True)
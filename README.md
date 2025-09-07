# DiabetesFL 🩺

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-orange?logo=flask\&logoColor=white)](https://flask.palletsprojects.com/)

**DiabetesFL** is a **privacy-preserving diabetes risk prediction web app** built using **Federated Learning (FL)**. The system allows multiple clients to train local models on their own data, aggregates these models into a global model without sharing sensitive patient information, and provides accurate predictions with confidence scores and personalized recommendations.


## 🚀 Features

* 🔒 **Privacy-preserving predictions**: Patient data never leaves the client machine.
* 📝 **Interactive web interface** for health parameter input.
* 📊 **Risk visualization**: Risk level (Low / Moderate / High) with confidence bars.
* 📈 **Statistics dashboard**: Displays model accuracy, total assessments, and features used.
* 🌐 **API endpoint**: `/api/predict` for programmatic access.
* 💡 **Personalized recommendations** based on risk level.
* 🔄 **Multiple assessments**: Users can perform repeated assessments and track risk over time.

## Technology Stack

* **Frontend**: HTML, CSS, JavaScript
* **Backend**: Python, Flask
* **Machine Learning**: Scikit-learn (Logistic Regression), NumPy, Pandas
* **Federated Learning**: Custom implementation with federated averaging
* **Model Persistence**: Joblib

## Input Features

| Feature                  | Description                         | Typical Range |
| ------------------------ | ----------------------------------- | ------------- |
| Pregnancies              | Number of times pregnant            | 0 – 17        |
| Glucose                  | Blood glucose concentration (mg/dL) | 50 – 200      |
| BloodPressure            | Diastolic blood pressure (mm Hg)    | 40 – 122      |
| SkinThickness            | Triceps skin fold thickness (mm)    | 10 – 80       |
| Insulin                  | 2-Hour serum insulin (mu U/ml)      | 15 – 400      |
| BMI                      | Body Mass Index (kg/m²)             | 15.0 – 67.1   |
| DiabetesPedigreeFunction | Genetic likelihood of diabetes      | 0.05 – 2.5    |
| Age                      | Age in years                        | 21 – 81       |

Target variable: **Outcome** – 1 (Diabetic), 0 (Non-Diabetic)

## Screenshots

**1. Home / Input Form**
<img width="917" height="932" alt="Screenshot 2025-09-07 at 1 13 40 PM" src="https://github.com/user-attachments/assets/12cf40fb-0100-4fa3-a65e-25fab3ee7e91" />

**2. Risk Assessment Result**
<img width="831" height="932" alt="Screenshot 2025-09-07 at 1 14 19 PM" src="https://github.com/user-attachments/assets/82ccf380-7f5c-4a17-827d-a5ca313e598e" />
<img width="835" height="932" alt="Screenshot 2025-09-07 at 1 15 10 PM" src="https://github.com/user-attachments/assets/c68dc02b-db42-43a7-9e44-946fcc89f508" />
<img width="869" height="932" alt="Screenshot 2025-09-07 at 1 15 40 PM" src="https://github.com/user-attachments/assets/6a466090-7b10-42cc-9d42-d47a504e1e79" />

## Installation

```bash
# Clone repository
git clone https://github.com/SriVarshaCheruku/Diabetes-FL.git
cd Diabetes-FL

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Project Structure

```
Diabetes-FL/
├── app.py                     # Flask backend
├── federated_learning.py      # Federated learning script
├── model/
│   └── global_model.pkl       # Saved global model
├── diabetes.csv               # Dataset
├── templates/
│   ├── index.html             # Input form page
│   └── result.html            # Result page
├── static/
│   └── styles.css             # CSS styles
├── docs/                      # Screenshots and documentation images
├── requirements.txt           # Python dependencies
└── README.md
```

## Federated Learning Workflow

1. Split the dataset into multiple **clients**.
2. Each client trains a **local Logistic Regression model** independently.
3. Local model coefficients and intercepts are **averaged** (Federated Averaging).
4. The **global model** is created from the averaged parameters.
5. The global model is deployed in the web app for **risk prediction**.

✅ Ensures collaborative learning **without sharing raw patient data**.

## API Usage

**POST** `/api/predict`

**Request JSON Example:**

```json
{
  "pregnancies": 2,
  "glucose": 130,
  "blood_pressure": 70,
  "skin_thickness": 25,
  "insulin": 100,
  "bmi": 28.5,
  "pedigree_function": 0.5,
  "age": 45
}
```

**Response JSON Example:**

```json
{
  "prediction": 1,
  "prediction_text": "Diabetic",
  "confidence": 87.5,
  "probability_diabetic": 87.5,
  "risk_level": "High",
  "interpretation": { ... },
  "success": true
}
```

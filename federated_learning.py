import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('diabetes.csv')

# Display shape and first few rows
print("Shape:", df.shape)
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Summary stats
print("\nSummary:\n", df.describe())

# Columns where 0 is invalid
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with median (except Outcome and Pregnancies)
for col in columns_with_zeros:
    median = df[col].median()
    df[col] = df[col].replace(0, median)

print("\nAfter cleaning:")
print(df[columns_with_zeros].describe())

from sklearn.model_selection import train_test_split

# Features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into 3 equal-sized clients
X_temp, X_client3, y_temp, y_client3 = train_test_split(X, y, test_size=0.33, random_state=42)
X_client1, X_client2, y_client1, y_client2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

clients = {
    'client1': (X_client1, y_client1),
    'client2': (X_client2, y_client2),
    'client3': (X_client3, y_client3)
}

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

local_models = {}
local_accuracies = {}
from sklearn.preprocessing import StandardScaler

for client_name, (X_client, y_client) in clients.items():
    # Split client data into train and test sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    # Save model and accuracy
    local_models[client_name] = model
    local_accuracies[client_name] = acc
    
    print(f"{client_name} - Local Accuracy: {acc:.4f}")


import numpy as np
from sklearn.linear_model import LogisticRegression

# Collect coefficients and intercepts from all local models
coef_list = [model.coef_ for model in local_models.values()]
intercept_list = [model.intercept_ for model in local_models.values()]

# Average coefficients and intercepts
avg_coef = np.mean(coef_list, axis=0)
avg_intercept = np.mean(intercept_list, axis=0)

# Initialize a new Logistic Regression model (same params)
global_model = LogisticRegression(max_iter=500)

# Manually assign averaged coefficients and intercepts
global_model.coef_ = avg_coef
global_model.intercept_ = avg_intercept
global_model.classes_ = np.array([0, 1])  # Set classes attribute

print("Federated averaging complete. Global model ready.")

# Prepare combined test data and scale it consistently

# First, combine all test data
X_test_all = pd.concat([train_test_split(clients[c][0], clients[c][1], test_size=0.2, random_state=42)[1] for c in clients])
y_test_all = pd.concat([train_test_split(clients[c][0], clients[c][1], test_size=0.2, random_state=42)[3] for c in clients])

# Scale combined test data using scaler from one client (or fit a new one)
scaler_global = StandardScaler()
X_test_all_scaled = scaler_global.fit_transform(X_test_all)

# Predict using global model
y_pred_global = global_model.predict(X_test_all_scaled)

from sklearn.metrics import accuracy_score
global_acc = accuracy_score(y_test_all, y_pred_global)

print(f"Global Model Accuracy after Federated Averaging: {global_acc:.4f}")

import joblib

# Save the global model
joblib.dump(global_model, 'model/global_model.pkl')

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:41:48 2024

@author: Kevin's Laptop
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

# Convert categorical features to numeric using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# Map income to binary values
data['income'] = data['income_ >50K']
data.drop('income_ >50K', axis=1, inplace=True)

# Split data into features and target
X = data.drop('income', axis=1)
y = data['income']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Logistic Regression Model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
model.fit(X_train, y_train)

# Extract coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# Print the equation
feature_names = X.columns
equation = f"Log-odds = {intercept:.4f} "
for coef, feature in zip(coefficients, feature_names):
    equation += f"+ ({coef:.4f} * {feature}) "
print("Logistic Regression Equation:")
print(equation)
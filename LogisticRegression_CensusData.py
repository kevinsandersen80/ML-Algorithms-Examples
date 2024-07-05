# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:23:12 2024

@author: Kevin's Laptop
"""

#Logistic Regression Census Income example, Classification Report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from IPython.display import display


# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
data.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income']

#Display Dataset
display(data.head())
print('-' * 80)

# Get a concise summary of the dataframe
display(data.info())
print('-' * 80)

# Check for missing values
display(data.isnull().sum())
print('-' * 80)

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

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print('Classification Report:')
print(cr)

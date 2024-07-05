# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:12:44 2024

@author: Kevin Sandersen
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from IPython.display import display

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert the Bunch object to a DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Remove column limit
pd.set_option('display.max_columns', None)

#Display Dataset
display(iris_df)
print('-' * 80)

# Get a concise summary of the dataframe
display(iris_df.info())
print('-' * 80)

display(iris_df.columns.values)
print("-" * 80)

# Check for missing values
display(iris_df.isnull().sum())
print('-' * 80)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report_result)
print("Confusion Matrix:")
print(confusion_matrix_result)




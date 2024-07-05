# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:14:02 2024

@author: Kevin's Laptop
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load a sample dataset
X, y = datasets.make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(logreg_model, 'logreg_model.pkl')

# Make predictions
y_pred = logreg_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Make probability predictions
y_prob = logreg_model.predict_proba(X_test)[:, 1]

# Generate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
roc_display.plot()
plt.show()

# Calculate and print AUC
auc_score = roc_auc_score(y_test, y_prob)
print(f'AUC: {auc_score}')


#Making predictions with Logistic Regression


# Load the trained model from the file
loaded_model = joblib.load('logreg_model.pkl')

# Create new data (replace this with actual new data)
new_data = np.random.rand(10, 20)  # Assuming 20 features as in the original dataset

# Make predictions
predictions = loaded_model.predict(new_data)
print('Predictions:', predictions)

# Make probability predictions
prob_predictions = loaded_model.predict_proba(new_data)
print('Probability Predictions:', prob_predictions)

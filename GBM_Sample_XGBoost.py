# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:12:59 2024

@author: Kevin's Laptop
"""

import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')

# Create a sample dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model to a file
joblib.dump(xgb_model, 'xgb_model.pkl')
print('Model saved to xgb_model.pkl')

# Load the model from the file
loaded_model = joblib.load('xgb_model.pkl')
print(f'Loaded model accuracy: {accuracy_score(y_test, loaded_model.predict(X_test))}')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Make probability predictions
y_prob = xgb_model.predict_proba(X_test)[:, 1]

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


#Making predictions with GBM


# Load the model from the file
model = joblib.load('xgb_model.pkl')

# Create a new set of data (replace this with actual new data)
new_data = np.random.rand(10, 20)  # Assuming 20 features as in the original dataset

# Make predictions
predictions = model.predict(new_data)
print('Predictions:', predictions)

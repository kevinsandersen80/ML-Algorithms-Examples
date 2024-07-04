# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:23:52 2024

@author: Kevin's Laptop
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
import xgboost as xgb

wine_data = load_wine()
print(wine_data.DESCR)
print('-' * 80)

df = pd.DataFrame(wine_data.data)
df.columns = wine_data.feature_names
print('Wine data')
print(df.head())
print('-' * 80)
print('Wine features')
print(df.columns)
print('-' * 80)


df.target = wine_data.target
print('Wine target data')
print(df.target)
print('-' * 80)

# Get a concise summary of the dataframe
print('Summary of wine dataset')
print(df.info())
print('-' * 80)

# Check for missing values
print('Number of missing values:')
print(df.isnull().sum())
print('-' * 80)
print('-' * 80)

correlation_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


#GBM using all variables

X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.Series(wine_data.target)

# Binarize the output for ROC curve (not needed for confusion matrix)
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.5, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a OneVsRestClassifier with GradientBoostingClassifier
classifier = OneVsRestClassifier(make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)))
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for the multiclass problem
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-class using GBM')
plt.legend(loc="lower right")
plt.show()

# Predict the classes for the test set
y_pred = classifier.predict(X_test)
print('-' * 80)
print('-' * 80)

print('Confusion matrix')


X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
y = pd.Series(wine_data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train the GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
classifier.fit(X_train, y_train)

# Predict the classes for the test set
y_pred = classifier.predict(X_test)
print(f'Accuracy with Scikit: {accuracy_score(y_test, y_pred)}')
print('-' * 80)
print('-' * 80)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine_data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Gradient Boosting Classifier')
plt.show()


#Plotting the important features with XGBoost
print('Graph of the important features')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the GBM Model
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy with XGBoost: {accuracy}')
print('-' * 80)

# Compute the confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=wine_data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for XGBoost Classifier')
plt.show()
print('-' * 80)

# Plotting feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model)
plt.show()

# Getting the feature importances as a DataFrame
feature_importance = model.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': feature_importance.keys(),
    'Importance': feature_importance.values()
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Identify top features contributing to 80% of the total importance
importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum() / importance_df['Importance'].sum()
top_features = importance_df[importance_df['Cumulative_Importance'] <= 0.8]

print("\nTop Features contributing to 80% of the importance:")
print(top_features)
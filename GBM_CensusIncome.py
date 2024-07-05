"""
Created on Sun Jun 30 14:44:12 2024

@author: Kevin Sandersen
"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

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

# Convert categorical features to numeric
data = pd.get_dummies(data, drop_first=True)

# Split data into features and target
X = data.drop('income_ >50K', axis=1)
y = data['income_ >50K']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the GBM Model
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

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
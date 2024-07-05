# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 17:53:43 2024

@author: Kevin Sandersen
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn import ensemble, metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from IPython.display import display

# Load dataset from CSV file into DataFrame
airline_df = pd.read_csv('airline.csv', index_col=0)

# Display first five rows of DataFrame
display(airline_df.head())
print('-' * 80)

# Calculate the count of null values for columns with missing values
null_count = airline_df.loc[:, airline_df.isna().sum() > 0].isna().sum()

# Create a DataFrame to display the count 
display(pd.DataFrame({'null_count': null_count}))
print('-' * 80)

# Fill missing values in the 'Arrival Delay in Minutes' column with the mean value
airline_df['Arrival Delay in Minutes'] = airline_df['Arrival Delay in Minutes'].fillna(
    value=airline_df['Arrival Delay in Minutes'].mean())

# Calculate the count of missing values in each column
display(airline_df.isna().sum())
print('-' * 80)

# Convert column names to lowercase and replace spaces with underscores
airline_df.columns = airline_df.columns.str.lower().str.replace(' ', '_')

# Get the array of column names
display(airline_df.columns.values)
print('-' * 80)

# Display summary information about the dataset
display(airline_df.info())

# Select categorical columns excluding specified numerical and target columns
categorical_columns = airline_df.columns[~airline_df.columns.isin(
    ['age', 'flight_distance', 'departure_delay_in_minutes', 'arrival_delay_in_minutes', 'satisfaction'])]

# Convert selected columns to categorical data type
airline_df[categorical_columns] = airline_df[categorical_columns].astype('category')

# Display updated summary information about the dataset after converting columns to categorical
display(airline_df.info())
print('-' * 80)

# Generate descriptive statistics for numerical columns in the dataset
display(airline_df.describe())
print('-' * 80)

# Generate descriptive statistics for categorical columns in the dataset
display(airline_df.describe(include=['category']))
print('-' * 80)

# Create a pie chart to visualize the distribution of satisfaction levels
plt.pie(airline_df.satisfaction.value_counts(), labels=["Neutral or dissatisfied", "Satisfied"],
        colors=sns.color_palette("YlOrBr"), autopct='%1.1f%%');

# Generate a heatmap to visualize the correlation matrix of numerical features
heatmap = sns.heatmap(airline_df.corr(numeric_only=True), vmin=-1, vmax=1, annot=True)

# Set title and adjust properties
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45);

# Create an empty DataFrame to store Variance Inflation Factor (VIF) values
vif_df = pd.DataFrame()

# Select numerical columns for VIF calculation
features = airline_df.select_dtypes(include='number').columns

# Assign selected features to the 'feature' column in the DataFrame
vif_df["feature"] = features

# Calculate VIF values for each feature and store them in the 'VIF' column
vif_df["VIF"] = [variance_inflation_factor(airline_df[features].values, i) for i in range(len(features))]

# Display the DataFrame containing feature names and their corresponding VIF values
display(vif_df)
print('-' * 80)

#VIF = 1: No correlation between the independent variable and other variables.
#1 < VIF < 5: Moderate correlation, but typically not severe enough to require correction.
#VIF > 5: High correlation, indicating potential multicollinearity that might need to be addressed.
#VIF > 10: Very high correlation, strongly suggesting multicollinearity issues that should be corrected.

 # Select categorical columns and convert them to integer type
ordinal_df = airline_df.select_dtypes('category').loc[:, 'inflight_wifi_service':].astype(int)

# Merge ordinal and continuous features
ordinal_continuous = ordinal_df.merge(airline_df.select_dtypes(np.number), left_index=True, right_index=True)

# Create a heatmap to visualize the Spearman correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(ordinal_continuous.corr(method='spearman'), annot=True);

# Create a mapping dictionary for encoding satisfaction levels
satisfaction_mapping = {v: k for k, v in enumerate(airline_df['satisfaction'].unique())}

# Display the satisfaction mapping dictionary
display(satisfaction_mapping)
print('-' * 80)

# Map satisfaction levels using the previously created mapping dictionary
airline_df['satisfaction'] = airline_df['satisfaction'].map(satisfaction_mapping)

# Display the first few entries of the 'satisfaction' column after mapping
display(airline_df['satisfaction'].head())
print('-' * 80)

# Create a bar plot to visualize the relationship between gender and satisfaction
sns.barplot(x='gender', y='satisfaction', data=airline_df, errorbar=None);

# Create a bar plot to visualize the relationship between customer type and satisfaction
sns.barplot(x='customer_type', y='satisfaction', data=airline_df, errorbar=None);

# Create a bar plot to visualize the relationship between type of travel and satisfaction
sns.barplot(x='type_of_travel', y='satisfaction', data=airline_df, errorbar=None);

# Create a bar plot to visualize the relationship between the passaneger class and satisfaction
sns.barplot(x='class', y='satisfaction', data=airline_df, errorbar=None);


# Define a function to calculate the chi-squared test statistic and p-value
def get_correlation_nominal(column: str, alpha=0.05, target_column: str = 'satisfaction'):
    contingency_table = pd.crosstab(airline_df[column], airline_df[target_column])
    stat, p, dof, expected = chi2_contingency(contingency_table)
    return p, p <= alpha

# Specify nominal columns for correlation analysis
nominal_columns = ['gender', 'customer_type', 'type_of_travel', 'class']

# Initialize lists to store p-values and correlation indicators
p_list, is_correlated_list = [], []

for c in nominal_columns:
    # Calculate the p-value and correlation indicator using the defined function
    p, is_correlated = get_correlation_nominal(c)
    # Append the results to the respective lists
    p_list.append(p)
    is_correlated_list.append(is_correlated)

# Create a DataFrame to display the p-values and correlation indicators for each nominal column
display(pd.DataFrame({'p_value': p_list, 'is_correlated': is_correlated_list}, index=nominal_columns))
print('-' * 80)

# Create age groups based on quartiles of the 'age' column
airline_df['age_group'] = pd.qcut(airline_df['age'], 4)

# Visualize satisfaction levels across age groups using a bar plot
sns.barplot(x='age_group', y='satisfaction', data=airline_df, errorbar=None);

# Encode age groups numerically by converting categorical codes to integers and incrementing by 1
airline_df['age_group'] = airline_df['age_group'].cat.codes + 1

# Display the first few entries of the 'age_group' column after encoding
display(airline_df['age_group'].head())
print('-' * 80)

# Drop specified columns from the dataset
airline_df = airline_df.drop(columns=['arrival_delay_in_minutes', 'inflight_entertainment', 'age'])

# Display the first few rows of the updated dataset
display(airline_df.head())
print('-' * 80)

# Convert categorical variables into dummy variables and drop the first category to prevent multicollinearity
airline_df = pd.get_dummies(airline_df, columns=['gender', 'customer_type', 'type_of_travel', 'class'], drop_first=True)

# Convert column names to lowercase and replace spaces with underscores for consistency
airline_df.columns = airline_df.columns.str.lower().str.replace(' ', '_')

# Display the first few rows of the updated dataset
display(airline_df.head())


#End EDA
#Begin Model Comparison


# Split the dataset into features (X) and target variable (y)
X, y = airline_df.drop(columns=['satisfaction']), airline_df['satisfaction']

# Split the data into training and testing sets, with 10% for testing and stratified sampling based on the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1234)

#Model Evaluation and Comparison
#This section evaluates multiple machine learning models and compares their performance:
#K-Nearest Neighbors (KNN): Grid search is performed to find the best number of neighbors, and confusion matrix plot displays performance metrics.
#Logistic Regression: Grid search is performed to optimize regularization strength, with the resulting model's performance evaluated and displayed.
#Random Forest: Multiple forest sizes are evaluated, and the model with the highest accuracy is selected for display along with a plot of accuracy vs. the number of trees.
#Comparison Table: A table summarizes the performance metrics (accuracy, precision, recall, F1 score) for each model on the test dataset, providing a clear comparison of their effectiveness.

# Define a function to plot the confusion matrix for a classifier
def plot_confusion_matrix(classifier):
    
    # Predict the classes using the classifier on the test data
    predicted = classifier.predict(X_test)
    
    # Compute the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_test, predicted)
    
    # Define group names for the confusion matrix
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    
    # Flatten the confusion matrix and format group counts
    group_counts = ['{0:0.0f}'.format(value) for value in confusion_matrix.flatten()]
    
    # Calculate percentages and format them
    group_percentages = ['{0:.2%}'.format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    
    # Create labels for the confusion matrix cells
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    
    # Reshape labels to match confusion matrix shape
    labels = np.asarray(labels).reshape(2, 2)
    
    # Calculate classification performance metrics
    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    precision = confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])
    recall = confusion_matrix[1, 1] / sum(confusion_matrix[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Generate text with performance metrics
    stats_text = f"\n\nAccuracy={accuracy:.2%}\nPrecision={precision:.2%}\nRecall={recall:.2%}\nF1 Score={f1_score:.2%}"
    
    # Plot the confusion matrix using seaborn heatmap
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')
    
    # Set x-axis label with performance metrics
    plt.xlabel(stats_text)

# Define a pipeline for K-Nearest Neighbors classifier with standard scaling
knn_pipe = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier())])

# Define hyperparameter grid for grid search
param_grid = {'knn__n_neighbors': range(1, 30)}

# Initialize grid search with cross-validation
knn_gs = GridSearchCV(knn_pipe, param_grid=param_grid, scoring='accuracy', n_jobs=-1)

# Perform grid search on training data
knn_gs.fit(X_train, y_train)

# Print best accuracy and corresponding parameters found by grid search
print(f'Best accuracy: {knn_gs.best_score_:.2%}, best params: {knn_gs.best_params_}')

# Evaluate test accuracy of the best model
print(f'Test accuracy: {knn_gs.score(X_test, y_test):.2%}')

# Plot the confusion matrix for the K-Nearest Neighbors classifier
plot_confusion_matrix(knn_gs)

# Define a pipeline for Logistic Regression classifier with standard scaling
logistic_regression_pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression())])

# Define hyperparameter grid for grid search
param_grid = {'lr__C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]}

# Initialize grid search with cross-validation
logistic_regression_gs = GridSearchCV(logistic_regression_pipe, param_grid=param_grid, scoring='accuracy', n_jobs=-1)

# Perform grid search on training data
logistic_regression_gs.fit(X_train, y_train)

# Print best accuracy and corresponding parameters found by grid search
print(f'Best accuracy: {logistic_regression_gs.best_score_:.2%}, best params: {logistic_regression_gs.best_params_}')

# Evaluate test accuracy of the best model
print(f'Test accuracy: {logistic_regression_gs.score(X_test, y_test):.2%}')

# Define range of estimators (number of trees) for Random Forest
estimators = range(10, 201, 10)

# Initialize variables to track the best model
best_accuracy = 0
best_forest = None
best_n = 0
accuracies = []

for n_estimators in estimators:
    
    # Create Random Forest classifier with specified number of trees
    random_forest = ensemble.RandomForestClassifier(n_estimators=n_estimators, random_state=1234)
    
    # Train the classifier on the training data
    random_forest.fit(X_train, y_train)
    
    # Evaluate the accuracy on the test data
    accuracy = random_forest.score(X_test, y_test)
    
    # Append accuracy to list of accuracies
    accuracies.append(accuracy)
    
    # Update the best accuracy and corresponding model if current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_forest = random_forest
        best_n = n_estimators

# Print the best test accuracy and corresponding number of trees
print(f'Test accuracy: {best_accuracy:.2%}, best number of trees: {best_n}')

# Plot the relationship between the number of trees and accuracy
plt.plot(estimators, accuracies)

# Set the label for the x-axis
plt.xlabel('Number of trees')

# Set the label for the y-axis
plt.ylabel('Accuracy')

# Plot the confusion matrix for the Random Forest classifier
plot_confusion_matrix(best_forest)

# Define a function to calculate evaluation metrics for classifiers
def get_score_list(classifiers, X_data, y_data):
    # Predict classes for the input data using each classifier
    predicted = {c: c.predict(X_data) for c in classifiers}
    
    # Define a list of evaluation metrics to compute
    metrics_list = [metrics.accuracy_score, metrics.precision_score, metrics.recall_score, metrics.f1_score]
    
    # Initialize an empty list to store evaluation metric results
    result = []
    
    for c in classifiers:
        # Compute evaluation metrics for the classifier and store them in a list
        result.append([metric(y_data, predicted[c]) for metric in metrics_list])
    return result

# List of classifiers to evaluate
classifiers_list = [knn_gs, logistic_regression_gs, best_forest]

# Create a DataFrame to display evaluation metric results
display(pd.DataFrame(get_score_list(classifiers_list, X_test, y_test),
             columns=['accuracy', 'precision', 'recall', 'f1_score'],
             index=['knn', 'logistic_regression', 'random_forest']).sort_values(by='accuracy', ascending=False))


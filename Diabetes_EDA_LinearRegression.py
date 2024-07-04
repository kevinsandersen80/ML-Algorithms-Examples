# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:45:02 2024

@author: Kevin's Laptop
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


diabetes_data = load_diabetes()
print(diabetes_data.DESCR)
print('-' * 80)

df = pd.DataFrame(diabetes_data.data)
df.columns = diabetes_data.feature_names
print('Diabetes features')
print(df.head())
print('-' * 80)

df.target = diabetes_data.target
print('Diabetes target data')
print(df.target)
print('-' * 80)

# Get a concise summary of the dataframe
print('Summary of diabetes dataset')
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


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df,df.target, test_size=0.2, random_state=42)

#Linear Regression with all features
print('Linear regression with all features:')

# Create a Linear Regression model
linreg_model = LinearRegression()

# Train the model using the training sets
linreg_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = linreg_model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
print('-' * 80)

#Linear regression equation: 

# Print the coefficients and intercept
print("Coefficients:", linreg_model.coef_)
print("Intercept:", linreg_model.intercept_)

equation = f'y = {linreg_model.intercept_:.2f} '
for i, coef in enumerate(linreg_model.coef_):
    equation += f'+ {coef:.2f} * X[{i}] '
print("Regression Equation:", equation)
print('-' * 80)
print('-' * 80)

#Stepwise model selection
print('Stepwise model selection:')

# Function for stepwise selection using AIC
def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

# Perform stepwise selection
selected_features = stepwise_selection(X_train, y_train)

# Print selected features
print("Selected features:", selected_features)

# Fit linear regression model with selected features
final_model = sm.OLS(y_train, sm.add_constant(X_train[selected_features])).fit()

# Print summary of the model
print(final_model.summary())

# Predict on test set
y_pred = final_model.predict(sm.add_constant(X_test[selected_features]))

# Evaluate the model


print("\nMean squared error on test set:", mean_squared_error(y_test, y_pred))
print("R-squared on test set:", r2_score(y_test, y_pred))
print('-' * 80)
print('-' * 80)


#Check for multicollinearity using variance inflation factor (VIF)
def calculate_vif(df):
    # Initialize VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    
    return vif_data

vif_df = calculate_vif(df)
print(vif_df)
print('-' * 80)

#Remove only s1
df_new = df.drop(['s1'], axis = 1)
print(df_new.head())
print('-' * 80)

#Check for multicollinearity using variance inflation factor (VIF)
def calculate_vif(df_new):
    # Initialize VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_new.columns
    
    # Calculate VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df_new.values, i) for i in range(len(df_new.columns))]
    
    return vif_data

vif_df_new = calculate_vif(df_new)
print(vif_df_new)
print('-' * 80)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_new,df.target, test_size=0.2, random_state=42)

#Linear Regression with all features
print('Linear regression with reduced features after VIF analysis:')

# Create a Linear Regression model
linreg_model = LinearRegression()

# Train the model using the training sets
linreg_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = linreg_model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
print('-' * 80)

#Linear regression equation: 

# Print the coefficients and intercept
print("Coefficients:", linreg_model.coef_)
print("Intercept:", linreg_model.intercept_)

equation = f'y = {linreg_model.intercept_:.2f} '
for i, coef in enumerate(linreg_model.coef_):
    equation += f'+ {coef:.2f} * X[{i}] '
print("Regression Equation:", equation)
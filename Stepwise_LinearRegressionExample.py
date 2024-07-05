# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:10:32 2024

@author: Kevin Sandersen
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# Convert to pandas DataFrame
df = pd.DataFrame(X, columns=[f'X{i}' for i in range(1, 11)])
df['target'] = y

print(df.head())

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

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
from sklearn.metrics import mean_squared_error, r2_score

print("\nMean squared error on test set:", mean_squared_error(y_test, y_pred))
print("R-squared on test set:", r2_score(y_test, y_pred))

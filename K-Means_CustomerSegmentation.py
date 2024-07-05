# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:44:12 2024

@author: Kevin Sandersen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Load the dataset
df = pd.read_csv('customer_segmentation.csv')

# Remove column limit
pd.set_option('display.max_columns', None)

# Display the first few rows of the dataframe
display(df.head())
print('-' * 80)

# Get a concise summary of the dataframe
display(df.info())
print('-' * 80)

# Check for missing values
display(df.isnull().sum())
print('-' * 80)

# Get descriptive statistics
display(df.describe())
print('-' * 80)


# Define the list of continuous and categorical data types
continuous_data_types = ['int64', 'float64']
categorical_data_types = ['object', 'category', 'bool']

# Determine the total number of plots needed
total_plots = df.shape[1]

# Set the number of columns for the subplot grid
num_columns = 4

# Calculate the number of rows needed for the subplot grid
num_rows = math.ceil(total_plots / num_columns)

# Create the subplot grid
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 5))
fig.tight_layout(pad=4.0)

# Flatten the axes array for easy iteration
axes_flat = axes.flatten()

# Initialize a counter for the subplot index
subplot_index = 0

for column in df.columns:
    # Determine the data type of the column
    if df[column].dtype in continuous_data_types:
        # Plot for continuous variables
        sns.histplot(df[column].dropna(), bins=30, kde=True, ax=axes_flat[subplot_index])  # Using dropna() to exclude missing values
        axes_flat[subplot_index].set_title(f'{column} Distribution')
        axes_flat[subplot_index].set_xlabel(column)
        axes_flat[subplot_index].set_ylabel('Frequency')
    elif df[column].dtype in categorical_data_types:
        # Plot for categorical variables
        sns.countplot(x=column, data=df, ax=axes_flat[subplot_index])
        axes_flat[subplot_index].set_title(f'{column} Distribution')
        axes_flat[subplot_index].set_xlabel(column)
        axes_flat[subplot_index].set_ylabel('Count')
        axes_flat[subplot_index].tick_params(axis='x', rotation=45)  # Rotate labels to prevent overlap
    subplot_index += 1

# Hide any unused subplot axes
for i in range(subplot_index, len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.show()
print('-' * 80)


# Scatter plot example using Year_Birth vs. Income
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year_Birth', y='Income', data=df)
plt.title('Year of Birth vs. Income')
plt.xlabel('Year of Birth')
plt.ylabel('Income')
plt.show()

# Box plot example using Education as a categorical variable and Income as a continuous variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='Education', y='Income', data=df)
plt.title('Income by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Income')
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.show()


# Select only numeric columns for the correlation matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix for numeric columns only
corr_matrix = numeric_df.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(16, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
print('-' * 80)


# Example using IQR for the 'MntFruits' variable
Q1 = df['MntFruits'].quantile(0.25)
Q3 = df['MntFruits'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['MntFruits'] < lower_bound) | (df['MntFruits'] > upper_bound)]
display(outliers)
print('-' * 80)

#End Exploratory Data Analysis
#Begin K-Means Clustering

# Assuming there's an arbitrary numerical column we can use for "target" encoding
# If the dataset lacks a clear target, choose a relevant numerical column
# For illustration, let's assume 'Income' is the column we use for encoding
numerical_column = 'Income'

# Impute missing values in the numerical column used for encoding
imputer = SimpleImputer(strategy='median')
df[numerical_column] = imputer.fit_transform(df[[numerical_column]])

# Perform mean encoding on categorical columns
for col in df.select_dtypes(include=['object']).columns:
    mean_encoded = df.groupby(col)[numerical_column].transform('mean')
    df[col + '_mean_encoded'] = mean_encoded

# Select encoded and numerical columns for clustering
encoded_cols = [col for col in df.columns if 'mean_encoded' in col]
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
features = df[encoded_cols + list(numeric_cols)]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Function to perform K-Means clustering and visualize with PCA
def kmeans_and_visualize_with_pca(data, n_clusters_list):
    for n_clusters in n_clusters_list:
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', marker='x', edgecolor='k', s=50, alpha=0.6)
        plt.title(f'K-Means with {n_clusters} clusters visualized with PCA')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')
        plt.show()

# Define the list of cluster numbers to try
n_clusters_list = [2, 3, 4, 5]

# Perform K-Means clustering and visualize with PCA
kmeans_and_visualize_with_pca(features_scaled, n_clusters_list)

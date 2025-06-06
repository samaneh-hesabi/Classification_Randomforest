import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('figures/eda', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Load the dataset
data_path = os.path.join('data', 'winequality-red.csv')
df = pd.read_csv(data_path, sep=';')

# Basic dataset information
print("=== Wine Quality Dataset Overview ===")
print(f"Original dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Clean data (remove duplicates)
df_clean = df.drop_duplicates()
print("\nShape after removing duplicates:", df_clean.shape)
print(f"Removed {len(df) - len(df_clean)} duplicate rows")

# Handle outliers using IQR method and replace with median
def handle_outliers(df):
    df_no_outliers = df.copy()
    
    print("\nHandling outliers:")
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        if outliers > 0:
            # Replace outliers with median
            median_value = df[column].median()
            df_no_outliers.loc[df[column] < lower_bound, column] = median_value
            df_no_outliers.loc[df[column] > upper_bound, column] = median_value
            
            print(f"  - {column}: {outliers} outliers replaced with median ({median_value:.2f})")
    
    return df_no_outliers

# Apply outlier handling
df_clean = handle_outliers(df_clean)

print("\nBasic statistics after cleaning:")
print(df_clean.describe())

# Distribution of wine quality
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df_clean)
plt.title('Distribution of Wine Quality')
plt.savefig('figures/eda/quality_distribution.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation = df_clean.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/eda/correlation_matrix.png')
plt.close()

# Correlation with quality
quality_correlation = correlation['quality'].sort_values(ascending=False)
print("\nCorrelation with quality:")
print(quality_correlation)

plt.figure(figsize=(10, 6))
quality_correlation.drop('quality').plot(kind='bar')
plt.title('Features Correlation with Wine Quality')
plt.tight_layout()
plt.savefig('figures/eda/quality_correlation.png')
plt.close()

# Histograms of all features
df_clean.hist(bins=20, figsize=(15, 10))
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('figures/eda/feature_histograms.png')
plt.close()

# Box plots for each feature by quality
features = df_clean.columns.drop('quality')
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='quality', y=feature, data=df_clean)
    plt.title(f'{feature} by Wine Quality')
    plt.tight_layout()
    plt.savefig(f'figures/eda/boxplot_{feature.replace(" ", "_")}.png')
    plt.close()

# Pairplot of selected features
selected_features = ['alcohol', 'volatile acidity', 'sulphates', 'quality']
plt.figure(figsize=(12, 10))
sns.pairplot(df_clean[selected_features], hue='quality', palette='viridis')
plt.savefig('figures/eda/pairplot_selected.png')
plt.close()

# Check for outliers after cleaning
plt.figure(figsize=(15, 10))
df_clean.boxplot(figsize=(15, 10))
plt.title('Boxplots After Outlier Handling')
plt.tight_layout()
plt.savefig('figures/eda/outliers_handled_boxplot.png')
plt.close()

# Save the cleaned data
clean_data_path = os.path.join('data', 'processed', 'wine_clean.csv')
df_clean.to_csv(clean_data_path, index=False)
print(f"\nCleaned data saved to: {clean_data_path}")

print("\nEDA completed. All visualizations saved in 'figures/eda' directory.") 
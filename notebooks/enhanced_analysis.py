import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import os

def check_figure_exists(figure_name):
    """Check if a figure already exists in the figures directory."""
    figures_dir = Path('figures')
    return (figures_dir / figure_name).exists()

def load_and_clean_data(file_path):
    """Load the dataset and remove duplicates."""
    # Load data
    df = pd.read_csv(file_path, sep=';')
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    print("\n=== Duplicate Removal ===")
    print(f"Original dataset size: {len(df)}")
    print(f"Clean dataset size: {len(df_clean)}")
    print(f"Removed {len(df) - len(df_clean)} duplicate rows")
    
    return df_clean

def handle_outliers(df, method='iqr'):
    """Handle outliers using either IQR or Z-score method."""
    df_no_outliers = df.copy()
    
    print("\n=== Outlier Handling ===")
    print(f"Method: {method.upper()}")
    
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
            # Replace outliers with bounds
            df_no_outliers.loc[df[column] < lower_bound, column] = lower_bound
            df_no_outliers.loc[df[column] > upper_bound, column] = upper_bound
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            outliers = (z_scores > 3).sum()
            
            # Replace outliers with mean
            df_no_outliers.loc[z_scores > 3, column] = df[column].mean()
        
        print(f"Handled {outliers} outliers in {column}")
    
    return df_no_outliers

def analyze_features(df):
    """Perform detailed analysis of features."""
    print("\n=== Feature Analysis ===")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Correlation analysis
    print("\nCorrelation with Quality:")
    correlations = df.corr()['quality'].sort_values(ascending=False)
    print(correlations)
    
    # Feature importance based on correlation with quality
    if not check_figure_exists('feature_importance.png'):
        plt.figure(figsize=(12, 6))
        correlations.drop('quality').plot(kind='bar')
        plt.title('Feature Correlation with Wine Quality')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png')
        plt.close()
        print("Generated feature_importance.png")

def create_visualizations(df):
    """Create various visualizations for the dataset."""
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Distribution of wine quality
    if not check_figure_exists('quality_distribution.png'):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='quality')
        plt.title('Distribution of Wine Quality')
        plt.savefig('figures/quality_distribution.png')
        plt.close()
        print("Generated quality_distribution.png")
    
    # 2. Feature distributions
    if not check_figure_exists('feature_distributions.png'):
        plt.figure(figsize=(15, 10))
        df.hist(bins=30, figsize=(15, 10))
        plt.suptitle('Distribution of Wine Features')
        plt.tight_layout()
        plt.savefig('figures/feature_distributions.png')
        plt.close()
        print("Generated feature_distributions.png")
    
    # 3. Correlation heatmap
    if not check_figure_exists('correlation_heatmap.png'):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('figures/correlation_heatmap.png')
        plt.close()
        print("Generated correlation_heatmap.png")
    
    # 4. Box plots for each feature by quality
    features = df.columns.drop('quality')
    for feature in features:
        figure_name = f'boxplot_{feature}.png'
        if not check_figure_exists(figure_name):
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='quality', y=feature)
            plt.title(f'{feature} by Quality')
            plt.savefig(f'figures/{figure_name}')
            plt.close()
            print(f"Generated {figure_name}")

def save_processed_data(df):
    """Save the processed dataset."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    df.to_csv('data/wine_clean.csv', index=False)
    print("\n=== Saved Processed Data ===")
    print("Clean data (without duplicates): data/wine_clean.csv")

def main():
    # Load and clean data
    data_path = Path('data/winequality-red.csv')
    df_clean = load_and_clean_data(data_path)
    
    # Handle outliers
    df_no_outliers = handle_outliers(df_clean, method='iqr')
    
    # Analyze features
    analyze_features(df_clean)
    
    # Create visualizations
    create_visualizations(df_clean)
    
    # Save processed dataset
    save_processed_data(df_clean)
    
    print("\n=== Analysis Complete ===")
    print("All figures are stored in the 'figures' directory")

if __name__ == "__main__":
    main() 
<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Source Code Directory</div>

# 1. Contents
This directory contains the main scripts for the wine quality classification project.

## 1.1 Files
- `simple_wine_classification.py`: The main script that performs the entire classification process.
- `wine_eda.py`: Exploratory Data Analysis script that analyzes the dataset and generates visualizations.

# 2. Script Description
The `simple_wine_classification.py` script performs the following operations:

## 2.1 Data Loading and Preprocessing
- Loads the wine dataset from CSV
- Removes duplicate entries
- Handles outliers using IQR method and replaces them with median values
- Standardizes features using StandardScaler

## 2.2 Model Training
- Splits data into training and test sets
- Creates and trains a Random Forest classifier
- Uses 100 trees with a maximum depth of 8

## 2.3 Evaluation and Visualization
- Calculates accuracy and classification metrics
- Identifies and ranks feature importance
- Generates visualizations:
  - Feature importance bar chart
  - Confusion matrix heatmap

# 3. Usage
Run the EDA script from the project root directory:
```bash
python src/wine_eda.py
```

Run the classification script from the project root directory:
```bash
python src/simple_wine_classification.py
``` <div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Wine Quality EDA (Exploratory Data Analysis)</div>

# 1. Overview
This document explains the Exploratory Data Analysis (EDA) performed on the wine quality dataset.

# 2. EDA Process
The EDA script (`wine_eda.py`) performs the following analyses:

## 2.1 Basic Data Exploration
- Dataset shape and structure
- Summary statistics
- Missing value detection
- Duplicate row identification

## 2.2 Distribution Analysis
- Distribution of wine quality scores
- Histograms of all features
- Boxplots to identify outliers

## 2.3 Correlation Analysis
- Correlation matrix of all features
- Feature correlations with wine quality
- Pairplot of selected important features

# 3. Generated Visualizations
All visualizations are saved in the `figures/eda/` directory:

## 3.1 Distribution Plots
- `quality_distribution.png`: Bar chart showing distribution of wine quality scores
- `feature_histograms.png`: Histograms of all features

## 3.2 Correlation Plots
- `correlation_matrix.png`: Heatmap showing correlations between all features
- `quality_correlation.png`: Bar chart showing each feature's correlation with quality

## 3.3 Feature Analysis by Quality
- `boxplot_*.png`: Box plots showing distribution of each feature by quality score
- `pairplot_selected.png`: Pairplot showing relationships between selected features

## 3.4 Outlier Detection
- `outliers_boxplot.png`: Boxplots to identify outliers in each feature

# 4. Key Insights
From the EDA, we can observe:
- The distribution of wine quality scores
- Which features have the strongest correlation with wine quality
- How feature distributions vary across different quality levels
- Potential outliers that might affect model performance

# 5. Usage
Run the EDA script from the project root directory:
```bash
python src/wine_eda.py
``` 
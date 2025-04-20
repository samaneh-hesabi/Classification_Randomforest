<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Wine Quality Classification Project</div>

This project implements a Random Forest Classifier to predict wine quality based on various chemical properties. The implementation includes comprehensive data preprocessing, model training, and evaluation metrics.

# 1. Contents

## 1.1 Python Files
- `ddd.py`: Main implementation file containing:
  - Data preprocessing and standardization
  - Random Forest Classifier implementation
  - SMOTE for handling class imbalance
  - Model evaluation and visualization
  - Feature importance analysis
  - ROC curve analysis

# 2. Project Overview

## 2.1 Data Processing
- Loads wine quality data from `data/wine_clean.csv`
- Separates features and target variable
- Standardizes features using StandardScaler
- Splits data into training and testing sets (80/20 split)
- Handles class imbalance using SMOTE

## 2.2 Model Implementation
- Uses Random Forest Classifier with optimized parameters:
  - 200 estimators
  - Maximum depth of 10
  - Minimum samples split of 5
  - Minimum samples leaf of 2
  - Balanced class weights

## 2.3 Evaluation Metrics
- Accuracy score
- Classification report
- 5-fold cross-validation
- Confusion matrix
- Feature importance analysis
- ROC curves for each class
- Class distribution visualization

# 3. Usage

## 3.1 Prerequisites
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - matplotlib
  - seaborn

## 3.2 Running the Code
1. Ensure all dependencies are installed
2. Place the wine dataset in `data/wine_clean.csv`
3. Run `ddd.py` to:
   - Train the model
   - Generate evaluation metrics
   - Create visualizations in the `figures` directory

## 3.3 Output
The script generates several visualizations in the `figures` directory:
- `confusion_matrix.png`: Model's confusion matrix
- `feature_importance.png`: Feature importance plot
- `roc_curves.png`: ROC curves for each class
- `class_distribution.png`: Class distribution before and after SMOTE

# 4. Results
The model provides:
- Classification accuracy
- Cross-validation scores
- Feature importance rankings
- ROC curves for each quality class
- Visual analysis of class distribution 
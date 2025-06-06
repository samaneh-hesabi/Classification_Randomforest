<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Wine Quality Classification using Random Forest</div>

# 1. Project Overview
This project classifies wine quality based on chemical properties using Random Forest classification. The implementation is designed to be simple and easy to understand.

# 2. Dataset Description
The dataset contains 1599 samples of red wine with 11 features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

The target variable is wine quality, scored between 3 and 8.

# 3. Project Structure
```
Classification_Randomforest/
├── data/                    
│   └── winequality-red.csv  # Original dataset
├── figures/                 # Generated visualizations
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── eda/                 # Exploratory Data Analysis visualizations
├── src/                     
│   ├── simple_wine_classification.py # Main classification script
│   └── wine_eda.py          # Exploratory Data Analysis script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

# 4. Dependencies
Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

Install with:
```bash
pip install -r requirements.txt
```

# 5. Usage
Run the EDA script to explore the dataset:
```bash
python src/wine_eda.py
```

Run the classification script:
```bash
python src/simple_wine_classification.py
```

# 6. Implementation Details
The implementation follows these steps:

## 6.1 Data Preprocessing
- Load the wine dataset
- Remove duplicate entries
- Handle outliers by replacing them with median values
- Standardize features

## 6.2 Model Training
- Split data into training (80%) and testing (20%) sets
- Train a Random Forest classifier with 100 trees

## 6.3 Evaluation
- Calculate accuracy and classification metrics
- Generate feature importance ranking
- Create visualizations

# 7. Results
The script produces:
- Model accuracy score
- Detailed classification report
- Feature importance ranking
- Visualizations saved to the figures directory

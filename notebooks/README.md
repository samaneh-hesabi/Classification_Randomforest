<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Notebooks Directory</div>

This directory contains Jupyter notebooks and Python scripts used for data analysis and model development.

# 1. Contents

## 1.1 Files
- `enhanced_analysis.py`: A Python script containing enhanced data analysis functions and visualizations for the classification project.

# 2. Purpose
The notebooks in this directory are used for:
- Data exploration and analysis
- Model development and testing
- Visualization of results
- Experimentation with different approaches

# 3. Dataset Description
The project uses wine quality datasets containing the following features:

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| fixed acidity | Tartaric acid content | g/dm³ | 4.6 - 15.9 |
| volatile acidity | Acetic acid content | g/dm³ | 0.12 - 1.58 |
| citric acid | Citric acid content | g/dm³ | 0.0 - 1.0 |
| residual sugar | Sugar remaining after fermentation | g/dm³ | 0.9 - 15.5 |
| chlorides | Salt content | g/dm³ | 0.012 - 0.611 |
| free sulfur dioxide | Free SO₂ content | mg/dm³ | 1 - 72 |
| total sulfur dioxide | Total SO₂ content | mg/dm³ | 6 - 289 |
| density | Density of wine | g/cm³ | 0.990 - 1.004 |
| pH | Acidity/basicity measure | - | 2.74 - 4.01 |
| sulphates | Potassium sulphate content | g/dm³ | 0.33 - 2.0 |
| alcohol | Alcohol content | % vol | 8.4 - 14.9 |
| quality | Wine quality rating | score | 3 - 9 |

The dataset contains two files:
- `wine_clean.csv`: Cleaned and processed wine quality data
- `winequality-red.csv`: Original red wine quality data

# 4. Analysis Methods
The analysis includes the following key components:

## 4.1 Data Preprocessing
- Duplicate removal
- Outlier handling using IQR and Z-score methods
- Data cleaning and standardization

## 4.2 Feature Analysis
- Statistical analysis of each feature
- Correlation analysis between features
- Feature importance assessment

## 4.3 Visualization
- Distribution plots
- Correlation heatmaps
- Feature relationship plots
- Quality score distribution

# 5. Project Structure
```
project_root/
├── data/                  # Data files
│   ├── wine_clean.csv    # Processed dataset
│   └── winequality-red.csv # Original dataset
├── notebooks/            # Analysis scripts
│   └── enhanced_analysis.py
└── figures/             # Generated visualizations
```

# 6. Usage
To use the notebooks:
1. Ensure you have Jupyter installed
2. Navigate to this directory
3. Run `jupyter notebook` to start the Jupyter server
4. Open the desired notebook or script

## 6.1 Running the Analysis
To run the enhanced analysis:
```bash
python enhanced_analysis.py
```

# 7. Dependencies
The notebooks require the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

# 8. Contributing
When contributing to this project:
1. Create a new branch for your feature
2. Follow the existing code style
3. Add appropriate documentation
4. Include tests for new functionality
5. Submit a pull request with a clear description

# 9. Best Practices
- Always backup your work before making major changes
- Document any modifications to the analysis pipeline
- Keep visualizations consistent with the project style
- Use meaningful variable names and comments
- Follow PEP 8 style guidelines for Python code 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories if they don't exist
os.makedirs('figures', exist_ok=True)

# Load the cleaned dataset
clean_data_path = os.path.join('data', 'processed', 'wine_clean.csv')

# Check if cleaned data exists, if not, prompt to run the EDA script first
if not os.path.exists(clean_data_path):
    print("Cleaned data not found. Please run the EDA script first:")
    print("python src/wine_eda.py")
    exit(1)

# Load the cleaned data
df = pd.read_csv(clean_data_path)

# Basic data exploration
print("Dataset shape:", df.shape)
print("\nFirst 1 rows:")
print(df.head(1))

# Separate features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=8,       # Maximum depth of trees
    random_state=42    # For reproducibility
)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

# Handle zero_division in classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Create a more readable confusion matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(cm)

# Calculate class-wise accuracy
classes = np.unique(y)
class_accuracy = pd.DataFrame({
    'Samples': [(y_test == cls).sum() for cls in classes],
    'Correct': [((y_test == cls) & (y_pred == cls)).sum() for cls in classes]
}, index=classes)

class_accuracy['Accuracy'] = class_accuracy['Correct'] / class_accuracy['Samples']
class_accuracy.fillna(0, inplace=True)  # Handles division by zero if any class has 0 samples

print("\nClass-wise Accuracy:")
print(class_accuracy)


# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('figures/feature_importance.png')

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png')

# Plot class-wise accuracy
plt.figure(figsize=(10, 6))
sns.barplot(x=class_accuracy.index, y='Accuracy', data=class_accuracy)
plt.title('Accuracy by Wine Quality Class')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.xlabel('Wine Quality')
plt.tight_layout()
plt.savefig('figures/class_accuracy.png')

print("\nAnalysis complete. Results saved in 'figures' directory.") 
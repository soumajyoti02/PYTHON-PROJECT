import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier  # Example model
from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with your data file)
data = pd.read_csv('your_dataset.csv')

# 1. Data Exploration and Visualization
print("Data Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

print("\nMissing Values:")
missing_values = data.isnull().sum()
print(missing_values)

# Visualize correlations using a heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Visualize the data (customize plots as needed)
# Example: Pairplot for numeric features
sns.pairplot(data, hue='target_column')
plt.title("Pairplot of Numeric Features")
plt.show()

# 2. Data Cleaning
# Handle missing values (customize as needed). Replace missing values in a dataset with the mean of the available values.
imputer = SimpleImputer(strategy='mean')
data[['numeric_column']] = imputer.fit_transform(data[['numeric_column']])

# Handle duplicate rows (customize as needed). It will delete the Duplicate rows.
data.drop_duplicates(inplace=True)

# 3. Feature Engineering
# Encoding categorical variables (customize as needed). 
# Encoding is needed in ML to convert categorical data into a numerical format that machine learning algorithms can understand and process.
label_encoder = LabelEncoder()
data['categorical_column'] = label_encoder.fit_transform(data['categorical_column'])

# Feature scaling (normalize or standardize numeric features)
# It will Scale the large values in the dataset to decrease OUTLIERS.
scaler = StandardScaler()
data[['numeric_column']] = scaler.fit_transform(data[['numeric_column']])

# Feature selection (customize as needed)
X = data.drop('target_column', axis=1)
y = data['target_column']

# If Dataset has many features then ---> SelectKBest Used to select the top 'k' most important features from a dataset
selector = SelectKBest(f_classif, k=5)  # Adjust 'k' as needed
X_new = selector.fit_transform(X, y)

# 4. Handling Imbalanced Data
# Oversample the minority class using SMOTE (Synthetic Minority Over-sampling Technique)
print(data['target_column'].value_counts()) # To check if the DataSet is balanced or not.

# If not balanced then Create an instance of the SMOTE class
# sampling_strategy='auto' means that SMOTE will balance the classes to have the same number of samples as the majority class.
# random_state=42 sets a random seed for reproducibility.
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_new, y)

# 5. Model Training and Evaluation
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a machine learning model (replace with your model of choice)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Model Interpretability (Example: SHAP values)
# Replace this section with your chosen interpretability method
# Example: Calculate SHAP values for feature importance
#  It provides a way to understand the contribution of each feature to a model's predictions. 
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot (customize as needed)
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title("SHAP Summary Plot")
plt.show()

'''
 7. Cross-Validation Strategies
 Perform stratified k-fold cross-validation (customize 'k' as needed)
 StratifiedKFold is a cross-validation method that ensures that,
    each fold maintains the same class distribution as the original dataset.
'''
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=k_fold, scoring=make_scorer(accuracy_score))

print("Cross-Validation Scores:")
print(cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# 8. Further Customization
# Depending on your dataset and problem, you may need to implement additional preprocessing steps or other advanced techniques.

# 9. Save the cleaned and preprocessed data if needed
# Example: data.to_csv('cleaned_data.csv', index=False)

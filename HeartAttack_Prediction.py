import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\HeartAttack_Predictor\Heart Attack.csv")
# Download Dataset From Here: https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset/download?datasetVersionNumber=2

# Overview of the Dataset
print(f"Shape is {df.shape}") # Getting Rows and Columns number
print(df.describe()) # getting information about DataSet
print(f"Columns are ---->\n{df.columns}") # Name of the columns

# Understanding the Variables
print(df['gender'].value_counts()) # To count the name and female numbers in Data
print(df['class'].value_counts()) # To count total Positive and negetive chance of Heart Attack in data

# sns.histplot(df['age']) # Checking the Age Frequency.
# plt.show()

# Data Cleaning and Preprocessing
print(f"Duplicate Values in DataSet are {df.duplicated().sum()}")
print(f"Null value Checking ColumnWise ---->\n{df.isnull().sum()}")

# Exploratory Data Analysis (EDA)
# df.hist() # Will Plot histogram of all the columns.
# plt.show()

# Converting 'class' column values into numerical values.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
df['class'] = le.fit_transform(df['class'])

# Correlation Measures the strength and direction of the linear relationship between two variables
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='Dark2') # Plotting this Correlation matrix
# plt.show()

# Scaling is necessary to ensure all features are on Similar scale or magnitude. 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
to_Scale = ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
df[to_Scale] = scaler.fit_transform(df[to_Scale])

X = df.drop('class', axis=1) # Selecting the Independent columns.
y = df['class'] # Selecting dependent Column.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
lgr = LogisticRegression()
lgr.fit(X_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

print(f"Score of Logistic Regression is {lgr.score(X_test, y_test)}")
print(f"Score of RandomForest Classification is {rfc.score(X_test, y_test)}")

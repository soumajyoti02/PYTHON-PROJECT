import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\Mobile Price Classification\mobile_price.csv")

# Checking the DataSet
print(f"Shape of Dataset is {df.shape}")
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(f"CHecking if any duplicate value is present: {df.duplicated().sum()}")

# Checking the Relation of each column with other columns
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(12, 12))
sns.heatmap(corr_matrix, annot=True, cmap="Purples", fmt=".2f")
plt.show()

# Dropping the columns which has least importance --> Found by correlation_matrix above
df.drop(columns=['fc','touch_screen', 'wifi', 'talk_time'], inplace=True)

# Check the dataset is balanced or not
print(df['price_range'].value_counts())

# Visualizing the frequency of target column
sns.histplot(data=df, x='price_range')
plt.show()

# Checking the Dependency of each columns with Target column
plt.subplot(2, 3, 1)
sns.barplot(x=df['price_range'], y=df['ram'])
plt.subplot(2, 3, 2)
sns.barplot(x=df['price_range'], y=df['clock_speed'])
plt.subplot(2, 3, 3)
sns.barplot(x=df['price_range'], y=df['mobile_wt'])
plt.subplot(2, 3, 4)
sns.barplot(x=df['price_range'], y=df['battery_power'])
plt.subplot(2, 3, 5)
sns.barplot(x=df['price_range'], y=df['n_cores'])
plt.subplot(2, 3, 6)
sns.barplot(x=df['price_range'], y=df['int_memory'])

plt.show()

# Extracting the Outliers
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05)  # Adjust the contamination parameter
# we set the contamination parameter to 0.05, indicating that we expect around 5% of the data to be outliers.
clf.fit(df)
outliers = clf.predict(df) == -1

# Checking no. of outliers in Dataset
df['outliers'] = outliers
print(df['outliers'].value_counts())

# Removing the outliers from the DataSet
df = df[df['outliers'] == False]
print(df['outliers'].value_counts())


# Modelling Part
X = df.drop(columns='price_range') # Independent Columns
y = df['price_range'] # Target Column

# Splitting the dataSet into 80% train Data & 20% test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data Preprocessing
robust_scaler = RobustScaler()
X_train = robust_scaler.fit_transform(X_train)
X_test = robust_scaler.fit_transform(X_test)

# Instantiating the Models
random_forest_model = RandomForestClassifier(criterion='gini')
gb_model = GradientBoostingClassifier()
logistic_model = LogisticRegression()
discitionTree_model = DecisionTreeClassifier()

# Training those Models
random_forest_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
discitionTree_model.fit(X_train, y_train)

# Checking the Results
print(f"Random Forest Score {random_forest_model.score(X_test, y_test)}")
print(f"Gradient Boosting Score {gb_model.score(X_test, y_test)}")
print(f"Logistic Score {logistic_model.score(X_test, y_test)}")
print(f"Discition Tree Score {discitionTree_model.score(X_test, y_test)}")

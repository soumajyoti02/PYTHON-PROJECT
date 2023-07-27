import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\House Price Prediction\Housing.csv")
# Download from here: https://www.kaggle.com/datasets/ashydv/housing-dataset/download?datasetVersionNumber=1
# Then Paste the path of that downloaded csv file above.

# Check these things 1st in every DataSet
print(f"Rows and Columns in this DataSet is {df.shape}")
print(f"Data Information is {df.info()}")
print(f"Columns are ---->\n{df.columns}")
print("Null Value Analysis ---->\n", df.isnull().sum()) # If any null values found, then deal with them by fillna or dropna method.

# Make a histogram to check the terget Column frequency
plt.subplot(2,2,1)
sns.histplot(x=df['price'])

# Plot All the independent column vs Target column plots one by one
plt.subplot(2,2,2)
sns.scatterplot(x=df['area'], y=df['price'], hue=df['airconditioning'], palette='Dark2')

# For Replacing text Containing columns into numerical Values
le = LabelEncoder()
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for cols in categorical_cols:
    df[cols] = le.fit_transform(df[cols])


# Creating an instance of the MinMaxScaler
# Scaling data is necessary to ensure that all features are on a similar scale or magnitude
# It prevents any particular feature from dominating the learning process simply because of its larger magnitude.
scaler = MinMaxScaler()
col_to_scale = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price'] # These are all independent numerical Column names
df[col_to_scale] = scaler.fit_transform(df[col_to_scale])

X = df.drop('price', axis=1) # Selecting all the independent columns
y = df['price'] # Selecting the target Column

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Dividing dataSet into 80% training & 20% Testing data

reg_model = LinearRegression() # Instantiation of Our Model
reg_model.fit(x_train, y_train)
pred = reg_model.predict(x_test)

# Plotting our Actual Value vs Predicted Value Graph
plt.subplot(2,2,3)
sns.regplot(x=y_test, y=pred)
plt.show()

# Evaluation of our model
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"Mean Square Error is {mse}")
print(f"R2 Score is {r2}")



















# Predicting the Wholesale_Price of Maple_Syrup 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\Maple Syrup Prices\Domestic_Maple_Syrup_Production_2000-2021.csv")
#Download DataSet From here: https://www.kaggle.com/datasets/datasciencedonut/maple-syrup-production-in-the-usa-2000-2021/download?datasetVersionNumber=3

print(f"Shape is {df.shape}") # Getting No. of Rows & Cols
print(df.info())
print(f"Columns are--->\n{df.columns}")
print(df.isna().sum()) # Getting Column wise total Null values

df.drop(['Date_Open', 'Date_Closed', 'Year', 'Bulk_P_Price', 'Bulk_G_Price'], axis=1, inplace=True) # Least Affecting Columns

# Fixing 'Production' & 'Value' Columns value because some values was in String Format initially.
df['Production'] = df['Num_of_Taps'] * df['Yield_per_Tap'] 
df['Value'] = df['Production'] * df['Avg_Price']

invalid_columns = ['Wholesale_Price'] # Some values was written '(D)' initially in this column. 
# Checking non-numeric Columns. errors='coerce' will replace all non-numeric values into 'NaN'.
df[invalid_columns] = df[invalid_columns].apply(pd.to_numeric, errors='coerce') 

nan_value = df[df.isnull().any(axis=1)] # Seperating those NaN columns from Main Dataset
df.dropna(subset=df.columns, inplace=True) # Removing NaN containing Rows from main DataSet

print(df.info())
print(nan_value.info())

# Plots
plt.subplot(2, 3, 1)
sns.regplot(x='Num_of_Taps', y='Wholesale_Price', data=df)
plt.subplot(2, 3, 2)
sns.regplot(x='Yield_per_Tap', y='Wholesale_Price', data=df)
plt.subplot(2, 3, 3)
sns.regplot(x='Production', y='Wholesale_Price', data=df)

plt.subplot(2, 3, 4)
sns.regplot(x='Avg_Price', y='Wholesale_Price', data=df)
plt.subplot(2, 3, 5)
sns.regplot(x='Value', y='Wholesale_Price', data=df)
plt.subplot(2, 3, 6)
sns.regplot(x='Retail_Price', y='Wholesale_Price', data=df)

plt.show()

X = df.iloc[:, 1:7] # Seperating all the independent Columns i.e [Num_of_Taps,Yield_per_Tap,Production,Avg_Price,Value,Retail_Price]
y = df['Wholesale_Price'] # Target Column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Splitting data into 80% Train & 20% test data

# Scaling Data so that all the features has similar weight.
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Instantiating Different models 
rfr = RandomForestRegressor()
xgbr = xgb.XGBRegressor(n_estimators=100, random_state=80)
knnr = KNeighborsRegressor(n_neighbors=3)
svrr = SVR()
linear = LinearRegression()

# Train those models
rfr.fit(X_train, y_train)
xgbr.fit(X_train, y_train)
knnr.fit(X_train, y_train)
svrr.fit(X_train, y_train)
linear.fit(X_train, y_train)

print(f'Random forest Score: {rfr.score(X_test, y_test)}')
print(f'XGBoost Score: {xgbr.score(X_test, y_test)}')
print(f'KNN Score: {knnr.score(X_test, y_test)}')
print(f'SVR Score: {svrr.score(X_test, y_test)}')
print(f'Linear Regresison Score: {linear.score(X_test, y_test)}')

# Now replacing the NaN values with prediction from our models
NaN_X_Value = nan_value.iloc[:, 1:7]
linear_prediction = linear.predict(NaN_X_Value)
print(linear_prediction)

nan_value['Wholesale_Price'] = abs(linear_prediction) # Modifying 'Wholesale_Price' column with our new Values
print("\nHere is the Predicted Missing Values in WholeSale Price Column--->")
print(nan_value)












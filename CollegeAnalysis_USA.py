import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\College Rank US\2022 US College Rankings.csv")
# Download From Here: https://www.kaggle.com/datasets/dylankarmin/2022-college-rankings-compared-to-tuition-costs/download?datasetVersionNumber=11

print(f"Shape is {df.shape}")
print(f"Columns are:--->\n{df.columns}")
print(f"Null Checking:-->\n{df.isnull().sum()}")
print(df.info())

#Quartile Deviation & Outliers Checking
q1 = df['Tuition'].quantile(.25)
q3 = df['Tuition'].quantile(.75)
iqr = q3 - q1 # Interquartile Range, describe the spread or dispersion of a dataset.
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
outliers = df[(df['Tuition'] < lower_bound) | (df['Tuition'] > upper_bound)]
print(outliers)

# Barplot of Top 10 Expensive College
sorted_tuition = df.sort_values(by='Tuition', ascending=False)
top_10 = sorted_tuition.head(10)
sns.barplot(x = 'Tuition', y = 'College Name', data=top_10, palette='ocean')
plt.title("Top 10 Expensive College")
plt.show()

# Barplot of Top 10 Least Populated College
least_expensive = df.sort_values(by='Enrollment Numbers')
least_10 = least_expensive.head(10)
sns.barplot(x = 'Enrollment Numbers', y = 'College Name', data=least_10)
plt.title("Top 10 Least Populated College")
plt.show()

# Barplot of Top 10 Revenue Generating College
df['University Revenue'] = (df['Tuition'] * df['Enrollment Numbers']) / 1000000
top_revenue = df.sort_values(by='University Revenue', ascending=False).head(10)
sns.barplot(x = 'University Revenue', y = 'College Name', data=top_revenue)
plt.title("Top 10 Revenue Generating College")
plt.show()

sns.histplot(data=df, x='University Revenue', bins=20)
plt.title("Histogram of Revenues")
plt.show()

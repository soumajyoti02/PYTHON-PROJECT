import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv(r'C:\Users\ASUS\Desktop\Codes\Python\Python Projects\GPA Study Hours\gpa_study_hours.csv')
# Download link: https://www.kaggle.com/datasets/joebeachcapital/gpa-study-hours/download?datasetVersionNumber=1
print(f"No. of rows and columns are: {df.shape}")
print(f"Data information is {df.info()}")
print(f"Null values are\n{df.isnull().sum()}")

def define_grade(gpa):
    if 4.0 >= gpa > 3.3:
        return 'A'
    elif 3.3 >= gpa > 2.3:
        return 'B'
    elif 2.3 >= gpa > 1.3:
        return 'C'
    elif 1.3 >= gpa > 1.0:
        return 'D'
    elif 1.0 >= gpa > 0.0:
        return 'F'
    else:
        return False

print(f"Correlation coeff is\n{df.corr()}")
print(f"\nExtracting Statistical Summery\n{df.describe()}")

# Seperate by grade type
df['grade'] = df["gpa"].apply(lambda x: define_grade(x))
df = df[df['gpa'] <= 4.0] # Removing the entries having grade greater than 4.0
print(f"\nSample Dataset is\n{df.head()}")

sns.scatterplot(x='study_hours', y='gpa', data=df,hue='grade')

# Seperating A and B grades entry from dataset
A = df[df['grade'] == 'A']
B = df[df['grade'] == 'B']

sns.displot(data=df, x='study_hours', hue='grade', col='grade', binwidth=4)
plt.show()

# Mean of hours studing for A and B students
mu_a = A['study_hours'].mean()
mu_b = B['study_hours'].mean()
mu_X = mu_a - mu_b # Mean Difference

print(f"Avarage Study hours of A is {mu_a}")
print(f"Avarage Study hours of B is {mu_b}")
print(f"Avarage difference between no. of hours study is {mu_X}")





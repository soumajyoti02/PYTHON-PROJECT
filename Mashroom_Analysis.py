import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

# About this file
# Attribute Information: (classes: edible=e, poisonous=p)
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# bruises: bruises=t,no=f
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# gill-attachment: attached=a,descending=d,free=f,notched=n
# gill-spacing: close=c,crowded=w,distant=d
# gill-size: broad=b,narrow=n
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# stalk-shape: enlarging=e,tapering=t
# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# veil-type: partial=p,universal=u
# veil-color: brown=n,orange=o,white=w,yellow=y
# ring-number: none=n,one=o,two=t
# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

df = pd.read_csv(r"C:\Users\ASUS\Desktop\Codes\Python\Python Projects\mashroom_dataset\mashroom.csv")
# Download from here: https://www.kaggle.com/datasets/uciml/mushroom-classification/download?datasetVersionNumber=1
df.columns=['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above-ring', 'stalk_surface_below-ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']

print(df.columns) # Checking Column Names
print(df.info()) # Checking Column Status
print(df.isna().sum()) # Checking about any Null values
print(df.shape) # Getting the size

df['class'].replace({'p': False, 'e': True}, inplace=True) # Replacing the Class column into True or False 

# Numericalize all the columns
all_columns = list(df.columns)
le = LabelEncoder()
for col in all_columns:
    df[col] = le.fit_transform(df[col])

corr_matrix = df.corr() # Checking how much each column is related with others
sns.heatmap(corr_matrix, annot=True, cmap='Dark2')
plt.show()

# Grouping all the Columns with Class result
# Checking The mashroom eadible status according to every column
for cols in all_columns:
    print(df.groupby([cols, 'class']).count())


# Model Making ----->
X = df.drop('class', axis=1) # Extracting independent columns for Model Making
y = df['class'] # Target Column

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Deviding our data into 80% training & 20% test data

# Model Instantiating
rfc = RandomForestClassifier()
logr = LogisticRegression()
linr = LinearRegression()

# Model Training
rfc.fit(X_train, y_train)
logr.fit(X_train, y_train)
linr.fit(X_train, y_train)

# Checking the scores of each models
print(f"\nRandom Forest Score: {rfc.score(x_test, y_test)}")
print(f"Logistic Score: {logr.score(x_test, y_test)}")
print(f"Linear Score: {linr.score(x_test, y_test)}")

# Checking the importance of each columns wrt our RandomForest Model
feature_importances = rfc.feature_importances_
print("\nImportance of each columns are:\n",feature_importances)













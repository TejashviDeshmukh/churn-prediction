import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD DATASET

data = pd.read_csv("customer_churn_business_dataset.csv")

print("FIRST 5 ROWS")
print(data.head())

print("\nDATASET INFO")
print(data.info())

print("\nMISSING VALUES")
print(data.isnull().sum())

# HANDLE MISSING VALUES


data['complaint_type'] = data['complaint_type'].fillna("Unknown")

# DROP CUSTOMER ID

data.drop("customer_id", axis=1, inplace=True)


# ENCODE CATEGORICAL COLUMNS

le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = le.fit_transform(data[column])


# FEATURES AND TARGET

X = data.drop("churn", axis=1)
y = data["churn"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# TRAIN MODEL

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced_subsample',
    random_state=42

)

model.fit(X_train, y_train)


# PREDICTIONS
y_pred = model.predict(X_test)

# EVALUATION

accuracy = accuracy_score(y_test, y_pred)

print("\nACCURACY:")
print(round(accuracy, 2))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# CONFUSION MATRIX HEATMAP

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


plt.figure(figsize=(5,4))
sns.countplot(x='churn', data=data)

plt.title("Churn Count")

plt.show()

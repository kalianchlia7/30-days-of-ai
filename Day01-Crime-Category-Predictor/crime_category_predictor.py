#Crime Category Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("day1/train.csv.zip")
df = df.sample(n=50000, random_state=42)

#analyzing data
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nMissng Values:\n")
print(df.isnull().sum())

print("\nData Types:\n")
print(df.dtypes)

#using Dates column to extract hour and month
df["Dates"] = pd.to_datetime(df["Dates"]) 
#converting from string to numerical datetime object
df["Hour"]=df["Dates"].dt.hour
#dt allows access to the datetime properties of the column, such as hour, month, year, etc.
df["Month"]=df["Dates"].dt.month
print(df[["Dates","Hour","Month"]].head())

#prepare input features (X) and target variable (y)
X = pd.get_dummies(
    #one-hot encoding categorical variables into numerical values (binary)
    df[["Hour","Month","DayOfWeek","PdDistrict"]],
    drop_first=True
)

y=df["Category"]
#defined target variable as the crime category (type of crime)

print("Feature matrix shape:", X.shape) #run: (878049, 17)
#features are four input columns (hour, month, dayofweek, pddistrict) that are encoded into numerical values

print("Target variable shape:", y.shape) #run: (878049,), means we have one correct crime category for every crime report.
#matches up exactly 

#target is the crime category (type of crime)
print("\nFirst 5 encoded columns:")
print(X.head())

#supervised mchine learning model

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,  #reserve 20% of the dataset for testing 
    random_state=42 #data shuffled randomly
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

#train the model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained successfully.")

predictions = model.predict(X_test)

#see accuracy of the model
print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
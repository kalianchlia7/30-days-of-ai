import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
df = pd.read_csv('day4/network_traffic.csv')

print("first 5 rows:")
print(df.head())

print("\ndataset Shape:")
print(df.shape)

#check dataset for missing values & type

print("\nmissing values")
print(df.isnull().sum())
print("\ndataset types")
print(df.dtypes)

#ENCODE CATEGORICAL DATA

encoder = LabelEncoder()

df["Protocol"]=encoder.fit_transform(df["Protocol"])
df["Attack"] = encoder.fit_transform(df["Attack"])

#Seperate features and target

X=df.drop("Attack", axis=1)
#drop the target column from the features
y=df["Attack"]

print("\nFeatures:")
#will print features not including 'attack' column
print(X.head())


print("\nLabels:")
#will print the target column 'attack'
print(y.head())

#SPLIT TRAIN AND TEST DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape) #this will print the shape of the training features
print("Testing Features:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

#CREATE RANDOM FOREST MODEL

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5 ,
    random_state=42
)

#TRAIN THE MODEL

model.fit(X_train, y_train)
print("\nRandom Forest trained successfully.")

#PREDICT ON TEST DATA

predictions=model.predict(X_test)
print("\nPredictions:")
print(predictions)

#EVALUATE THE MODEL

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        #this will print the class names instead of 0 and 1
        target_names=["Normal", "Attack"],
        #this will prevent division by zero error in case of no samples for a class
        zero_division=0
    )
)

#CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)
#from sklearn.metrics import ConfusionMatrixDisplay

display.plot() #this will plot the confusion matrix
plt.title("Cyberattack Detection Confusion Matrix")
plt.show()

#FEATURE IMPORTANCE

#*this will print the feature importance of each feature in the model
importance = pd.Series( #this is from pandas library, it will create a series object
    #series object is a one-dimensional array with axis labels, it can hold any data type
    model.feature_importances_, #aka random forest model's feature importance values, numbers
    index=X.columns #this will set the index of the series to the feature names
    #aka use features names as the labels for these importance values, numbers
)

importance = importance.sort_values( #this will sort the series in descending order of importance
    ascending=False
)

print("\nFeature Importance:")
print(importance)
#aka how useful each feature is in predicting the target variable, the higher the value, the more important the feature is

#VISUALIZE FEATURE IMPORTANCE

importance.plot(
    kind="bar" #this will plot a bar chart of the feature importance
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=45) #this will rotate the x-axis labels by 45 degrees to make them more readable
plt.tight_layout() #this will adjust the padding between and around subplots to minimize overlap
plt.show()



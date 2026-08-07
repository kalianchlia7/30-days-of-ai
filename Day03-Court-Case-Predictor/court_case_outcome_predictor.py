# Import pandas for working with datasets
import pandas as pd

# Used to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Converts text categories into numerical values
from sklearn.preprocessing import LabelEncoder
#different from tdidf vectorizer, this is for categorical data

# Our machine learning model
from sklearn.tree import DecisionTreeClassifier

# Used to visualize the decision tree
from sklearn.tree import plot_tree

# Evaluate the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Plotting library
import matplotlib.pyplot as plt

#loading dataset
# Load the CSV file
df = pd.read_csv("day3/court_cases.csv")

# Display the first few rows
print(df.head())

# Print dataset shape
print("\nDataset Shape:", df.shape)

# Identify categorical columns
# Create a LabelEncoder object

encoder = LabelEncoder()

# Encode every categorical column
df["CrimeType"] = encoder.fit_transform(df["CrimeType"])
df["WeaponUsed"] = encoder.fit_transform(df["WeaponUsed"])
df["Plea"] = encoder.fit_transform(df["Plea"])
df["Outcome"] = encoder.fit_transform(df["Outcome"])

#there is a priorconvictions column, but its not categorical!
#no neede to encode

# Seperating features and labels
# Everything except Outcome becomes our features
X = df.drop("Outcome", axis=1)
#axis=1 means drop column, axis=0 means drop row

# Outcome is what we're predicting
y = df["Outcome"]

print("\nFeatures:\n") #features are the input data
print(X.head())

print("\nLabels:\n") #outcome is the label
print(y.head())

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Examples:", len(X_train))
print("Testing Examples:", len(X_test))

# Create the Decision Tree Classifier
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

#Train model

model.fit(
    X_train,
    y_train
)
print("Decision Tree trained successfully.")

#predictions
predictions = model.predict(X_test)
print(predictions)

#evaluate model
accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions
    )
)

#Visualize the decision tree
plt.figure(
    figsize=(12,8)
)

plot_tree(
    model,
    feature_names=X.columns, #this is the column names of the features
    class_names=["Acquitted","Convicted"], #this is the class names of the labels
    #what two outcomes we are predicting
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()

#interactive prediction

print("\nCourt Case Outcome Predictor")

while True:
    print("\nCrime Types:")
    print("0 = Assault")
    print("1 = Burglary")
    print("2 = Drug Possession")
    print("3 = Fraud")
    print("4 = Robbery")
    print("5 = Theft")
    print("6 = Vandalism")

    crime = int(input("\nCrime Type: "))
    priors = int(input("Prior Convictions: "))
    weapon = int(input("Weapon Used? (0=No, 1=Yes): "))
    plea = int(input("Plea (0=Guilty, 1=Not Guilty): "))

    prediction = model.predict(
        [[crime, priors, weapon, plea]]
    )[0]

    if prediction == 1:
        print("\nPrediction: Convicted")
    else:
        print("\nPrediction: Acquitted")
    
    again = input("\nTry another case? (y/n): ")

    if again.lower() != "y":
        break
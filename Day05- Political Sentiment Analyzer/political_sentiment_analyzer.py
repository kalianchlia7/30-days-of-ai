import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Small dataset of political statements
data = {
    "text": [
        "The government successfully passed a major economic reform.",
        "The new policy has greatly improved international cooperation.",
        "The administration's diplomatic efforts have been very successful.",
        "The new law provides important protections for citizens.",
        "The country has made significant progress in reducing crime.",
        "The government failed to address the growing economic crisis.",
        "The new policy has caused serious problems for working families.",
        "The administration handled the diplomatic negotiations poorly.",
        "The law has created unnecessary restrictions on citizens.",
        "Crime has increased significantly under the current government.",
        "The government announced a new infrastructure program.",
        "Parliament introduced legislation concerning data privacy.",
        "Officials met with international leaders to discuss trade.",
        "The department released its annual crime statistics.",
        "The administration proposed changes to immigration policy."
    ],
    "sentiment": [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "neutral",
        "neutral",
        "neutral"
    ]
}

# Convert our dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Political Sentiment Dataset:")
print(df)

# Show how many examples belong to each sentiment
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

#seperate features and target

# X contains the text the model will learn from
X = df["text"]

# y contains the sentiment we want the model to predict
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Testing examples:", len(X_test))

#CONVERT text into numbers with TF-IDF

# Convert text into numerical TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#fit.transform vs transform: fit_transform is used on the training data to learn the vocabulary and idf from the training set, 
#while transform is used on the test data to convert it into the same feature space without learning the new (test) vocabulary

print("\nTF-IDF training matrix shape:", X_train_tfidf.shape)
print("TF-IDF testing matrix shape:", X_test_tfidf.shape)

#TRAIN logistic regression model

# Create the classification model
model = LogisticRegression(max_iter=1000)
# Train the model using the TF-IDF features
model.fit(X_train_tfidf, y_train) #fit model using x training inputs and respective y training outputs
print("\nModel trained successfully.")

#MAKE predictions

predictions = model.predict(X_test_tfidf)
#X_test_tfidf is the test data that we want to predict the sentiment for, 
# and predictions will contain the predicted sentiment labels for each of the test examples
#y test is the true sentiment labels for the test examples, which we will use to evaluate the model's performance

print("\nPredictions:")
print(predictions)

#EVALUATE model

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

#TEST new political statements

new_statements = [
    "The government's new economic policy has produced excellent results.",
    "The administration completely failed to address the crisis.",
    "Officials announced a meeting with international representatives."
]

new_statements_tfidf = vectorizer.transform(new_statements)

new_predictions = model.predict(new_statements_tfidf)

print("\nNew Statement Predictions:")

for statement, prediction in zip(
    new_statements,
    new_predictions
):
    print(f"\nStatement: {statement}")
    print(f"Predicted sentiment: {prediction}")

#PREDICTION probabilities

probabilities = model.predict_proba(new_statements_tfidf)

print("\nPrediction Probabilities:")

for statement, prediction, probability in zip(
    new_statements,
    new_predictions,
    probabilities
):
    confidence = max(probability)

    print(f"\nStatement: {statement}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")


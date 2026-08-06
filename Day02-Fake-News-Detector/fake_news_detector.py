# Import pandas for working with datasets
import pandas as pd

# Import matplotlib for visualization
import matplotlib.pyplot as plt

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Convert text into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# ML model...
from sklearn.naive_bayes import MultinomialNB

# Building ML pipeline
from sklearn.pipeline import Pipeline

# Evaluating model.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

#read both datasets
fake=pd.read_csv("day2/Fake.csv.zip")
true=pd.read_csv("day2/True.csv.zip")

fake['label']=0
true['label']=1

df=pd.concat([fake,true], ignore_index=True)

print(df.head())

print("\nDataset Shape", df.shape)


# Shuffle the dataset, frac=1 means we want 100% of rows
df = df.sample(frac=1, random_state=42)
# Reset row numbers after the random of shuffle
df.reset_index(drop=True, inplace=True)

print(df.head())

#creating features
# X contains the article text
X = df["text"]
# y contains the labels
y = df["label"]

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#pipeline for text classification
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    #stop_words="english" removes common words like "the", "is", "in", etc. that do not carry much meaning for classification
    ("model", MultinomialNB())
])
#the pipeline first converts the text into numerical features using TfidfVectorizer,
#and then trains a Multinomial Naive Bayes model on those features.

#train model
#using pipeline for automated feature extraction and model training
#pipeline calls on tfidf without needing to call it separately
#then calls on the model to train it on the features extracted by tfidf
pipeline.fit(X_train, y_train)
print("Model trained successfully!")

#predict on test data
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions
)

plt.title("Fake News Classification")
plt.show()



#Interactive testing

print("\n---------------")
print("Fake News Detector")
print("Type 'quit' to exit.")
print("---------------")

#initiate an infinite loop to allow the user to input news articles for classification
while True:

    article = input("\nPaste a news article:\n")

    if article.lower() == "quit": #if user types "quit", exit the loop and end the program
        break

    prediction = pipeline.predict([article])[0]
    #same idea as writing my_list=[article] and then calling pipeline.predict(my_list)
    #[0] because it calls the first element of the article label (the 0 or 1) in integer format (not list format)

    if prediction == 1:
        print("\nPrediction: REAL NEWS")
    else:
        print("\nPrediction: FAKE NEWS")
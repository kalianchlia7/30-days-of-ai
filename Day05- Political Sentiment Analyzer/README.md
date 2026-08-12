# Political Sentiment Analyzer

## Project Overview

This project uses Natural Language Processing and supervised machine learning to classify political statements as positive, negative, or neutral.

The project demonstrates how text can be converted into numerical features using TF-IDF and then classified using Logistic Regression.

## Objective

The objective is to build a simple NLP pipeline capable of analyzing the sentiment expressed in political statements.

## Machine Learning Pipeline

```text
Political Statements
        ↓
Train/Test Split
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Sentiment Prediction
        ↓
Model Evaluation
```

## Methodology

### TF-IDF

TF-IDF, or Term Frequency-Inverse Document Frequency, converts text into numerical feature vectors.

The technique gives greater importance to words that are informative within a document while reducing the influence of words that appear frequently throughout the dataset.

### Logistic Regression

Logistic Regression is used as the classification model.

Although the model is called "regression," it can be used for classification tasks by estimating the probability that an observation belongs to each class.

In this project, the possible classes are:

* Positive
* Negative
* Neutral

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

The project also uses prediction probabilities to examine the model's estimated confidence when classifying new political statements.

## Example Applications

The same general NLP pipeline could be adapted for:

* Political speech analysis
* Policy document analysis
* News sentiment analysis
* Public opinion research
* Legislative text classification
* International relations research

## Limitations

This project uses a very small manually constructed dataset for demonstration purposes.

Therefore, the resulting accuracy and classification metrics should not be interpreted as evidence of a reliable real-world political sentiment model.

Political language is highly contextual, and a larger dataset would be required to build a robust sentiment classifier.

Potential improvements include:

* Using a substantially larger dataset
* Incorporating real political speeches or news articles
* Performing cross-validation
* Testing additional NLP models
* Using word embeddings
* Comparing Logistic Regression with neural networks and transformer-based models

## Technologies

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression

## Key Concepts Demonstrated

* Natural Language Processing
* Text Classification
* TF-IDF
* Logistic Regression
* Train/Test Splitting
* Classification Metrics
* Probability-Based Predictions
* Supervised Learning

## Author

Kali Anchlia

UC San Diego

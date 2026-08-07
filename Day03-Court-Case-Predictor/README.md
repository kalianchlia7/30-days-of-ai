# Court Case Outcome Predictor

A machine learning project that uses a Decision Tree classifier to predict a simplified court case outcome based on case characteristics.

This project is part of my **30 Days of AI** challenge, where I build and publish one AI or machine learning project each day while exploring different algorithms, techniques, and real-world applications.

## Project Overview

The goal of this project is to demonstrate how a Decision Tree can be used for binary classification.

The model uses simplified case characteristics including:

- Crime type
- Number of prior convictions
- Whether a weapon was used
- Plea entered

The model then predicts one of two simplified outcomes:

- Convicted
- Acquitted

The dataset is intentionally small and synthetic for educational purposes. It should not be interpreted as a real-world system for predicting judicial decisions.

## Machine Learning Approach

The project uses a Decision Tree Classifier.

A Decision Tree learns a series of decision rules from the training data. These rules form a tree structure in which the model progressively splits observations into groups before reaching a final prediction.

Example:

```text
Weapon Used?
    |
    ├── Yes → Prior Convictions?
    |              |
    |              ├── High → Convicted
    |              └── Low → Acquitted
    |
    └── No → Plea?
                   |
                   ├── Guilty → Convicted
                   └── Not Guilty → Acquitted

# Cyberattack Detection Using Random Forest

## Project Overview

This project uses a Random Forest classification model to distinguish between simulated normal network traffic and simulated cyberattack traffic.

The project demonstrates a complete machine learning classification pipeline, including data preparation, categorical encoding, train/test splitting, model training, evaluation, confusion matrix analysis, and feature importance analysis.

## Objective

The objective is to investigate whether characteristics of network connections can be used to classify traffic as either:

* Normal
* Attack

## Machine Learning Approach

The project uses a Random Forest Classifier.

Random Forest is an ensemble learning algorithm that combines predictions from multiple Decision Trees. Using multiple trees can make the model more robust than relying on a single Decision Tree.

The model was configured with:

* 100 Decision Trees
* Maximum tree depth of 5
* Fixed random state for reproducibility

## Features

The model uses the following network characteristics:

* Protocol
* Duration
* Source Bytes
* Destination Bytes
* Failed Logins
* Connection Count
* Port

The target variable is:

* Attack

## Workflow

```text
Network Traffic Data
        ↓
Data Inspection
        ↓
Categorical Encoding
        ↓
Feature / Target Separation
        ↓
Train/Test Split
        ↓
Random Forest Training
        ↓
Predictions
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
```

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Precision measures how often predicted attacks are actually attacks.

Recall measures how many actual attacks were successfully detected.

The confusion matrix provides a breakdown of true positives, true negatives, false positives, and false negatives.

## Feature Importance

The Random Forest's built-in feature importance scores are used to identify which network characteristics were most useful for distinguishing simulated attacks from normal traffic.

This provides an additional way to interpret the model beyond its overall prediction accuracy.

## Limitations

The dataset used in this project is synthetic and intentionally small.

Therefore, the results should not be interpreted as evidence that the model can reliably detect real-world cyberattacks.

A more advanced version of this project could use a real network intrusion dataset and investigate:

* Class imbalance
* Different types of attacks
* Model generalization
* Hyperparameter tuning
* Cross-validation
* Comparison with other classification algorithms

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Key Concepts Demonstrated

* Supervised Machine Learning
* Binary Classification
* Random Forest
* Decision Trees
* Ensemble Learning
* Train/Test Splitting
* Classification Metrics
* Confusion Matrices
* Feature Importance
* Model Interpretability

## Author

Kali Anchlia

UC San Diego

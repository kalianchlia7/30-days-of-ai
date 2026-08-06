# Fake News Detector

A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP). This project is the second installment in my **30 Days of AI** series, where I build and publish one AI or machine learning project each day to explore different models, algorithms, and real-world AI applications.

---

# Project Overview

The goal of this project is to determine whether a news article is fake or real based solely on its text.

Unlike traditional machine learning datasets that consist of numerical features, this project works with unstructured text data. Using Natural Language Processing (NLP), article text is converted into numerical representations before being classified by a machine learning model.

This project demonstrates a complete NLP workflow, including:

- Loading and combining datasets
- Creating classification labels
- Preprocessing text
- Converting text into TF-IDF features
- Training a Naive Bayes classifier
- Evaluating model performance
- Visualizing results with a confusion matrix
- Building an interactive prediction tool

---

# Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

The dataset contains thousands of labeled news articles divided into:

- `Fake.csv`
- `True.csv`

Each article contains:

- Title
- Text
- Subject
- Publication date

Due to dataset licensing and file size, the dataset is not included in this repository.

---

# Machine Learning Pipeline

The workflow for this project is:

```text
News Article
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Numerical Features
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Prediction
```

---

# Why TF-IDF?

Machine learning models cannot understand raw text.

TF-IDF (Term Frequency–Inverse Document Frequency) converts words into numerical values by measuring how important each word is within an article relative to the rest of the dataset.

This representation allows the classifier to identify meaningful language patterns associated with fake and real news.

---

# Why Naive Bayes?

Multinomial Naive Bayes is a classic algorithm for text classification because it is:

- Fast to train
- Computationally efficient
- Highly effective for document classification
- A strong baseline model for many NLP tasks

---

# Results

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Because this dataset contains clear linguistic differences between fake and real articles, Naive Bayes typically achieves very high accuracy.

---

# Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

# Project Structure

```text
fake-news-detector/
│
├── fake_news_detector.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── images/
    └── confusion_matrix.png
```

---

# Features

- Binary text classification
- TF-IDF feature extraction
- Naive Bayes classifier
- Automatic model evaluation
- Confusion matrix visualization
- Interactive command-line prediction tool

---

# Future Improvements

Potential enhancements include:

- Text preprocessing (stemming and lemmatization)
- N-gram feature extraction
- Hyperparameter tuning
- Comparing multiple classifiers such as Logistic Regression, Support Vector Machines, Random Forests, and XGBoost
- Fine-tuning transformer models such as BERT
- Building a web interface with Flask or Streamlit

---

# Key Concepts Demonstrated

- Natural Language Processing (NLP)
- Text Classification
- TF-IDF Vectorization
- Supervised Learning
- Naive Bayes Classification
- Scikit-learn Pipelines
- Model Evaluation
- Confusion Matrix Interpretation

---

# What I Learned

Through this project I learned how to:

- Build an end-to-end NLP pipeline
- Transform text into machine-readable features
- Train and evaluate a Naive Bayes classifier
- Interpret classification metrics
- Visualize model performance
- Build an interactive AI application capable of classifying unseen news articles

---

# 30 Days of AI

This repository is part of my **30 Days of AI** challenge.

The objective of this challenge is to design, build, and publish one AI or machine learning project each day while exploring a wide variety of machine learning and deep learning techniques. Each project focuses on learning a new AI concept while building a portfolio of practical, real-world applications.

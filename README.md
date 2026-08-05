# Crime Category Predictor

A machine learning classification project that predicts the category of a crime using historical crime data from San Francisco. This project is the first in my **30 Days of AI** series, where I build and publish one AI or machine learning project each day to strengthen my understanding of machine learning models, algorithms, and data science workflows.

---

# Project Overview

The goal of this project is to build a multiclass classification model capable of predicting the type of crime based on temporal and location-related features.

This project follows the complete machine learning pipeline:

- Load and explore the dataset
- Check for missing values and data types
- Perform feature engineering
- Encode categorical variables
- Split the data into training and testing sets
- Train a Logistic Regression classifier
- Evaluate model performance

---

# Dataset

The dataset contains **878,049 historical crime reports** from San Francisco.

Each record contains information such as:

- Date and time
- Crime category (target)
- Day of the week
- Police district
- Address
- Geographic coordinates (latitude and longitude)

To keep this project lightweight and fast to train, a random sample of **50,000 observations** was used during model development.

---

# Feature Engineering

The following features were used for training:

- Hour of the day
- Month
- Day of the week
- Police district

The target variable was:

- **Crime Category**

Categorical variables were converted into numerical features using **One-Hot Encoding**.

---

# Machine Learning Model

**Algorithm**

- Logistic Regression (Scikit-learn)

## Why Logistic Regression?

This project intentionally begins with a simple baseline model.

Logistic Regression was chosen because it is:

- Fast to train
- Easy to interpret
- Well-suited for multiclass classification problems
- A strong baseline before experimenting with more advanced machine learning algorithms

---

# Results

## Model Accuracy

**22.5%**

Although the accuracy may seem modest, the model predicts among **39 different crime categories** using only a limited set of engineered features.

This project establishes a baseline that can be improved through additional feature engineering, class balancing techniques, hyperparameter tuning, and more advanced machine learning models.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# Project Structure

```text
crime-category-predictor/
│
├── crime_category_predictor.py
├── train.csv.zip
├── pyproject.toml
├── uv.lock
└── README.md

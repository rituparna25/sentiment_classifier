# Simple Sentiment Classifier (CSV-Based)

This project is a basic sentiment analysis model using scikit-learn, trained on a simple CSV dataset of positive and negative phrases.

## 📝 Dataset Format

CSV file: `sentiment_data.csv`

| sentence               | label     |
|------------------------|-----------|
| we are happy           | POSITIVE  |
| we are sad             | NEGATIVE  |
| ...                    | ...       |

## 🚀 How to Run

1. Install requirements:
```bash
pip install scikit-learn pandas
```

2. Run the script:
```bash
python sentiment_classifier.py
```

## ✅ Features

- Loads labeled sentence data from CSV

- Converts text to vectors using TfidfVectorizer

- Trains two models:

- DecisionTreeClassifier

- MultinomialNB (Naive Bayes)

- Performs stratified train-test split to ensure label balance

- Prints accuracy and detailed classification report for both models

## 📦 Files

- `sentiment_data.csv` – training data
- `sentiment_classifier.py` – main script
- `README.md` – project guide


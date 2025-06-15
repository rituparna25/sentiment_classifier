import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['sentence'], df['label']

def train_and_evaluate_model(model, vectorizer, X_train, X_test, y_train, y_test, model_name):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    print(f"\n✅ {model_name} trained successfully.")
    print(f"🎯 Accuracy ({model_name}): {accuracy_score(y_test, y_pred):.2f}")
    print(f"📊 Classification Report ({model_name}):\n")
    print(classification_report(y_test, y_pred))

def main():
    sentences, labels = load_data('sentiment_data.csv')

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.3, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer()

    # Decision Tree
    train_and_evaluate_model(DecisionTreeClassifier(), vectorizer, X_train, X_test, y_train, y_test, "Decision Tree")

    # Naive Bayes
    train_and_evaluate_model(MultinomialNB(), vectorizer, X_train, X_test, y_train, y_test, "Naive Bayes")

if __name__ == "__main__":
    main()

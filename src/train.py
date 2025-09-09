import sys
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from preprocess import load_csv


def train_model(csv_path, model_out="src/model.pkl"):
    # Load data
    df = load_csv(csv_path)
    X = df["text"]
    y = df["intent"]

    # Stratified train/test split to make sure every intent is represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Build pipeline: TF-IDF + Logistic Regression
    vect = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    clf = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(vect, clf)

    # Train model
    pipeline.fit(X_train, y_train_enc)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test_enc, y_pred))

    # Only report on labels actually in y_test
    labels = np.unique(y_test_enc)
    print(
        classification_report(
            y_test_enc, y_pred, labels=labels, target_names=le.classes_[labels]
        )
    )

    # Save pipeline + label encoder
    joblib.dump({"pipeline": pipeline, "label_encoder": le}, model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_data.csv"
    train_model(csv_path)

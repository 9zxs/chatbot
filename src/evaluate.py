import joblib
from preprocess import load_csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

def evaluate_model(csv_path, model_path='src/model.pkl'):
    # Load model
    data = joblib.load(model_path)
    pipeline = data['pipeline']
    le = data['label_encoder']

    # Load dataset
    df = load_csv(csv_path)
    X = df['text'].tolist()
    y_true = le.transform(df['intent'])
    y_pred = pipeline.predict(X)

    # Intent classification metrics
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    # Approximate response quality using BLEU between text input and predicted intent name
    # bleu_scores = []
    # for xi, yi in zip(X, y_pred):
    #     predicted_intent = le.inverse_transform([yi])[0]
    #     # Use intent name as a stand-in "reference" text
    #     ref = predicted_intent.split()
    #     bleu = sentence_bleu([ref], xi.split(), weights=(0.5, 0.5))
    #     bleu_scores.append(bleu)

    # print("Average BLEU (approx):", np.mean(bleu_scores))

if __name__ == "__main__":
    evaluate_model('data/train_data.csv')

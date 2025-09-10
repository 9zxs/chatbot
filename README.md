# University FAQ Chatbot

This project implements a simple ML-based FAQ chatbot for a university helpdesk.
- Intent classification: TF-IDF + Logistic Regression
- Response retrieval from pre-defined intents JSON
- Evaluation: Accuracy, Precision, Recall, F1, BLEU

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py data/train_data.csv
python src/app.py
```

Send a request:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"When is the application deadline?"}' http://localhost:5000/chat
```


https://chatbot-mqpgezxnybyno2kz9per3m.streamlit.app/

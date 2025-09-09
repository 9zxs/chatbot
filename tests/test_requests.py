import requests

BASE = "http://localhost:5000/chat"

def test_greet():
    r = requests.post(BASE, json={"text":"hello"})
    d = r.json()
    assert "greet" in d['intent'] or d['confidence']>0.4

def test_deadline():
    r = requests.post(BASE, json={"text":"When is the application deadline?"})
    d = r.json()
    assert d['intent'] == "admissions_deadline"

# Serving The Model - Setup

Minimal setup instructions for the assignment environment.

Prerequisites
- Python 3.10+ (3.12 is available in the venv here)

Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick dataset check
```bash
python check_dataset.py
```

Run FastAPI (after training and saving model)
```bash
uvicorn app.main:app --reload --port 8000
```
